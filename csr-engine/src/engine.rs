use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use rmcp::ServiceExt;
use tokio::sync::RwLock;

use crate::embeddings::EmbeddingEngine;
use crate::import;
use crate::import::watcher::FileWatcher;
use crate::mcp::CsrServer;
use crate::search::SearchEngine;
use crate::storage::Storage;

/// Append a timing line to ~/.claude-self-reflect/hook-timing.log.
fn log_timing(line: &str) {
    crate::telemetry::append_timing_line(line);
}

/// Initial HNSW capacity for an import-only engine's empty in-memory index.
/// Only a construction hint — hnsw_rs grows as needed — sized to comfortably
/// hold one session's worth of new chunks without reallocation.
const IMPORT_ONLY_INDEX_CAPACITY: usize = 4096;

/// Orchestrates all subsystems: storage, embeddings, search, import, and MCP.
#[derive(Clone)]
pub struct Engine {
    storage: Arc<Storage>,
    embeddings: Arc<EmbeddingEngine>,
    search: Arc<RwLock<SearchEngine>>,
    projects_dir: PathBuf,
    index_dir: PathBuf,
    /// When true, the engine never loads the on-disk HNSW cache at construction
    /// and `flush_index` is a no-op. Used by write-only hooks (precompact,
    /// session-end) that import a transcript but never search: they pay neither
    /// the ~O(corpus) `HnswIo::load_hnsw` at startup nor a full re-dump. New
    /// embeddings still land in SQLite; the next search-constructing process
    /// reconciles them via the additive-backfill path in `Engine::new`.
    ///
    /// The flush guard is the critical invariant: without it, dispatch_hook's
    /// unconditional `flush_index` would dump the near-empty in-memory index
    /// with a manifest count equal to the FULL DB count, which the next
    /// `load_from_disk` accepts as current — replacing the real many-node graph
    /// with a handful of points. Recall is not permanently lost (the next
    /// `Engine::new` compares DB ids against the cache and backfills the missing
    /// ones), but recovery costs a full re-insert of the corpus, which defeats
    /// the entire point of the import-only path.
    skip_index_persistence: bool,
}

impl Engine {
    /// Create an engine from pre-built components (for testing).
    pub fn from_parts(
        storage: Arc<Storage>,
        embeddings: Arc<EmbeddingEngine>,
        search: Arc<RwLock<SearchEngine>>,
        projects_dir: PathBuf,
    ) -> Self {
        // Derive index_dir from projects_dir for test constructor
        let index_dir = projects_dir.join("index");
        Self::from_parts_with_index_dir(storage, embeddings, search, projects_dir, index_dir)
    }

    /// Create an engine from pre-built components with an explicit
    /// `index_dir`. Use this whenever the caller knows its real on-disk index
    /// directory (the daemon does) instead of `from_parts`'s
    /// `projects_dir.join("index")`, which is wrong whenever the database and
    /// the projects tree live in different places (they do on a real install).
    pub fn from_parts_with_index_dir(
        storage: Arc<Storage>,
        embeddings: Arc<EmbeddingEngine>,
        search: Arc<RwLock<SearchEngine>>,
        projects_dir: PathBuf,
        index_dir: PathBuf,
    ) -> Self {
        Self {
            storage,
            embeddings,
            search,
            projects_dir,
            index_dir,
            skip_index_persistence: false,
        }
    }

    pub fn new(db_path: &Path, projects_dir: &Path) -> Result<Self> {
        let t0 = std::time::Instant::now();

        tracing::info!(?db_path, "opening storage");
        let storage = Arc::new(Storage::open(db_path)?);
        let t_storage = t0.elapsed();

        tracing::info!("initializing embedding engine");
        let embeddings = Arc::new(EmbeddingEngine::new()?);
        let t_embed = t0.elapsed();

        // Compute index cache directory alongside the database
        let index_dir = db_path.parent().unwrap_or(Path::new(".")).join("index");

        // Fast O(1) counts for staleness check (~1ms)
        let chunk_count = storage.count_chunk_embeddings()?;
        let reflection_count = storage.count_reflection_embeddings()?;
        let t_count = t0.elapsed();

        // Try loading from disk cache first
        let search = if let Some(cached) =
            SearchEngine::load_from_disk(&index_dir, chunk_count, reflection_count)
        {
            // Track the additive-backfill delta and the loaded cache's
            // on-disk age purely as a staleness diagnostic (logged below) —
            // how far behind SQLite the on-disk HNSW cache was at load time,
            // and how long it's been since anything last flushed it. Local
            // to this branch; nothing downstream of `Engine::new` reads
            // either value (there is no cache staleness to report on the
            // rebuild path below — a fresh dump has zero drift and zero age
            // by construction).
            let cache_age = std::fs::metadata(index_dir.join("manifest.json"))
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|modified| std::time::SystemTime::now().duration_since(modified).ok());

            let t_total = t0.elapsed();
            let startup_line = format!(
                "CSR startup: storage={:.0}ms embed={:.0}ms cache_load={:.0}ms total={:.0}ms ({} chunks, cached)",
                t_storage.as_secs_f64() * 1000.0,
                (t_embed - t_storage).as_secs_f64() * 1000.0,
                (t_total - t_count).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
                chunk_count,
            );
            eprintln!("{}", startup_line);
            log_timing(&startup_line);
            tracing::info!(
                chunks = chunk_count,
                reflections = reflection_count,
                "search index loaded from cache"
            );
            // Reconcile HNSW cache with DB:
            // 1. Blank orphan entries (deleted from DB since last dump)
            // 2. Backfill missing entries (added to DB since last dump)
            let mut search = cached;
            let backfill_added = reconcile_search_with_db(
                &storage,
                &mut search,
                storage.load_all_reflection_ids().ok().as_deref(),
                storage.load_all_chunk_ids().ok().as_deref(),
            );
            if backfill_added > 0 {
                tracing::info!(
                    backfill_added,
                    cache_age_secs = cache_age.map(|d| d.as_secs()),
                    "HNSW cache behind DB at startup — additive backfill applied"
                );
            }
            // hnsw_rs cannot delete, so blanked slots stay in the graph until a
            // full rebuild. A foreground process never runs that rebuild here
            // (minutes on a large corpus); it leaves a marker for the daemon's
            // compaction loop or the explicit `rebuild-index` command. Search
            // already overfetches past blanked slots in the meantime.
            let tombstones = search.tombstones();
            if tombstones.material() && crate::search::mark_needs_compaction(&index_dir) {
                log_timing(&format!(
                    "CSR compaction: marked needs_compaction (chunk tombstones {}/{}, reflection tombstones {}/{})",
                    tombstones.chunk_blank,
                    tombstones.chunk_slots,
                    tombstones.reflection_blank,
                    tombstones.reflection_slots,
                ));
            }
            search
        } else {
            // Cache miss — rebuild from SQLite vectors, streaming in bounded
            // batches so peak transient memory is O(batch) instead of
            // O(corpus) (no `load_all_chunk_vectors()` + clone-into-index).
            // The dump below writes a brand-new manifest, so there is no drift
            // and no stale age to report.
            tracing::info!("building search index from stored vectors");

            let (mut search, chunk_total, reflection_total) =
                rebuild_search_index_streaming(&storage, RECONCILE_BATCH)?;
            let t_hnsw = t0.elapsed();

            // Re-query counts right before dump to minimize staleness window (E-1)
            let chunk_count = storage.count_chunk_embeddings().unwrap_or(chunk_count);
            let reflection_count = storage
                .count_reflection_embeddings()
                .unwrap_or(reflection_count);
            // Dump to disk for next startup. A rebuild from the whole of SQLite
            // is compact by construction: publish it as a new compaction epoch
            // and satisfy any pending compaction request.
            search.mark_as_compaction();
            match search.dump_to_disk(&index_dir, chunk_count, reflection_count) {
                Ok(()) => crate::search::clear_needs_compaction(&index_dir),
                Err(e) => tracing::warn!(error = %e, "failed to cache HNSW index (non-fatal)"),
            }
            let t_total = t0.elapsed();

            tracing::info!(
                chunks = chunk_total,
                reflections = reflection_total,
                "search index rebuilt and cached"
            );
            let startup_line = format!(
                "CSR startup: storage={:.0}ms embed={:.0}ms hnsw={:.0}ms dump={:.0}ms total={:.0}ms ({} chunks, rebuilt)",
                t_storage.as_secs_f64() * 1000.0,
                (t_embed - t_storage).as_secs_f64() * 1000.0,
                (t_hnsw - t_count).as_secs_f64() * 1000.0,
                (t_total - t_hnsw).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
                chunk_total,
            );
            eprintln!("{}", startup_line);
            log_timing(&startup_line);

            search
        };

        // Clean stale numbered HNSW files from previous sessions (crash resilience).
        // Needs the exclusive lock to avoid racing concurrent dump_to_disk writers,
        // but only TRY for it: any other process loading the cache holds the lock
        // shared, and a blocking request here stalls this startup until all of them
        // finish. Skipping is safe because every dump_to_disk runs the same cleanup
        // under its own exclusive lock.
        if let Ok(lock_file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(index_dir.join("index.lock"))
        {
            if fs2::FileExt::try_lock_exclusive(&lock_file).is_ok() {
                crate::search::cleanup_stale_index_files(&index_dir);
            }
            // lock released on drop
        }

        // The index is built and reconciled; hand the allocator's freed transient
        // pages back to the OS so this server's steady-state footprint reflects live
        // memory rather than retained slack (macOS-only; no-op elsewhere).
        crate::runtime::release_freed_pages();

        Ok(Self {
            storage,
            embeddings,
            search: Arc::new(RwLock::new(search)),
            projects_dir: projects_dir.to_path_buf(),
            index_dir,
            skip_index_persistence: false,
        })
    }

    /// Construct an engine for write-only hooks (precompact, session-end) that
    /// import a transcript but never search.
    ///
    /// Unlike [`Engine::new`], this skips loading the on-disk HNSW cache
    /// entirely — the `HnswIo::load_hnsw` that walks every point in the graph
    /// (the "setting number of points …" pass, ~O(corpus)). The in-memory index
    /// starts empty and, together with the `flush_index` guard, is never dumped,
    /// so the on-disk cache is left untouched. New embeddings are written to
    /// SQLite; the next search-constructing process picks them up via the
    /// additive-backfill path in [`Engine::new`].
    pub fn new_import_only(db_path: &Path, projects_dir: &Path) -> Result<Self> {
        let storage = Arc::new(Storage::open(db_path)?);
        let embeddings = Arc::new(EmbeddingEngine::new()?);
        let index_dir = db_path.parent().unwrap_or(Path::new(".")).join("index");
        // Small empty index: import inserts land here and are discarded on
        // process exit. Never dumped, so no on-disk cache is touched.
        let search = SearchEngine::new(IMPORT_ONLY_INDEX_CAPACITY);
        Ok(Self {
            storage,
            embeddings,
            search: Arc::new(RwLock::new(search)),
            projects_dir: projects_dir.to_path_buf(),
            index_dir,
            skip_index_persistence: true,
        })
    }

    /// Import conversations from the Claude projects directory.
    /// Uses batch embedding for ~3.4x speedup over single embeds.
    pub async fn import_conversations(&self, limit: Option<usize>) -> Result<usize> {
        let projects = import::discover_projects(&self.projects_dir)?;
        let mut total = 0usize;

        for (dir, project_name) in &projects {
            let files = import::list_jsonl_files(dir)?;
            for file_path in &files {
                if self.storage.is_file_imported(file_path)? {
                    continue;
                }
                total += self.import_file(file_path, project_name).await?;

                if let Some(lim) = limit {
                    if total >= lim {
                        return Ok(total);
                    }
                }
            }
        }

        self.flush_index().await;
        Ok(total)
    }

    /// Import a single JSONL file: parse, embed, store chunks, enrich.
    /// Supports incremental import — when a transcript grows mid-session,
    /// only new chunks are embedded (chunks beyond prev_count are new).
    /// Returns the number of NEW chunks imported (0 if nothing new).
    pub async fn import_file(&self, file_path: &Path, project_name: &str) -> Result<usize> {
        // Check if file is unchanged (mtime match = fully imported, nothing new)
        if self.storage.is_file_imported(file_path)? {
            return Ok(0);
        }

        let chunks = import::parse_jsonl_file(file_path, project_name)?;
        if chunks.is_empty() {
            // Record the skip (agent transcripts, empty conversations) so the
            // watcher doesn't re-parse the file every pass and import_percent
            // counts it as processed instead of silently under-reporting.
            self.storage.mark_file_imported(file_path, 0)?;
            return Ok(0);
        }

        // Incremental: skip chunks we already embedded
        let prev_count = self.storage.get_imported_chunk_count(file_path)?;
        if chunks.len() <= prev_count {
            // File parsed to same/fewer chunks — just update mtime
            self.storage.mark_file_imported(file_path, chunks.len())?;
            return Ok(0);
        }

        let new_chunks = &chunks[prev_count..];
        const BATCH_SIZE: usize = 10;

        for batch in new_chunks.chunks(BATCH_SIZE) {
            let texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();
            let emb = self.embeddings.clone();
            let embeddings = tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                emb.embed(&refs)
            })
            .await??;

            let mut idx = self.search.write().await;
            for (chunk, embedding) in batch.iter().zip(embeddings) {
                self.storage.insert_chunk(chunk, &embedding)?;
                // Persist provenance: who authored this chunk + its source conv.
                // Supersession detection is deferred (None) — recall still gains
                // from author-authority weighting. Non-fatal on error.
                if let Err(e) = self.storage.insert_chunk_provenance(
                    &chunk.id,
                    &crate::provenance::ChunkProvenance {
                        author: chunk.author,
                        source_conv_id: chunk.conversation_id.clone(),
                        supersedes: None,
                    },
                ) {
                    eprintln!("CSR: chunk provenance persist error (non-fatal): {e}");
                }
                idx.insert_chunk(chunk.id.clone(), embedding);
            }
        }

        self.storage.mark_file_imported(file_path, chunks.len())?;

        // Layer 1: Heuristic enrichment only on first import (not incremental updates)
        if prev_count == 0 {
            let conv_id = file_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if !self
                .storage
                .is_conversation_enriched(&conv_id, "heuristic")
                .unwrap_or(false)
            {
                if let Err(e) = crate::extraction::heuristic::enrich_conversation(
                    file_path,
                    &conv_id,
                    project_name,
                    &self.storage,
                    &self.embeddings,
                    &self.search,
                )
                .await
                {
                    tracing::warn!(
                        conv = %conv_id,
                        error = %e,
                        "heuristic enrichment failed (non-fatal)"
                    );
                }
            }
        }

        Ok(new_chunks.len())
    }

    /// Backfill missing import_state rows and run heuristic enrichment for all unenriched conversations.
    /// This repairs the enrichment pipeline when conversations exist in `chunks` but lack import_state entries.
    pub async fn backfill_and_enrich(&self) -> Result<(usize, usize)> {
        // Build conv_id -> (file_path, project_name) map once for both steps
        let projects = import::discover_projects(&self.projects_dir)?;
        let mut conv_to_file: std::collections::HashMap<String, (std::path::PathBuf, String)> =
            std::collections::HashMap::new();

        for (dir, project_name) in &projects {
            if let Ok(files) = import::list_jsonl_files(dir) {
                for file_path in files {
                    let conv_id = file_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    conv_to_file.insert(conv_id, (file_path, project_name.clone()));
                }
            }
        }

        // Step 1: Backfill missing import_state rows
        let missing = self.storage.get_conversations_missing_import_state()?;
        let mut backfilled = 0usize;

        if !missing.is_empty() {
            eprintln!(
                "CSR: found {} conversations missing import_state rows",
                missing.len()
            );

            for conv_id in &missing {
                if let Some((file_path, _)) = conv_to_file.get(conv_id) {
                    if let Err(e) = self.storage.mark_file_imported(file_path, 0) {
                        tracing::warn!(conv = %conv_id, error = %e, "failed to backfill import_state");
                    } else {
                        backfilled += 1;
                    }
                }
            }
            eprintln!("CSR: backfilled {} import_state rows", backfilled);
        }

        // Step 2: Heuristic enrichment for all unenriched conversations
        let needing = self.storage.get_conversations_needing_heuristic()?;
        let mut enriched = 0usize;

        if !needing.is_empty() {
            eprintln!(
                "CSR: found {} conversations needing heuristic enrichment",
                needing.len()
            );

            for (conv_id, project_name) in &needing {
                if let Some((file_path, _)) = conv_to_file.get(conv_id) {
                    if let Err(e) = crate::extraction::heuristic::enrich_conversation(
                        file_path,
                        conv_id,
                        project_name,
                        &self.storage,
                        &self.embeddings,
                        &self.search,
                    )
                    .await
                    {
                        tracing::warn!(
                            conv = %conv_id,
                            error = %e,
                            "heuristic enrichment failed (non-fatal)"
                        );
                    } else {
                        enriched += 1;
                        if enriched.is_multiple_of(100) {
                            eprintln!(
                                "CSR: enriched {}/{} conversations...",
                                enriched,
                                needing.len()
                            );
                            self.flush_index().await;
                        }
                    }
                }
            }
            eprintln!(
                "CSR: heuristic enrichment complete: {}/{}",
                enriched,
                needing.len()
            );
        }

        // Final flush to persist any remaining vectors
        self.flush_index().await;

        Ok((backfilled, enriched))
    }

    // ─── Accessors for hooks ───

    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    pub fn embeddings(&self) -> &Arc<EmbeddingEngine> {
        &self.embeddings
    }

    pub fn search(&self) -> &Arc<RwLock<SearchEngine>> {
        &self.search
    }

    pub fn projects_dir(&self) -> &Path {
        &self.projects_dir
    }

    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// Rebuild the whole HNSW index from the vectors in SQLite, publish it as
    /// a new compaction epoch, and swap it in as this engine's live index.
    ///
    /// hnsw_rs cannot delete, so blanked (tombstoned) slots accumulate until a
    /// full rebuild. That rebuild takes minutes on a large corpus, so it never
    /// runs in a foreground `Engine::new`: those leave a `needs_compaction`
    /// marker, and the daemon's compaction loop (or the explicit
    /// `rebuild-index` command) calls this.
    ///
    /// The stream and the publish run on the blocking pool, outside the live
    /// index's lock, so searches keep serving the old index meanwhile. Rows
    /// that land or vanish while the stream runs are reconciled against the
    /// DB twice: once before publishing, and once more under the write lock
    /// just before the swap (ids are loaded before taking the lock).
    ///
    /// Returns `Ok(None)` when `cancel` is set before the stream finishes:
    /// nothing is published and the live index is untouched.
    pub async fn rebuild_search_index(
        &self,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<Option<RebuildReport>> {
        let t_build = std::time::Instant::now();
        let storage = self.storage.clone();
        let cancel_build = cancel.clone();
        let built = tokio::task::spawn_blocking(move || {
            let Some((mut fresh, chunks, reflections)) =
                rebuild_search_index_streaming_cancellable(
                    &storage,
                    RECONCILE_BATCH,
                    cancel_build.as_deref(),
                )?
            else {
                return Ok::<_, anyhow::Error>(None);
            };
            reconcile_search_with_db(
                &storage,
                &mut fresh,
                storage.load_all_reflection_ids().ok().as_deref(),
                storage.load_all_chunk_ids().ok().as_deref(),
            );
            Ok(Some((fresh, chunks, reflections)))
        })
        .await
        .map_err(|e| anyhow::anyhow!("index rebuild task failed: {e}"))??;
        let Some((mut fresh, chunks, reflections)) = built else {
            return Ok(None);
        };
        if is_cancelled(cancel.as_deref()) {
            return Ok(None);
        }
        let build = t_build.elapsed();

        let t_publish = std::time::Instant::now();
        if !self.skip_index_persistence {
            fresh.mark_as_compaction();
            let storage = self.storage.clone();
            let index_dir = self.index_dir.clone();
            fresh = tokio::task::spawn_blocking(move || {
                let chunk_count = storage.count_chunk_embeddings()?;
                let reflection_count = storage.count_reflection_embeddings()?;
                fresh.dump_to_disk(&index_dir, chunk_count, reflection_count)?;
                Ok::<_, anyhow::Error>(fresh)
            })
            .await
            .map_err(|e| anyhow::anyhow!("index publish task failed: {e}"))??;
        }
        let publish = t_publish.elapsed();

        let storage = self.storage.clone();
        let (reflection_ids, chunk_ids) = tokio::task::spawn_blocking(move || {
            (
                storage.load_all_reflection_ids().ok(),
                storage.load_all_chunk_ids().ok(),
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("index id probe task failed: {e}"))?;
        let epoch = fresh.compaction_epoch();
        {
            let mut live = self.search.write().await;
            reconcile_search_with_db(
                &self.storage,
                &mut fresh,
                reflection_ids.as_deref(),
                chunk_ids.as_deref(),
            );
            *live = fresh;
        }
        crate::search::clear_needs_compaction(&self.index_dir);
        crate::runtime::release_freed_pages();
        Ok(Some(RebuildReport {
            chunks,
            reflections,
            build,
            publish,
            epoch,
        }))
    }

    /// Flush the HNSW index to disk if it has been modified.
    /// Safe to call multiple times — skips if not dirty.
    pub async fn flush_index(&self) {
        // Write-only engines (import-only hooks) never persist the index. The
        // in-memory index holds at most this session's chunks against an empty
        // base; dumping it would overwrite the real on-disk cache with a
        // near-empty graph carrying a full-count manifest. The next load would
        // accept it as current and (via the id-backfill in `Engine::new`)
        // recover only by re-inserting the whole corpus — the expensive rebuild
        // this path exists to avoid. See `skip_index_persistence`.
        if self.skip_index_persistence {
            return;
        }
        let mut idx = self.search.write().await;
        if idx.is_dirty() {
            // Query current DB counts for staleness-correct manifest
            let chunk_count = self.storage.count_chunk_embeddings().unwrap_or(0);
            let refl_count = self.storage.count_reflection_embeddings().unwrap_or(0);
            if let Err(e) = idx.dump_to_disk(&self.index_dir, chunk_count, refl_count) {
                tracing::warn!(error = %e, "failed to flush HNSW index (non-fatal)");
            }
        }
    }

    /// Start the file system watcher as a background task.
    /// Returns a JoinHandle that can be awaited or dropped.
    pub fn start_watcher(&self) -> tokio::task::JoinHandle<()> {
        let watcher = FileWatcher::new(
            self.projects_dir.clone(),
            self.storage.clone(),
            self.embeddings.clone(),
            self.search.clone(),
            self.index_dir.to_path_buf(),
        );
        watcher.spawn()
    }

    /// Start the MCP server on stdio with background enrichment loops.
    ///
    /// `stdout` is fd 1 already claimed for the JSON-RPC transport by
    /// `main.rs`, before this engine's own construction had a chance to run
    /// `hnsw_rs` through its index rebuild path. `Some` on unix routes the
    /// transport's write half through that claimed descriptor instead of
    /// `rmcp::transport::io::stdio()`'s own fd 1, so anything the process
    /// still writes to fd 1 directly (the enrichment loops and `--watch`
    /// importer started below, both of which insert into the same HNSW index)
    /// lands on stderr rather than corrupting the protocol stream. `None`
    /// (claim failed, or non-unix) falls back to the unmodified stdio
    /// transport — the pre-existing behaviour.
    pub async fn serve_mcp(self, stdout: Option<crate::hooks::ClaimedStdout>) -> Result<()> {
        // Spawn enrichment loops as background tasks (extraction, narrator, consolidation)
        let enrichment_handles = crate::daemon::spawn_enrichment_loops(
            self.storage.clone(),
            self.embeddings.clone(),
            self.search.clone(),
        );

        let server = CsrServer::new(
            self.storage,
            self.embeddings,
            self.search,
            self.projects_dir,
            self.index_dir,
        );
        #[cfg(unix)]
        let service = match stdout {
            Some(claimed) => {
                server
                    .serve((tokio::io::stdin(), claimed.into_async_writer()))
                    .await?
            }
            None => server.serve(rmcp::transport::io::stdio()).await?,
        };
        #[cfg(not(unix))]
        let service = {
            let _ = stdout;
            server.serve(rmcp::transport::io::stdio()).await?
        };
        service.waiting().await?;

        // Clean up enrichment loops on MCP server exit
        for handle in enrichment_handles {
            handle.abort();
        }

        Ok(())
    }

    /// Start the MCP server on Streamable HTTP instead of stdio, with the
    /// same background enrichment loops.
    ///
    /// stdio gives every Claude Code session its own process and its own copy
    /// of the index. This path is the other arrangement: one process holds
    /// one loaded index and answers every session that registers
    /// `http://<addr>/mcp`. No stdout claim is needed here: JSON-RPC travels
    /// over HTTP, so a stray library print to fd 1 cannot corrupt it.
    pub async fn serve_mcp_http(self, addr: std::net::SocketAddr) -> Result<()> {
        let enrichment_handles = crate::daemon::spawn_enrichment_loops(
            self.storage.clone(),
            self.embeddings.clone(),
            self.search.clone(),
        );

        let server = CsrServer::new(
            self.storage,
            self.embeddings,
            self.search,
            self.projects_dir,
            self.index_dir,
        );

        let listener = crate::mcp::http::bind(addr).await?;
        let bound = listener.local_addr()?;
        eprintln!("{}", crate::mcp::http::ready_line(bound));
        let result = crate::mcp::http::serve(server, listener).await;

        for handle in enrichment_handles {
            handle.abort();
        }

        result
    }
}

/// Batch size for reconciling the HNSW cache against SQLite: both the
/// additive-drift backfill (missing ids only) and the cache-miss full
/// rebuild fetch vectors this many ids at a time, so peak transient memory
/// is O(batch) instead of O(corpus). See src/engine.rs:126-165 history.
const RECONCILE_BATCH: usize = 2_000;

/// Fetch and insert only the given missing chunk ids, in bounded batches.
/// Never materializes more than `batch_size` vectors at once. Returns the
/// number of points actually inserted.
fn backfill_missing_chunks_batched(
    storage: &Storage,
    search: &mut SearchEngine,
    missing_ids: &[String],
    batch_size: usize,
) -> Result<usize> {
    let mut added = 0;
    for batch in missing_ids.chunks(batch_size.max(1)) {
        let vecs = storage.get_chunk_vectors_by_ids(batch)?;
        added += vecs.len();
        for (id, vec) in vecs {
            search.insert_chunk(id, vec);
        }
    }
    Ok(added)
}

pub(crate) fn backfill_missing_chunks(
    storage: &Storage,
    search: &mut SearchEngine,
    missing_ids: &[String],
) -> Result<usize> {
    backfill_missing_chunks_batched(storage, search, missing_ids, RECONCILE_BATCH)
}

/// Reflection counterpart of [`backfill_missing_chunks_batched`].
fn backfill_missing_reflections_batched(
    storage: &Storage,
    search: &mut SearchEngine,
    missing_ids: &[String],
    batch_size: usize,
) -> Result<usize> {
    let mut added = 0;
    for batch in missing_ids.chunks(batch_size.max(1)) {
        let vecs = storage.get_reflection_vectors_by_ids(batch)?;
        added += vecs.len();
        for (id, vec) in vecs {
            search.insert_reflection(id, vec);
        }
    }
    Ok(added)
}

fn backfill_missing_reflections(
    storage: &Storage,
    search: &mut SearchEngine,
    missing_ids: &[String],
) -> Result<usize> {
    backfill_missing_reflections_batched(storage, search, missing_ids, RECONCILE_BATCH)
}

/// Reconcile a loaded or freshly rebuilt index with the DB ids: blank ids the
/// DB no longer has and backfill ids the index lacks, for both sides. Returns
/// how many entries were backfilled. A side whose id probe failed (`None`) is
/// left as it is.
fn reconcile_search_with_db(
    storage: &Storage,
    search: &mut SearchEngine,
    db_reflection_ids: Option<&[String]>,
    db_chunk_ids: Option<&[String]>,
) -> usize {
    let mut backfill_added = 0usize;
    if let Some(db_ids) = db_reflection_ids {
        let db_id_set: std::collections::HashSet<&str> =
            db_ids.iter().map(|s| s.as_str()).collect();
        let blanked = search.blank_orphan_reflections(&db_id_set);
        if blanked > 0 {
            tracing::info!(blanked, "blanked orphan reflection entries in HNSW cache");
        }

        // Backfill reflections that exist in DB but not in HNSW.
        // Fetch only the missing ids, in bounded batches — never the
        // whole reflection table.
        let missing: Vec<String> = db_id_set
            .iter()
            .filter(|id| !search.has_reflection(id))
            .map(|s| s.to_string())
            .collect();
        if !missing.is_empty() {
            if let Ok(added) = backfill_missing_reflections(storage, search, &missing) {
                if added > 0 {
                    tracing::info!(added, "backfilled missing reflections into HNSW cache");
                }
                backfill_added += added;
            }
        }
    }

    // Backfill chunks added to DB since the last dump (additive drift).
    // Cheap ID probe first; only fetch the vectors that are actually
    // missing (in bounded batches) if something is missing. This is what
    // lets a stale-but-additive cache load in ~ms instead of triggering a
    // full HNSW rebuild (~tens of seconds), while keeping peak transient
    // memory O(batch) rather than O(corpus).
    if let Some(db_chunk_ids) = db_chunk_ids {
        let missing: Vec<String> = db_chunk_ids
            .iter()
            .filter(|id| !search.has_chunk(id))
            .cloned()
            .collect();
        let mut chunks_added = 0usize;
        if !missing.is_empty() {
            if let Ok(added) = backfill_missing_chunks(storage, search, &missing) {
                if added > 0 {
                    tracing::info!(added, "backfilled missing chunks into HNSW cache");
                }
                chunks_added = added;
                backfill_added += added;
            }
        }
        // Blank chunks the DB has deleted since the last dump (negative
        // drift; `load_from_disk` no longer rebuilds for it). After the
        // backfill, the DB ids live in the index number
        // `db - missing + added`, so a larger live count means the cache
        // holds orphans. The O(1) gate keeps the common no-orphan load
        // from building a corpus-sized id set on every hook.
        let db_ids_live = db_chunk_ids.len() - missing.len() + chunks_added;
        if search.live_chunk_count() > db_ids_live {
            let db_id_set: std::collections::HashSet<&str> =
                db_chunk_ids.iter().map(|s| s.as_str()).collect();
            let blanked = search.blank_orphan_chunks(&db_id_set);
            if blanked > 0 {
                tracing::info!(blanked, "blanked orphan chunk entries in HNSW cache");
            }
        }
    }
    backfill_added
}

/// Rebuild the HNSW index from SQLite, streaming vectors in bounded batches
/// instead of materializing `load_all_chunk_vectors()` (the whole corpus) as
/// one `Vec<(String, Vec<f32>)>` before inserting. Cheap id probes
/// (`load_all_chunk_ids` / `load_all_reflection_ids`) drive the batching;
/// each batch's vectors are moved (not cloned) into the index and dropped
/// before the next batch is fetched. Returns the built index plus the
/// (chunks, reflections) counts actually inserted.
fn rebuild_search_index_streaming(
    storage: &Storage,
    batch_size: usize,
) -> Result<(SearchEngine, usize, usize)> {
    rebuild_search_index_streaming_cancellable(storage, batch_size, None)?
        .ok_or_else(|| anyhow::anyhow!("index rebuild cancelled"))
}

fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|flag| flag.load(Ordering::SeqCst))
}

/// [`rebuild_search_index_streaming`] that checks `cancel` before every batch
/// and returns `Ok(None)` once it is set, so a shutdown waits for at most one
/// batch instead of a whole multi-minute rebuild.
fn rebuild_search_index_streaming_cancellable(
    storage: &Storage,
    batch_size: usize,
    cancel: Option<&AtomicBool>,
) -> Result<Option<(SearchEngine, usize, usize)>> {
    let chunk_ids = storage.load_all_chunk_ids()?;
    let reflection_ids = storage.load_all_reflection_ids()?;

    let estimated_size = (chunk_ids.len() + 1000).max(10_000);
    let mut search = SearchEngine::new(estimated_size);

    let mut chunk_total = 0usize;
    for batch in chunk_ids.chunks(batch_size.max(1)) {
        if is_cancelled(cancel) {
            return Ok(None);
        }
        let vecs = storage.get_chunk_vectors_by_ids(batch)?;
        chunk_total += vecs.len();
        for (id, vec) in vecs {
            search.insert_chunk(id, vec);
        }
    }

    let mut reflection_total = 0usize;
    for batch in reflection_ids.chunks(batch_size.max(1)) {
        if is_cancelled(cancel) {
            return Ok(None);
        }
        let vecs = storage.get_reflection_vectors_by_ids(batch)?;
        reflection_total += vecs.len();
        for (id, vec) in vecs {
            search.insert_reflection(id, vec);
        }
    }

    Ok(Some((search, chunk_total, reflection_total)))
}

/// What [`Engine::rebuild_search_index`] built and how long each phase took.
#[derive(Debug, Clone, Copy)]
pub struct RebuildReport {
    pub chunks: usize,
    pub reflections: usize,
    /// Streaming the vectors out of SQLite into a new HNSW graph.
    pub build: std::time::Duration,
    /// Writing the new generation and manifest under the exclusive index lock.
    pub publish: std::time::Duration,
    /// The compaction epoch the new manifest carries.
    pub epoch: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::ConversationChunk;
    use crate::provenance::Speaker;

    fn insert_test_chunk(storage: &Storage, id: &str) {
        storage
            .insert_chunk(
                &ConversationChunk {
                    id: id.into(),
                    conversation_id: "conv".into(),
                    project_name: "proj".into(),
                    timestamp: "2026-09-08T00:00:00Z".into(),
                    content: "x".into(),
                    message_count: 1,
                    summary: None,
                    author: Speaker::User,
                    seq: 0,
                    is_sidechain: false,
                },
                &[0.1_f32; 384],
            )
            .unwrap();
    }

    #[test]
    fn backfill_missing_chunks_inserts_exactly_the_missing_ids_in_batches() {
        let storage = Storage::open_memory().unwrap();
        for i in 0..10 {
            insert_test_chunk(&storage, &format!("chunk-{i}"));
        }
        let mut search = SearchEngine::new(100);
        // Pre-seed half of them as already-present — these must be skipped,
        // not re-fetched (proves we fetch only the missing ids, not the corpus).
        for i in 0..5 {
            search.insert_chunk(format!("chunk-{i}"), vec![0.1_f32; 384]);
        }

        let db_ids = storage.load_all_chunk_ids().unwrap();
        let missing: Vec<String> = db_ids
            .into_iter()
            .filter(|id| !search.has_chunk(id))
            .collect();
        assert_eq!(missing.len(), 5, "sanity: half should be missing");

        // batch_size=2 over 5 missing ids forces 3 separate batched lookups
        // (2, 2, 1) — exercises the bounded-batch path, not a single big fetch.
        let added = backfill_missing_chunks_batched(&storage, &mut search, &missing, 2).unwrap();

        assert_eq!(
            added, 5,
            "must insert exactly the missing ids, not the whole corpus"
        );
        for i in 5..10 {
            assert!(search.has_chunk(&format!("chunk-{i}")));
        }
    }

    #[test]
    fn backfill_missing_reflections_inserts_exactly_the_missing_ids_in_batches() {
        let storage = Storage::open_memory().unwrap();
        for i in 0..7 {
            storage
                .insert_reflection(&format!("refl-{i}"), "content", &[], &[0.2_f32; 384])
                .unwrap();
        }
        let mut search = SearchEngine::new(100);
        for i in 0..3 {
            search.insert_reflection(format!("refl-{i}"), vec![0.2_f32; 384]);
        }

        let db_ids = storage.load_all_reflection_ids().unwrap();
        let missing: Vec<String> = db_ids
            .into_iter()
            .filter(|id| !search.has_reflection(id))
            .collect();
        assert_eq!(missing.len(), 4, "sanity: 4 should be missing");

        let added =
            backfill_missing_reflections_batched(&storage, &mut search, &missing, 3).unwrap();

        assert_eq!(added, 4);
        for i in 3..7 {
            assert!(search.has_reflection(&format!("refl-{i}")));
        }
    }

    #[test]
    fn rebuild_search_index_streaming_inserts_all_corpus_points_via_batches() {
        let storage = Storage::open_memory().unwrap();
        for i in 0..7 {
            insert_test_chunk(&storage, &format!("chunk-{i}"));
        }
        for i in 0..3 {
            storage
                .insert_reflection(&format!("refl-{i}"), "content", &[], &[0.2_f32; 384])
                .unwrap();
        }

        // batch_size=2 forces multiple batches for both chunks (7) and
        // reflections (3) — proves the rebuild path never needs a single
        // corpus-sized fetch to produce a correct index.
        let (search, chunk_total, reflection_total) =
            rebuild_search_index_streaming(&storage, 2).unwrap();

        assert_eq!(chunk_total, 7);
        assert_eq!(reflection_total, 3);
        for i in 0..7 {
            assert!(search.has_chunk(&format!("chunk-{i}")));
        }
        for i in 0..3 {
            assert!(search.has_reflection(&format!("refl-{i}")));
        }
    }

    fn insert_test_chunk_in(storage: &Storage, id: &str, conversation_id: &str) {
        storage
            .insert_chunk(
                &ConversationChunk {
                    id: id.into(),
                    conversation_id: conversation_id.into(),
                    project_name: "proj".into(),
                    timestamp: "2026-09-08T00:00:00Z".into(),
                    content: "x".into(),
                    message_count: 1,
                    summary: None,
                    author: Speaker::User,
                    seq: 0,
                    is_sidechain: false,
                },
                &[0.1_f32; 384],
            )
            .unwrap();
    }

    /// The DB can hold fewer rows than the published manifest describes: plan
    /// reimport deletes before it re-embeds, reflection supersession deletes the
    /// old layer, and a manifest's counts are captured independently of the
    /// in-memory id maps, so a concurrent insert-then-delete can land between
    /// them. Every process that constructed an engine inside that window used to
    /// rebuild the whole corpus (805k vectors on a real install: minutes of CPU
    /// per hook, 20+ stacked hooks, none finishing). It must instead reuse the
    /// cache and blank the vanished ids.
    #[tokio::test]
    async fn negative_drift_reuses_cache_and_blanks_orphans_instead_of_rebuilding() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.db");
        let projects_dir = tmp.path().join("projects");
        std::fs::create_dir_all(&projects_dir).unwrap();
        let index_dir = tmp.path().join("index");

        {
            let storage = Storage::open(&db_path).unwrap();
            for i in 0..10 {
                insert_test_chunk_in(&storage, &format!("keep-{i}"), "conv-keep");
            }
            for i in 0..2 {
                insert_test_chunk_in(&storage, &format!("gone-{i}"), "plan:gone");
            }
            for i in 0..3 {
                storage
                    .insert_reflection(&format!("refl-{i}"), "content", &[], &[0.2_f32; 384])
                    .unwrap();
            }
        }

        // No cache yet: the first construction builds one and publishes it.
        drop(Engine::new(&db_path, &projects_dir).unwrap());
        let manifest_before = std::fs::read(index_dir.join("manifest.json"))
            .expect("first Engine::new must publish a manifest");

        // Rows vanish after the manifest was published.
        {
            let storage = Storage::open(&db_path).unwrap();
            storage.delete_chunks_for_conversation("plan:gone").unwrap();
            storage.delete_reflection("refl-0").unwrap();
        }

        let eng = Engine::new(&db_path, &projects_dir).unwrap();

        // Receipt that the cache was reused: the rebuild path always publishes
        // a new manifest (new generation basename, new created_at).
        let manifest_after = std::fs::read(index_dir.join("manifest.json")).unwrap();
        assert!(
            manifest_before == manifest_after,
            "negative drift must reuse the published cache, not rebuild and republish it"
        );

        let search = eng.search.read().await;
        for i in 0..2 {
            assert!(
                !search.has_chunk(&format!("gone-{i}")),
                "a chunk deleted from the DB must be blanked on load"
            );
        }
        for i in 0..10 {
            assert!(search.has_chunk(&format!("keep-{i}")));
        }
        assert!(
            !search.has_reflection("refl-0"),
            "a reflection deleted from the DB must be blanked on load"
        );
        assert!(search.has_reflection("refl-1") && search.has_reflection("refl-2"));
        assert_eq!(search.reflection_count(), 2);

        let hits = search.search_chunks(&[0.1_f32; 384], 20, 0.0);
        assert!(
            hits.iter().all(|h| !h.id.starts_with("gone-")),
            "blanked chunks must never be returned by search"
        );
        assert_eq!(hits.len(), 10, "every live chunk is still returned");
    }

    /// Temp DB + projects dir; the index lives at `<tmp>/index`, beside the DB,
    /// exactly where `Engine::new` derives it.
    struct Corpus {
        _tmp: tempfile::TempDir,
        db_path: std::path::PathBuf,
        projects_dir: std::path::PathBuf,
        index_dir: std::path::PathBuf,
    }

    impl Corpus {
        fn new() -> Self {
            let tmp = tempfile::TempDir::new().unwrap();
            let db_path = tmp.path().join("test.db");
            let projects_dir = tmp.path().join("projects");
            std::fs::create_dir_all(&projects_dir).unwrap();
            let index_dir = tmp.path().join("index");
            Self {
                _tmp: tmp,
                db_path,
                projects_dir,
                index_dir,
            }
        }

        fn seed(&self, conversation_id: &str, prefix: &str, n: usize) {
            let storage = Storage::open(&self.db_path).unwrap();
            for i in 0..n {
                insert_test_chunk_in(&storage, &format!("{prefix}-{i}"), conversation_id);
            }
        }

        fn delete_conversation(&self, conversation_id: &str) {
            Storage::open(&self.db_path)
                .unwrap()
                .delete_chunks_for_conversation(conversation_id)
                .unwrap();
        }

        fn engine(&self) -> Engine {
            Engine::new(&self.db_path, &self.projects_dir).unwrap()
        }

        fn manifest_bytes(&self) -> Vec<u8> {
            std::fs::read(self.index_dir.join("manifest.json")).unwrap()
        }
    }

    #[tokio::test]
    async fn engine_new_marks_compaction_only_once_tombstones_are_material() {
        let c = Corpus::new();
        c.seed("conv-keep", "keep", 39);
        c.seed("plan:one", "one", 1);
        c.seed("plan:four", "four", 4);
        drop(c.engine()); // no cache yet: builds and publishes one (44 slots)

        // 1 blank of 44 slots is under the 1-in-20 threshold.
        c.delete_conversation("plan:one");
        drop(c.engine());
        assert!(
            !crate::search::needs_compaction(&c.index_dir),
            "a non-material tombstone share must not request a compaction"
        );

        // 5 of 44 is over it. Engine::new never flushes, so the published
        // manifest still lists every id and both deletions blank here.
        c.delete_conversation("plan:four");
        drop(c.engine());
        assert!(
            crate::search::needs_compaction(&c.index_dir),
            "a material tombstone share must request a compaction"
        );
    }

    /// The reason the compaction guard exists: a process that loaded the index
    /// before a compaction must not publish its pre-compaction maps over the
    /// compacted ones afterward. Without the guard it puts every blank slot
    /// back, the next load requests another compaction, and the daemon
    /// rebuilds the whole corpus again.
    #[tokio::test]
    async fn compaction_survives_a_process_that_loaded_the_pre_compaction_index() {
        let c = Corpus::new();
        c.seed("conv-keep", "keep", 10);
        c.seed("plan:gone", "gone", 30);
        drop(c.engine());

        c.delete_conversation("plan:gone");
        let stale = c.engine(); // loads 40 slots, blanks 30: material
        assert!(crate::search::needs_compaction(&c.index_dir));

        let report = c
            .engine()
            .rebuild_search_index(None)
            .await
            .unwrap()
            .expect("an uncancelled rebuild completes");
        assert_eq!(report.chunks, 10);
        assert!(
            !crate::search::needs_compaction(&c.index_dir),
            "a completed compaction clears the marker"
        );
        let compacted = crate::search::on_disk_tombstones(&c.index_dir).unwrap();
        assert_eq!(compacted.chunk_blank, 0);
        assert_eq!(compacted.chunk_slots, 10);

        // The stale process saves (its blanking left it dirty).
        stale.flush_index().await;
        let after = crate::search::on_disk_tombstones(&c.index_dir).unwrap();
        assert_eq!(
            (after.chunk_blank, after.chunk_slots),
            (0, 10),
            "a process that loaded the pre-compaction index must not republish its tombstones"
        );

        drop(c.engine());
        assert!(
            !crate::search::needs_compaction(&c.index_dir),
            "the next load must not request another compaction"
        );
    }

    #[tokio::test]
    async fn cancelled_rebuild_publishes_nothing_and_keeps_the_live_index() {
        let c = Corpus::new();
        c.seed("conv-keep", "keep", 10);
        let eng = c.engine();
        crate::search::mark_needs_compaction(&c.index_dir);
        let before = c.manifest_bytes();

        let cancel = Arc::new(AtomicBool::new(true));
        assert!(eng
            .rebuild_search_index(Some(cancel))
            .await
            .unwrap()
            .is_none());

        assert!(
            c.manifest_bytes() == before,
            "a cancelled rebuild must not publish"
        );
        assert!(
            crate::search::needs_compaction(&c.index_dir),
            "a cancelled rebuild leaves the request for the next pass"
        );
        assert_eq!(eng.search.read().await.live_chunk_count(), 10);
    }

    /// A rebuild swaps in an index that also holds rows imported while it ran:
    /// the reconcile passes backfill from SQLite whatever the stream missed.
    #[tokio::test]
    async fn rebuild_swaps_in_rows_the_live_index_took_after_the_stream() {
        let c = Corpus::new();
        c.seed("conv-keep", "keep", 10);
        let eng = c.engine();
        // Lands in SQLite after the engine loaded; a rebuild must pick it up.
        c.seed("conv-late", "late", 3);
        let report = eng.rebuild_search_index(None).await.unwrap().unwrap();
        assert_eq!(report.chunks, 13);
        let live = eng.search.read().await;
        for i in 0..3 {
            assert!(live.has_chunk(&format!("late-{i}")));
        }
        assert_eq!(live.tombstones().chunk_blank, 0);
    }

    #[tokio::test]
    async fn compaction_tick_rebuilds_only_a_marked_and_still_material_index() {
        use crate::daemon::{compaction_tick, CompactionOutcome};
        let c = Corpus::new();
        c.seed("conv-keep", "keep", 10);
        c.seed("plan:gone", "gone", 30);
        let shutdown = Arc::new(AtomicBool::new(false));

        let eng = c.engine();
        assert!(matches!(
            compaction_tick(&eng, &shutdown).await,
            CompactionOutcome::NoMarker
        ));

        // A marker over a published index that is not material: cleared, no rebuild.
        crate::search::mark_needs_compaction(&c.index_dir);
        let before = c.manifest_bytes();
        assert!(matches!(
            compaction_tick(&eng, &shutdown).await,
            CompactionOutcome::StaleMarkerCleared(_)
        ));
        assert!(!crate::search::needs_compaction(&c.index_dir));
        assert!(
            c.manifest_bytes() == before,
            "a stale marker must not rebuild"
        );

        // Material and marked: rebuilt and cleared. A stop hook's save makes
        // the blanks part of the published index, as it does on a real install.
        c.delete_conversation("plan:gone");
        let hook = c.engine();
        assert!(crate::search::needs_compaction(&c.index_dir));
        hook.flush_index().await;
        match compaction_tick(&eng, &shutdown).await {
            CompactionOutcome::Rebuilt(report) => assert_eq!(report.chunks, 10),
            other => panic!("expected a rebuild, got {other:?}"),
        }
        assert!(!crate::search::needs_compaction(&c.index_dir));
        assert_eq!(
            crate::search::on_disk_tombstones(&c.index_dir)
                .unwrap()
                .chunk_blank,
            0
        );
    }
}
