use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use rmcp::ServiceExt;
use tokio::sync::RwLock;

use crate::embeddings::EmbeddingEngine;
use crate::import;
use crate::import::watcher::FileWatcher;
use crate::mcp::CsrServer;
use crate::search::{SearchEngine, VectorStore};
use crate::storage::{Storage, VectorTable};

/// Append a timing line to ~/.claude-self-reflect/hook-timing.log.
fn log_timing(line: &str) {
    crate::telemetry::append_timing_line(line);
}

/// How long [`Engine::sync_vectors`] waits before re-checking a side whose row
/// count disagrees with SQLite. The daemon's own writes touch SQLite and then
/// the store a moment later, so a single mismatch is usually that gap.
const COUNT_RECHECK_DELAY: std::time::Duration = std::time::Duration::from_secs(2);

/// Highest SQLite rowid of each embedding table already applied to the
/// in-memory stores. Rows above it were written after the last load or sync.
#[derive(Default)]
struct Watermarks {
    chunks: AtomicI64,
    reflections: AtomicI64,
}

impl Watermarks {
    fn of(&self, table: VectorTable) -> &AtomicI64 {
        match table {
            VectorTable::Chunks => &self.chunks,
            VectorTable::Reflections => &self.reflections,
        }
    }
}

/// What [`Engine::sync_vectors`] changed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VectorSyncReport {
    pub chunks_added: usize,
    pub reflections_added: usize,
    pub chunks_reloaded: bool,
    pub reflections_reloaded: bool,
}

impl VectorSyncReport {
    pub fn changed(&self) -> bool {
        self.chunks_added + self.reflections_added > 0
            || self.chunks_reloaded
            || self.reflections_reloaded
    }
}

/// Orchestrates all subsystems: storage, embeddings, search, import, and MCP.
///
/// SQLite is the only thing on disk. The vectors live in memory in one
/// process (the daemon), which builds them from SQLite at startup.
#[derive(Clone)]
pub struct Engine {
    storage: Arc<Storage>,
    embeddings: Arc<EmbeddingEngine>,
    search: Arc<RwLock<SearchEngine>>,
    projects_dir: PathBuf,
    watermarks: Arc<Watermarks>,
    /// False only between [`Engine::new_unloaded`] and the end of the first
    /// [`Engine::load_vectors`].
    vectors_ready: Arc<AtomicBool>,
}

fn side_mut(idx: &mut SearchEngine, table: VectorTable) -> &mut VectorStore {
    match table {
        VectorTable::Chunks => idx.chunks_mut(),
        VectorTable::Reflections => idx.reflections_mut(),
    }
}

fn side(idx: &SearchEngine, table: VectorTable) -> &VectorStore {
    match table {
        VectorTable::Chunks => idx.chunks(),
        VectorTable::Reflections => idx.reflections(),
    }
}

/// Read every vector of `table` into a new store. Returns it with the highest
/// rowid read.
fn load_side(storage: &Storage, table: VectorTable) -> Result<(VectorStore, i64)> {
    let expected = storage.count_vectors(table).unwrap_or(0);
    let mut store = VectorStore::with_capacity(expected + expected / 16);
    let max = storage.stream_vectors_after(table, 0, |id, v| {
        store.upsert(id.to_owned(), v);
    })?;
    Ok((store, max))
}

impl Engine {
    /// Create an engine from pre-built components (for testing). The search
    /// engine is used as given; nothing is loaded from SQLite.
    pub fn from_parts(
        storage: Arc<Storage>,
        embeddings: Arc<EmbeddingEngine>,
        search: Arc<RwLock<SearchEngine>>,
        projects_dir: PathBuf,
    ) -> Self {
        Self {
            storage,
            embeddings,
            search,
            projects_dir,
            watermarks: Arc::new(Watermarks::default()),
            vectors_ready: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Open storage and load every vector before returning. For one-shot CLI
    /// commands and the stdio MCP server, which own their engine for one run.
    pub fn new(db_path: &Path, projects_dir: &Path) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let mut engine = Self::new_unloaded(db_path, projects_dir)?;
        let (chunks, cmax) = load_side(&engine.storage, VectorTable::Chunks)?;
        let (reflections, rmax) = load_side(&engine.storage, VectorTable::Reflections)?;
        let idx = Arc::get_mut(&mut engine.search)
            .expect("a new engine's search index is not shared yet")
            .get_mut();
        *idx.chunks_mut() = chunks;
        *idx.reflections_mut() = reflections;
        engine.watermarks.chunks.store(cmax, Ordering::Release);
        engine.watermarks.reflections.store(rmax, Ordering::Release);
        engine.vectors_ready.store(true, Ordering::Release);
        tracing::info!(
            chunks = idx.chunk_count(),
            reflections = idx.reflection_count(),
            ms = t0.elapsed().as_millis() as u64,
            "vectors loaded"
        );
        Ok(engine)
    }

    /// Open storage and load only the reflection vectors (about 14k, versus
    /// 900k chunks). For `generate-story`, a short-lived process that only
    /// searches reflections.
    pub fn new_reflections_only(db_path: &Path, projects_dir: &Path) -> Result<Self> {
        let mut engine = Self::new_unloaded(db_path, projects_dir)?;
        let (reflections, rmax) = load_side(&engine.storage, VectorTable::Reflections)?;
        *Arc::get_mut(&mut engine.search)
            .expect("a new engine's search index is not shared yet")
            .get_mut()
            .reflections_mut() = reflections;
        engine.watermarks.reflections.store(rmax, Ordering::Release);
        engine.vectors_ready.store(true, Ordering::Release);
        Ok(engine)
    }

    /// Open storage without loading any vectors. The daemon uses this so it
    /// can bind its endpoints first and call [`Self::load_vectors`] after.
    pub fn new_unloaded(db_path: &Path, projects_dir: &Path) -> Result<Self> {
        tracing::info!(?db_path, "opening storage");
        let storage = Arc::new(Storage::open(db_path)?);
        let embeddings = Arc::new(EmbeddingEngine::new()?);
        let engine = Self::from_parts(
            storage,
            embeddings,
            Arc::new(RwLock::new(SearchEngine::default())),
            projects_dir.to_path_buf(),
        );
        engine.vectors_ready.store(false, Ordering::Release);
        Ok(engine)
    }

    /// Whether the vectors have been loaded (always true except while a
    /// daemon is starting).
    pub fn vectors_ready(&self) -> bool {
        self.vectors_ready.load(Ordering::Acquire)
    }

    /// Wait up to `limit` for the vectors to finish loading.
    pub async fn wait_vectors_ready(&self, limit: std::time::Duration) -> bool {
        let deadline = tokio::time::Instant::now() + limit;
        while !self.vectors_ready() {
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        true
    }

    /// Replace both vector stores with everything in SQLite.
    ///
    /// Holds the search write lock for the whole load. Searches wait for the
    /// full store instead of seeing a partial one, and an insert or removal
    /// made meanwhile waits too and lands on the new store, so none is lost.
    /// The rows are read on a private connection, so storage keeps serving.
    pub async fn load_vectors(&self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let mut idx = self.search.write().await;
        let storage = self.storage.clone();
        let (chunks, cmax, reflections, rmax) = tokio::task::spawn_blocking(move || {
            let (c, cmax) = load_side(&storage, VectorTable::Chunks)?;
            let (r, rmax) = load_side(&storage, VectorTable::Reflections)?;
            anyhow::Ok((c, cmax, r, rmax))
        })
        .await??;
        *idx.chunks_mut() = chunks;
        *idx.reflections_mut() = reflections;
        self.watermarks.chunks.store(cmax, Ordering::Release);
        self.watermarks.reflections.store(rmax, Ordering::Release);
        self.vectors_ready.store(true, Ordering::Release);
        let line = format!(
            "CSR vectors loaded: chunks={} reflections={} in {}ms",
            idx.chunk_count(),
            idx.reflection_count(),
            t0.elapsed().as_millis()
        );
        drop(idx);
        log_timing(&line);
        tracing::info!("{line}");
        Ok(())
    }

    /// Reload one side from SQLite, holding the write lock as
    /// [`Self::load_vectors`] does.
    async fn reload_side(&self, table: VectorTable) -> Result<()> {
        let mut idx = self.search.write().await;
        let storage = self.storage.clone();
        let (store, max) =
            tokio::task::spawn_blocking(move || load_side(&storage, table)).await??;
        *side_mut(&mut idx, table) = store;
        self.watermarks.of(table).store(max, Ordering::Release);
        Ok(())
    }

    /// Bring the in-memory stores up to date with rows written by another
    /// process (a CLI import, for example).
    ///
    /// New rows are found by rowid above the watermark. A side whose row count
    /// still disagrees with SQLite after a short re-check is reloaded whole:
    /// that only happens when some other process deleted rows, and a reload
    /// costs about 3 s for 900k chunks.
    pub async fn sync_vectors(&self) -> Result<VectorSyncReport> {
        let mut report = VectorSyncReport::default();
        for table in [VectorTable::Chunks, VectorTable::Reflections] {
            let added = self.apply_new_rows(table).await?;
            let mut reloaded = false;
            if !self.count_matches(table).await? {
                tokio::time::sleep(COUNT_RECHECK_DELAY).await;
                self.apply_new_rows(table).await?;
                if !self.count_matches(table).await? {
                    self.reload_side(table).await?;
                    reloaded = true;
                }
            }
            match table {
                VectorTable::Chunks => {
                    report.chunks_added = added;
                    report.chunks_reloaded = reloaded;
                }
                VectorTable::Reflections => {
                    report.reflections_added = added;
                    report.reflections_reloaded = reloaded;
                }
            }
        }
        if report.chunks_reloaded || report.reflections_reloaded {
            log_timing(&format!("CSR vectors resynced: {report:?}"));
        }
        Ok(report)
    }

    async fn apply_new_rows(&self, table: VectorTable) -> Result<usize> {
        let after = self.watermarks.of(table).load(Ordering::Acquire);
        let storage = self.storage.clone();
        let (rows, max) = tokio::task::spawn_blocking(move || {
            let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
            let max = storage.stream_vectors_after(table, after, |id, v| {
                rows.push((id.to_owned(), v.to_vec()));
            })?;
            anyhow::Ok((rows, max))
        })
        .await??;
        if rows.is_empty() {
            return Ok(0);
        }
        let mut idx = self.search.write().await;
        let store = side_mut(&mut idx, table);
        for (id, v) in rows.iter() {
            store.upsert(id.clone(), v);
        }
        self.watermarks.of(table).fetch_max(max, Ordering::AcqRel);
        Ok(rows.len())
    }

    async fn count_matches(&self, table: VectorTable) -> Result<bool> {
        let storage = self.storage.clone();
        let db = tokio::task::spawn_blocking(move || storage.count_vectors(table)).await??;
        Ok(side(&*self.search.read().await, table).len() == db)
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

    /// Start the file system watcher as a background task.
    /// Returns a JoinHandle that can be awaited or dropped.
    pub fn start_watcher(&self) -> tokio::task::JoinHandle<()> {
        let watcher = FileWatcher::new(
            self.projects_dir.clone(),
            self.storage.clone(),
            self.embeddings.clone(),
            self.search.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::ConversationChunk;
    use crate::provenance::Speaker;

    fn chunk(id: &str) -> ConversationChunk {
        ConversationChunk {
            id: id.into(),
            conversation_id: "conv".into(),
            project_name: "proj".into(),
            timestamp: "2026-09-08T00:00:00Z".into(),
            content: format!("content {id}"),
            message_count: 1,
            summary: None,
            author: Speaker::User,
            seq: 0,
            is_sidechain: false,
        }
    }

    fn vec_for(i: usize) -> Vec<f32> {
        let mut v = vec![0.01_f32; 384];
        v[i % 384] = 1.0;
        v
    }

    struct Corpus {
        _tmp: tempfile::TempDir,
        db: PathBuf,
        projects: PathBuf,
    }

    impl Corpus {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let db = tmp.path().join("csr-engine.db");
            let projects = tmp.path().join("projects");
            std::fs::create_dir_all(&projects).unwrap();
            Self {
                _tmp: tmp,
                db,
                projects,
            }
        }

        fn storage(&self) -> Storage {
            Storage::open(&self.db).unwrap()
        }
    }

    #[tokio::test]
    async fn new_loads_every_vector_from_sqlite_and_finds_them() {
        let c = Corpus::new();
        let s = c.storage();
        for i in 0..40 {
            s.insert_chunk(&chunk(&format!("c{i}")), &vec_for(i))
                .unwrap();
        }
        s.insert_reflection("r1", "insight", &["t".into()], &vec_for(7))
            .unwrap();
        drop(s);

        let e = Engine::new(&c.db, &c.projects).unwrap();
        let idx = e.search().read().await;
        assert_eq!(idx.chunk_count(), 40);
        assert_eq!(idx.reflection_count(), 1);
        assert_eq!(idx.search_chunks(&vec_for(13), 1, 0.5)[0].id, "c13");
        assert_eq!(idx.search_reflections(&vec_for(7), 1, 0.5)[0].id, "r1");
    }

    #[tokio::test]
    async fn load_vectors_fills_an_unloaded_engine() {
        let c = Corpus::new();
        let s = c.storage();
        for i in 0..10 {
            s.insert_chunk(&chunk(&format!("c{i}")), &vec_for(i))
                .unwrap();
        }
        drop(s);
        let e = Engine::new_unloaded(&c.db, &c.projects).unwrap();
        assert_eq!(e.search().read().await.chunk_count(), 0);
        e.load_vectors().await.unwrap();
        assert_eq!(e.search().read().await.chunk_count(), 10);
    }

    #[tokio::test]
    async fn sync_picks_up_rows_another_process_wrote() {
        let c = Corpus::new();
        let e = Engine::new(&c.db, &c.projects).unwrap();
        let other = c.storage();
        for i in 0..5 {
            other
                .insert_chunk(&chunk(&format!("late{i}")), &vec_for(100 + i))
                .unwrap();
        }
        other
            .insert_reflection("r-late", "x", &[], &vec_for(3))
            .unwrap();
        let report = e.sync_vectors().await.unwrap();
        assert_eq!(report.chunks_added, 5);
        assert_eq!(report.reflections_added, 1);
        assert!(!report.chunks_reloaded && !report.reflections_reloaded);
        let idx = e.search().read().await;
        assert_eq!(idx.search_chunks(&vec_for(102), 1, 0.5)[0].id, "late2");
        drop(idx);
        assert!(
            !e.sync_vectors().await.unwrap().changed(),
            "second sync finds nothing new"
        );
    }

    #[tokio::test]
    async fn sync_reloads_a_side_after_another_process_deletes_rows() {
        let c = Corpus::new();
        let s = c.storage();
        for i in 0..6 {
            s.insert_reflection(&format!("r{i}"), "x", &[], &vec_for(i))
                .unwrap();
        }
        drop(s);
        let e = Engine::new(&c.db, &c.projects).unwrap();
        let other = c.storage();
        other.delete_reflection("r2").unwrap();
        let report = e.sync_vectors().await.unwrap();
        assert!(report.reflections_reloaded);
        let idx = e.search().read().await;
        assert_eq!(idx.reflection_count(), 5);
        assert!(!idx.has_reflection("r2"));
    }

    #[tokio::test]
    async fn writes_racing_a_load_are_never_lost() {
        // Production writers write SQLite first and then the store. Whichever
        // side of the load's lock they land on, the row must end up loaded.
        for round in 0..20 {
            let c = Corpus::new();
            let s = Arc::new(c.storage());
            for i in 0..50 {
                s.insert_chunk(&chunk(&format!("c{i}")), &vec_for(i))
                    .unwrap();
            }
            let e = Engine::new_unloaded(&c.db, &c.projects).unwrap();
            let loader = {
                let e = e.clone();
                tokio::spawn(async move { e.load_vectors().await })
            };
            let writer = {
                let e = e.clone();
                tokio::spawn(async move {
                    if round % 2 == 0 {
                        tokio::task::yield_now().await;
                    }
                    e.storage()
                        .insert_chunk(&chunk("racer"), &vec_for(300))
                        .unwrap();
                    e.search()
                        .write()
                        .await
                        .insert_chunk("racer".into(), vec_for(300));
                })
            };
            loader.await.unwrap().unwrap();
            writer.await.unwrap();
            let idx = e.search().read().await;
            assert!(idx.has_chunk("racer"), "round {round}: racing write lost");
            assert_eq!(idx.chunk_count(), 51);
        }
    }
}
