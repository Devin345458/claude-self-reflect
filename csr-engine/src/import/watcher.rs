use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use tokio::sync::RwLock;

use crate::embeddings::EmbeddingEngine;
use crate::import;
use crate::import::ConversationChunk;
use crate::search::SearchEngine;
use crate::storage::Storage;

/// Debounce interval for collecting file changes before processing.
const DEBOUNCE_SECS: u64 = 5;

/// Batch size for embedding during auto-import.
const BATCH_SIZE: usize = 10;

/// The running watcher's queue, so a hook handled inside the daemon can hand
/// it a transcript path instead of importing the file itself.
static NUDGE: std::sync::OnceLock<tokio::sync::mpsc::UnboundedSender<PathBuf>> =
    std::sync::OnceLock::new();

/// Queue `path` for the watcher's next import pass. Returns false when no
/// watcher is running in this process (or it has stopped), so the caller can
/// import the file itself.
pub fn nudge(path: PathBuf) -> bool {
    NUDGE.get().is_some_and(|tx| tx.send(path).is_ok())
}

/// Watches the Claude projects directory for new JSONL files and auto-imports them.
pub struct FileWatcher {
    projects_dir: PathBuf,
    storage: Arc<Storage>,
    embeddings: Arc<EmbeddingEngine>,
    search: Arc<RwLock<SearchEngine>>,
}

impl FileWatcher {
    pub fn new(
        projects_dir: PathBuf,
        storage: Arc<Storage>,
        embeddings: Arc<EmbeddingEngine>,
        search: Arc<RwLock<SearchEngine>>,
    ) -> Self {
        Self {
            projects_dir,
            storage,
            embeddings,
            search,
        }
    }

    /// Spawn the watcher as a background tokio task. Returns a JoinHandle.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            if let Err(e) = self.run().await {
                tracing::error!(error = %e, "file watcher stopped with error");
            }
        })
    }

    /// Main watcher loop: watches for new JSONL files, debounces, and imports.
    async fn run(&self) -> Result<()> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PathBuf>();
        let _ = NUDGE.set(tx.clone());

        // Create the notify watcher — sends events via a closure that bridges to tokio channel
        let sender = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            if let Ok(event) = res {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                        for path in event.paths {
                            if path.extension().is_some_and(|ext| ext == "jsonl") {
                                let _ = sender.send(path);
                            }
                        }
                    }
                    _ => {}
                }
            }
        })?;

        // Watch the projects directory recursively
        watcher.watch(&self.projects_dir, RecursiveMode::Recursive)?;
        tracing::info!(dir = %self.projects_dir.display(), "file watcher started");

        // Debounce loop: collect changed files over DEBOUNCE_SECS, then process
        loop {
            let mut pending: HashSet<PathBuf> = HashSet::new();

            // Wait for the first event
            match rx.recv().await {
                Some(path) => {
                    pending.insert(path);
                }
                None => {
                    tracing::info!("file watcher channel closed, stopping");
                    break;
                }
            }

            // Collect more events within the debounce window
            let deadline =
                tokio::time::Instant::now() + tokio::time::Duration::from_secs(DEBOUNCE_SECS);
            loop {
                let timeout = tokio::time::timeout_at(deadline, rx.recv());
                match timeout.await {
                    Ok(Some(path)) => {
                        pending.insert(path);
                    }
                    Ok(None) => {
                        // Channel closed
                        break;
                    }
                    Err(_) => {
                        // Timeout — debounce window expired
                        break;
                    }
                }
            }

            // Process all pending files
            if !pending.is_empty() {
                tracing::info!(count = pending.len(), "processing new/modified JSONL files");
                for file_path in &pending {
                    if let Err(e) = self.import_file(file_path).await {
                        tracing::warn!(
                            file = %file_path.display(),
                            error = %e,
                            "failed to import file"
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Import a single JSONL file: parse → embed → store → index.
    async fn import_file(&self, file_path: &Path) -> Result<()> {
        // Security: resolve symlinks and verify the file is within our projects directory
        let canonical = file_path
            .canonicalize()
            .context("failed to canonicalize import path")?;
        let canonical_base = self
            .projects_dir
            .canonicalize()
            .context("failed to canonicalize projects dir")?;
        if !canonical.starts_with(&canonical_base) {
            tracing::warn!(
                file = %file_path.display(),
                resolved = %canonical.display(),
                "refusing to import file outside projects directory (symlink or traversal)"
            );
            return Ok(());
        }

        // Skip if already imported
        if self.storage.is_file_imported(file_path)? {
            return Ok(());
        }

        // One label per session, the same one its hooks and MCP scope resolve.
        let project_name = import::project_label_for_transcript(file_path);

        let chunks = import::parse_jsonl_file(file_path, &project_name)?;
        if chunks.is_empty() {
            // Record the skip so this file isn't re-parsed every watcher pass
            // and import_percent counts it as processed.
            self.storage.mark_file_imported(file_path, 0)?;
            return Ok(());
        }

        let chunk_count = chunks.len();

        // Batch embed and store — but only what actually changed.
        //
        // This is the live path, and it has no tail arithmetic: `is_file_imported`
        // returns false on any mtime change, and Claude Code appends to the session
        // JSONL after every message, so `parse_jsonl_file_with_stats` above re-derives
        // EVERY chunk of the file every ~5s (DEBOUNCE_SECS) for the whole life of a
        // session. Re-deriving is cheap; re-embedding and rewriting all N chunks is
        // not, and it grows with the session. Only chunks that differ from the row
        // already stored need work — appends change the final partial chunk and add
        // new ones, leaving every earlier chunk byte-identical.
        //
        // Compared fields are the ones that decide what search returns: `content`
        // (the indexed text), plus `project_name`/`is_sidechain`, which scope it.
        // `summary`/`seq`/`timestamp` are deterministic re-derivations of the same
        // prefix bytes, so they cannot drift without `content` drifting too.
        let mut reindexed = 0usize;
        let mut relabeled = 0usize;
        let mut unchanged = 0usize;
        for batch in chunks.chunks(BATCH_SIZE) {
            let ids: Vec<String> = batch.iter().map(|c| c.id.clone()).collect();
            let stored: HashMap<String, ConversationChunk> = self
                .storage
                .get_chunks_by_ids(&ids)?
                .into_iter()
                .map(|chunk| (chunk.id.clone(), chunk))
                .collect();
            let mut pending: Vec<&ConversationChunk> = Vec::new();
            for chunk in batch {
                match change_for(chunk, stored.get(&chunk.id)) {
                    Change::None => unchanged += 1,
                    // Same text, so the same embedding: fix the label in place.
                    Change::Scope => {
                        self.storage.update_chunk_scope(
                            &chunk.id,
                            &chunk.project_name,
                            chunk.is_sidechain,
                        )?;
                        relabeled += 1;
                    }
                    Change::Content => pending.push(chunk),
                }
            }
            reindexed += pending.len();
            if pending.is_empty() {
                continue;
            }

            let texts: Vec<String> = pending.iter().map(|c| c.content.clone()).collect();
            let emb = self.embeddings.clone();
            let embeddings = tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                emb.embed(&refs)
            })
            .await??;

            let mut idx = self.search.write().await;
            for (chunk, embedding) in pending.iter().zip(embeddings) {
                self.storage.insert_chunk(chunk, &embedding)?;
                idx.insert_chunk(chunk.id.clone(), embedding);
            }
        }

        self.storage.mark_file_imported(file_path, chunk_count)?;

        // Layer 1: Heuristic enrichment (inline, instant, free)
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
                &project_name,
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

        tracing::info!(
            file = %file_path.display(),
            chunks = chunk_count,
            reindexed,
            relabeled,
            unchanged,
            project = %project_name,
            "auto-imported conversation"
        );

        Ok(())
    }
}

/// Does this freshly parsed chunk need to be embedded and written again?
///
/// `stored` is the row already in `chunks` under the same deterministic id
/// (`uuid_v5(conversation_id, seq)`), or `None` when this chunk is new.
///
/// Answering "no" is what stops the live watcher from re-embedding and
/// rewriting an entire session every debounce window. It is safe because the
/// storage write is an upsert keyed on that same id: re-running it with
/// identical values produces a byte-identical row, an identical embedding, and
/// an identical FTS document.
/// How a freshly parsed chunk differs from the stored row with its id.
#[derive(Debug, PartialEq, Eq)]
enum Change {
    /// Identical where it matters.
    None,
    /// Same text, different label or sidechain flag: no new embedding needed.
    Scope,
    /// New chunk or new text: embed and store.
    Content,
}

/// Compared fields are the ones that decide what search returns: `content`
/// (the indexed text) and `project_name`/`is_sidechain`, which scope it.
fn change_for(parsed: &ConversationChunk, stored: Option<&ConversationChunk>) -> Change {
    match stored {
        None => Change::Content,
        Some(stored) if stored.content != parsed.content => Change::Content,
        Some(stored)
            if stored.project_name != parsed.project_name
                || stored.is_sidechain != parsed.is_sidechain =>
        {
            Change::Scope
        }
        Some(_) => Change::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::Speaker;

    fn chunk(id: &str, content: &str) -> ConversationChunk {
        ConversationChunk {
            id: id.to_string(),
            conversation_id: "conv".to_string(),
            project_name: "proj".to_string(),
            timestamp: "2026-08-12T00:00:00Z".to_string(),
            content: content.to_string(),
            message_count: 1,
            summary: None,
            author: Speaker::ToolResult,
            seq: 0,
            is_sidechain: false,
        }
    }

    #[test]
    fn unseen_chunk_is_indexed() {
        assert_eq!(change_for(&chunk("a", "hello"), None), Change::Content);
    }

    #[test]
    fn byte_identical_chunk_is_skipped() {
        // The whole point: an append to the transcript re-derives every earlier
        // chunk unchanged, and those must cost neither an embedding nor a write.
        let parsed = chunk("a", "hello");
        let stored = chunk("a", "hello");
        assert_eq!(change_for(&parsed, Some(&stored)), Change::None);
    }

    #[test]
    fn grown_final_chunk_is_reindexed() {
        // The last chunk of a live session is partial and gains text on append.
        let stored = chunk("a", "hello");
        let parsed = chunk("a", "hello there");
        assert_eq!(change_for(&parsed, Some(&stored)), Change::Content);
    }

    #[test]
    fn rescoped_chunk_is_relabeled_without_a_new_embedding() {
        let stored = chunk("a", "hello");
        let mut parsed = chunk("a", "hello");
        parsed.project_name = "other".to_string();
        assert_eq!(change_for(&parsed, Some(&stored)), Change::Scope);

        let mut parsed = chunk("a", "hello");
        parsed.is_sidechain = true;
        assert_eq!(change_for(&parsed, Some(&stored)), Change::Scope);

        // New text and a new label: the text wins, it needs an embedding.
        let mut parsed = chunk("a", "hello there");
        parsed.project_name = "other".to_string();
        assert_eq!(change_for(&parsed, Some(&stored)), Change::Content);
    }

    #[test]
    fn a_label_only_update_changes_the_row_but_not_the_embedding() {
        let storage = Storage::open_memory().unwrap();
        let mut c = chunk("a", "hello");
        c.project_name = "-Users-u-PhpstormProjects-app".into();
        storage.insert_chunk(&c, &[0.5_f32; 384]).unwrap();
        assert!(storage.update_chunk_scope("a", "app", false).unwrap());
        assert!(
            !storage.update_chunk_scope("a", "app", false).unwrap(),
            "no-op second time"
        );
        let row = &storage.get_chunks_by_ids(&["a".to_string()]).unwrap()[0];
        assert_eq!(row.project_name, "app");
        assert_eq!(row.content, "hello");
        let mut n = 0;
        storage
            .stream_vectors_after(crate::storage::VectorTable::Chunks, 0, |id, v| {
                assert_eq!(id, "a");
                assert_eq!(v[0], 0.5);
                n += 1;
            })
            .unwrap();
        assert_eq!(n, 1, "embedding untouched");
    }
}
