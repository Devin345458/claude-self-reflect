//! MCP Resource endpoints for system health and status information.

use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use tokio::sync::RwLock;

use crate::search::SearchEngine;
use crate::storage::Storage;

/// Build system health JSON from storage and search state.
pub async fn system_health(
    storage: &Arc<Storage>,
    search: &Arc<RwLock<SearchEngine>>,
    db_path: &str,
) -> String {
    let t0 = Instant::now();

    let chunks_indexed = storage.count_chunk_embeddings().unwrap_or(0);
    let reflections_indexed = storage.count_reflection_embeddings().unwrap_or(0);

    let idx = search.read().await;
    let (chunks_in_memory, reflections_in_memory) = (idx.chunk_count(), idx.reflection_count());
    drop(idx);

    let health = json!({
        "chunks_indexed": chunks_indexed,
        "reflections_indexed": reflections_indexed,
        "chunks_in_memory": chunks_in_memory,
        "reflections_in_memory": reflections_in_memory,
        "query_ms": t0.elapsed().as_secs_f64() * 1000.0,
        "db_path": db_path,
        "binary_version": env!("CARGO_PKG_VERSION"),
    });

    serde_json::to_string_pretty(&health).unwrap_or_else(|_| health.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn system_health_reports_sqlite_and_in_memory_counts() {
        let storage = Arc::new(Storage::open_memory().unwrap());
        storage
            .insert_reflection("r1", "x", &[], &[1.0_f32; 384])
            .unwrap();
        let mut idx = SearchEngine::new(4);
        idx.insert_chunk("c1".into(), vec![1.0; 384]);
        idx.insert_chunk("c2".into(), vec![0.5; 384]);
        let search = Arc::new(RwLock::new(idx));

        let v: serde_json::Value =
            serde_json::from_str(&system_health(&storage, &search, "/tmp/x.db").await).unwrap();
        assert_eq!(v["chunks_indexed"], 0);
        assert_eq!(v["reflections_indexed"], 1);
        assert_eq!(v["chunks_in_memory"], 2);
        assert_eq!(v["reflections_in_memory"], 0);
        assert_eq!(v["db_path"], "/tmp/x.db");
        assert!(v["binary_version"].is_string());
    }
}
