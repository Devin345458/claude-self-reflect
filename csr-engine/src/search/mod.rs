pub mod code_rank;
pub mod cross_project;
pub mod decay;
pub mod reinstatement;
pub mod rerank;

use std::collections::{HashMap, HashSet};

/// A search result with score and ID.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

/// Below this many candidate rows a search runs on the calling thread: spawning
/// workers costs more than the scan itself.
const PARALLEL_THRESHOLD: usize = 32_768;
/// Upper bound on scan threads. The scan is memory-bandwidth bound; measured on
/// 908k x 384 vectors (M-series, 2026-11-10) it stops improving past 4 to 8.
const MAX_SCAN_THREADS: usize = 8;

/// Exact cosine search over vectors held in memory.
///
/// Every vector is L2-normalized on insert and every query is normalized before
/// a search, so cosine similarity is a plain dot product. Rows live in one
/// contiguous `Vec<f32>`. Removing a row moves the last row into its slot, so
/// there are never tombstones and the store never needs compacting.
///
/// There is no on-disk form: SQLite holds every embedding, and the daemon
/// builds the store from it at startup (about 3 s for 900k vectors when the
/// database is in the page cache).
#[derive(Default)]
pub struct VectorStore {
    dim: usize,
    data: Vec<f32>,
    ids: Vec<String>,
    rows: HashMap<String, usize>,
    /// Ids whose last upsert was refused (wrong dimension, zero vector). They
    /// exist in SQLite but not here, so count comparisons add them back:
    /// otherwise one bad row reads as permanent drift.
    rejected: HashSet<String>,
}

impl VectorStore {
    pub fn with_capacity(rows: usize) -> Self {
        Self {
            dim: 0,
            data: Vec::new(),
            ids: Vec::with_capacity(rows),
            rows: HashMap::with_capacity(rows),
            rejected: HashSet::new(),
        }
    }

    /// Stored rows plus refused ones: the number SQLite should hold.
    pub fn len_with_rejected(&self) -> usize {
        self.ids.len() + self.rejected.len()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.rows.contains_key(id)
    }

    /// Dimension fixed by the first insert (0 while empty).
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.ids.iter().map(String::as_str)
    }

    /// Insert or replace the vector for `id`. Returns false (and stores
    /// nothing) for an empty vector, a vector whose length differs from the
    /// store's dimension, or a zero vector.
    pub fn upsert(&mut self, id: String, embedding: &[f32]) -> bool {
        let ok = self.upsert_inner(&id, embedding);
        if ok {
            self.rejected.remove(&id);
        } else {
            // A refused replacement also drops the old vector: SQLite no
            // longer holds it.
            self.remove_row(&id);
            self.rejected.insert(id);
        }
        ok
    }

    fn upsert_inner(&mut self, id: &str, embedding: &[f32]) -> bool {
        if embedding.is_empty() {
            return false;
        }
        if self.dim == 0 {
            self.dim = embedding.len();
            if self.data.capacity() == 0 {
                self.data.reserve(self.ids.capacity() * self.dim);
            }
        }
        if embedding.len() != self.dim {
            tracing::warn!(
                id = %id,
                got = embedding.len(),
                want = self.dim,
                "vector store: skipped an embedding with the wrong dimension"
            );
            return false;
        }
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return false;
        }
        let dim = self.dim;
        match self.rows.get(id) {
            Some(&row) => {
                let dst = &mut self.data[row * dim..(row + 1) * dim];
                for (d, s) in dst.iter_mut().zip(embedding) {
                    *d = s / norm;
                }
            }
            None => {
                let row = self.ids.len();
                self.data.extend(embedding.iter().map(|s| s / norm));
                self.rows.insert(id.to_string(), row);
                self.ids.push(id.to_string());
            }
        }
        true
    }

    /// Remove `id`. Returns whether it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        let refused = self.rejected.remove(id);
        self.remove_row(id) || refused
    }

    fn remove_row(&mut self, id: &str) -> bool {
        let Some(row) = self.rows.remove(id) else {
            return false;
        };
        let last = self.ids.len() - 1;
        let dim = self.dim;
        if row != last {
            self.data
                .copy_within(last * dim..(last + 1) * dim, row * dim);
            self.ids.swap(row, last);
            self.rows.insert(self.ids[row].clone(), row);
        }
        self.ids.pop();
        self.data.truncate(last * dim);
        if self.ids.is_empty() {
            self.dim = 0;
            self.data = Vec::new();
        }
        true
    }

    /// Remove every id for which `keep` returns false. Returns how many went.
    pub fn retain(&mut self, mut keep: impl FnMut(&str) -> bool) -> usize {
        let doomed: Vec<String> = self.ids.iter().filter(|id| !keep(id)).cloned().collect();
        for id in &doomed {
            self.remove(id);
        }
        doomed.len()
    }

    /// Top `limit` rows by cosine similarity, at or above `min_score`, sorted
    /// by descending score (ties broken by id so results are deterministic).
    pub fn search(&self, query: &[f32], limit: usize, min_score: f32) -> Vec<SearchResult> {
        let Some(q) = self.prepare_query(query, limit) else {
            return Vec::new();
        };
        let hits = self.scan(&q, Candidates::All(self.len()), limit, min_score);
        self.finish(hits, limit)
    }

    /// Like [`Self::search`], restricted to `allowed` ids. Ids not in the
    /// store are ignored.
    pub fn search_among(
        &self,
        query: &[f32],
        allowed: &HashSet<String>,
        limit: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        let Some(q) = self.prepare_query(query, limit) else {
            return Vec::new();
        };
        let mut rows: Vec<usize> = allowed
            .iter()
            .filter_map(|id| self.rows.get(id.as_str()).copied())
            .collect();
        if rows.is_empty() {
            return Vec::new();
        }
        rows.sort_unstable();
        let hits = self.scan(&q, Candidates::Rows(&rows), limit, min_score);
        self.finish(hits, limit)
    }

    fn prepare_query(&self, query: &[f32], limit: usize) -> Option<Vec<f32>> {
        if limit == 0 || self.is_empty() || query.len() != self.dim {
            return None;
        }
        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        Some(query.iter().map(|x| x / norm).collect())
    }

    fn scan(
        &self,
        q: &[f32],
        cands: Candidates<'_>,
        limit: usize,
        min_score: f32,
    ) -> Vec<(f32, usize)> {
        let n = cands.len();
        let threads = if n < PARALLEL_THRESHOLD {
            1
        } else {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
                .clamp(1, MAX_SCAN_THREADS)
        };
        if threads == 1 {
            return self.scan_range(q, &cands, 0, n, limit, min_score);
        }
        let per = n.div_ceil(threads);
        let mut merged: Vec<(f32, usize)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let lo = (t * per).min(n);
                    let hi = ((t + 1) * per).min(n);
                    let cands = &cands;
                    s.spawn(move || self.scan_range(q, cands, lo, hi, limit, min_score))
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("vector scan thread panicked"))
                .collect()
        });
        merged.sort_by(|a, b| b.0.total_cmp(&a.0));
        merged
    }

    /// Scan candidates `lo..hi`, keeping the best `limit` as (score, row),
    /// ordered by descending score and then ascending id, so a tie at the
    /// cut-off keeps the same rows however the scan was split across threads.
    fn scan_range(
        &self,
        q: &[f32],
        cands: &Candidates<'_>,
        lo: usize,
        hi: usize,
        limit: usize,
        min_score: f32,
    ) -> Vec<(f32, usize)> {
        let dim = self.dim;
        let ids = &self.ids;
        // a ranks before b
        let before =
            |a: (f32, usize), b: (f32, usize)| a.0 > b.0 || (a.0 == b.0 && ids[a.1] < ids[b.1]);
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(limit + 1);
        for i in lo..hi {
            let row = cands.row(i);
            let s = dot(&self.data[row * dim..(row + 1) * dim], q);
            if s < min_score {
                continue;
            }
            let hit = (s, row);
            if best.len() == limit && !before(hit, best[limit - 1]) {
                continue;
            }
            let pos = best.partition_point(|&x| before(x, hit));
            best.insert(pos, hit);
            if best.len() > limit {
                best.pop();
            }
        }
        best
    }

    /// Turn the merged (score, row) hits into results, sorted by descending
    /// score then id, cut to `limit`.
    fn finish(&self, hits: Vec<(f32, usize)>, limit: usize) -> Vec<SearchResult> {
        let mut out: Vec<SearchResult> = hits
            .into_iter()
            .map(|(score, row)| SearchResult {
                id: self.ids[row].clone(),
                score,
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        out.truncate(limit);
        out
    }
}

enum Candidates<'a> {
    All(usize),
    Rows(&'a [usize]),
}

impl Candidates<'_> {
    fn len(&self) -> usize {
        match self {
            Candidates::All(n) => *n,
            Candidates::Rows(r) => r.len(),
        }
    }

    #[inline]
    fn row(&self, i: usize) -> usize {
        match self {
            Candidates::All(_) => i,
            Candidates::Rows(r) => r[i],
        }
    }
}

/// Dot product with 16 independent accumulators, which lets LLVM vectorize it
/// (NEON on Apple silicon) without fast-math.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, ra) = a.as_chunks::<16>();
    let (cb, rb) = b.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// The two vector stores the engine searches: conversation chunks and
/// reflections (stored insights and session episodes).
///
/// Exactly one process holds this: the daemon. Hook processes and the CLI
/// reach it through the daemon instead of loading their own copy.
#[derive(Default)]
pub struct SearchEngine {
    chunks: VectorStore,
    reflections: VectorStore,
}

impl SearchEngine {
    /// `estimated_size` pre-sizes the chunk side; the reflection side is sized
    /// at a tenth of it.
    pub fn new(estimated_size: usize) -> Self {
        Self {
            chunks: VectorStore::with_capacity(estimated_size),
            reflections: VectorStore::with_capacity(estimated_size / 10),
        }
    }

    /// Insert or replace a chunk vector.
    pub fn insert_chunk(&mut self, id: String, embedding: Vec<f32>) {
        self.chunks.upsert(id, &embedding);
    }

    /// Insert or replace a reflection vector.
    pub fn insert_reflection(&mut self, id: String, embedding: Vec<f32>) {
        self.reflections.upsert(id, &embedding);
    }

    pub fn remove_reflection(&mut self, id: &str) {
        self.reflections.remove(id);
    }

    pub fn remove_chunk(&mut self, id: &str) {
        self.chunks.remove(id);
    }

    pub fn has_reflection(&self, id: &str) -> bool {
        self.reflections.contains(id)
    }

    pub fn has_chunk(&self, id: &str) -> bool {
        self.chunks.contains(id)
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Same as [`Self::chunk_count`]: the store has no dead rows.
    pub fn live_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn reflection_count(&self) -> usize {
        self.reflections.len()
    }

    pub fn chunks(&self) -> &VectorStore {
        &self.chunks
    }

    pub fn chunks_mut(&mut self) -> &mut VectorStore {
        &mut self.chunks
    }

    pub fn reflections(&self) -> &VectorStore {
        &self.reflections
    }

    pub fn reflections_mut(&mut self) -> &mut VectorStore {
        &mut self.reflections
    }

    /// Search chunks. Returns results sorted by descending score.
    pub fn search_chunks(
        &self,
        query_vec: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        self.chunks.search(query_vec, limit, min_score)
    }

    /// Search reflections. Returns results sorted by descending score.
    pub fn search_reflections(
        &self,
        query_vec: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        self.reflections.search(query_vec, limit, min_score)
    }

    /// Search chunks, returning only ids in `allowed_ids`. Used for
    /// project-scoped and time-range-scoped searches. Exact, so a sparse
    /// filter never loses results the way an approximate index did.
    pub fn search_chunks_filtered(
        &self,
        query_vec: &[f32],
        limit: usize,
        min_score: f32,
        allowed_ids: &HashSet<String>,
    ) -> Vec<SearchResult> {
        self.chunks
            .search_among(query_vec, allowed_ids, limit, min_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random unit-ish vectors.
    fn vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        x = x
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    fn brute(store_vecs: &[(String, Vec<f32>)], q: &[f32], limit: usize, min: f32) -> Vec<String> {
        let qn = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut s: Vec<(f32, String)> = store_vecs
            .iter()
            .map(|(id, v)| {
                let vn = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let d: f32 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                (d / (vn * qn), id.clone())
            })
            .filter(|(s, _)| *s >= min)
            .collect();
        s.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        s.into_iter().take(limit).map(|(_, id)| id).collect()
    }

    #[test]
    fn empty_store_returns_nothing() {
        let s = VectorStore::default();
        assert!(s.search(&[1.0, 0.0], 5, -1.0).is_empty());
        let e = SearchEngine::new(10);
        assert!(e.search_chunks(&[1.0; 4], 5, 0.0).is_empty());
        assert!(e.search_reflections(&[1.0; 4], 5, 0.0).is_empty());
    }

    #[test]
    fn matches_a_brute_force_reference_serial_and_parallel() {
        for n in [10usize, 1_000, PARALLEL_THRESHOLD + 5_000] {
            let dim = 48;
            let data: Vec<(String, Vec<f32>)> = vecs(n, dim, n as u64)
                .into_iter()
                .enumerate()
                .map(|(i, v)| (format!("id-{i:06}"), v))
                .collect();
            let mut s = VectorStore::with_capacity(n);
            for (id, v) in &data {
                assert!(s.upsert(id.clone(), v));
            }
            for q in vecs(3, dim, 99 + n as u64) {
                let got: Vec<String> = s.search(&q, 25, -1.0).into_iter().map(|r| r.id).collect();
                assert_eq!(got, brute(&data, &q, 25, -1.0), "n={n}");
            }
        }
    }

    #[test]
    fn scores_are_cosine_and_respect_min_score() {
        let mut s = VectorStore::default();
        s.upsert("same".into(), &[2.0, 0.0, 0.0, 0.0]);
        s.upsert("orth".into(), &[0.0, 3.0, 0.0, 0.0]);
        s.upsert("opp".into(), &[-1.0, 0.0, 0.0, 0.0]);
        let r = s.search(&[5.0, 0.0, 0.0, 0.0], 10, 0.5);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, "same");
        assert!((r[0].score - 1.0).abs() < 1e-6);
        let all = s.search(&[5.0, 0.0, 0.0, 0.0], 10, -1.0);
        assert_eq!(
            all.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["same", "orth", "opp"]
        );
    }

    #[test]
    fn upsert_replaces_instead_of_duplicating() {
        let mut s = VectorStore::default();
        s.upsert("a".into(), &[1.0, 0.0]);
        s.upsert("a".into(), &[0.0, 1.0]);
        assert_eq!(s.len(), 1);
        let r = s.search(&[0.0, 1.0], 5, 0.9);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id, "a");
    }

    #[test]
    fn remove_keeps_every_other_row_findable() {
        let dim = 8;
        let data = vecs(50, dim, 7);
        let mut s = VectorStore::default();
        for (i, v) in data.iter().enumerate() {
            s.upsert(format!("{i}"), v);
        }
        for i in (0..50).step_by(3) {
            assert!(s.remove(&format!("{i}")));
        }
        assert!(!s.remove("0"), "second remove is a no-op");
        assert_eq!(s.len(), 50 - 17);
        for (i, v) in data.iter().enumerate() {
            let top = s.search(v, 1, -1.0);
            if i % 3 == 0 {
                assert_ne!(top[0].id, format!("{i}"));
                assert!(!s.contains(&format!("{i}")));
            } else {
                assert_eq!(top[0].id, format!("{i}"), "row {i} lost after removals");
            }
        }
    }

    #[test]
    fn removing_everything_resets_the_dimension() {
        let mut s = VectorStore::default();
        s.upsert("a".into(), &[1.0, 0.0]);
        s.remove("a");
        assert!(s.is_empty());
        assert!(
            s.upsert("b".into(), &[1.0, 0.0, 0.0]),
            "a new dimension is accepted once empty"
        );
    }

    #[test]
    fn refused_rows_still_count_toward_the_sqlite_total() {
        let mut s = VectorStore::default();
        s.upsert("good".into(), &[1.0, 0.0]);
        s.upsert("zero".into(), &[0.0, 0.0]);
        assert_eq!((s.len(), s.len_with_rejected()), (1, 2));
        // Fixed later: counted once, as a real row.
        s.upsert("zero".into(), &[0.0, 1.0]);
        assert_eq!((s.len(), s.len_with_rejected()), (2, 2));
        // A refused replacement drops the old vector but still counts.
        s.upsert("good".into(), &[0.0, 0.0, 1.0]);
        assert!(!s.contains("good"));
        assert_eq!((s.len(), s.len_with_rejected()), (1, 2));
        assert!(s.remove("good"), "removing a refused id is a real removal");
        assert_eq!(s.len_with_rejected(), 1);
    }

    #[test]
    fn rejects_wrong_dimension_and_zero_vectors() {
        let mut s = VectorStore::default();
        assert!(s.upsert("a".into(), &[1.0, 0.0, 0.0]));
        assert!(!s.upsert("b".into(), &[1.0, 0.0]));
        assert!(!s.upsert("c".into(), &[0.0, 0.0, 0.0]));
        assert!(!s.upsert("d".into(), &[]));
        assert_eq!(s.len(), 1);
        assert!(
            s.search(&[1.0, 0.0], 5, -1.0).is_empty(),
            "query of the wrong dimension"
        );
    }

    #[test]
    fn filtered_search_only_returns_allowed_ids_and_ignores_unknown_ones() {
        let dim = 16;
        let data = vecs(200, dim, 3);
        let mut e = SearchEngine::new(200);
        for (i, v) in data.iter().enumerate() {
            e.insert_chunk(format!("c{i}"), v.clone());
        }
        let allowed: HashSet<String> = ["c5", "c17", "c150", "not-there"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let r = e.search_chunks_filtered(&data[17], 10, -1.0, &allowed);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].id, "c17");
        assert!(r.iter().all(|x| allowed.contains(&x.id)));
        assert!(e
            .search_chunks_filtered(&data[17], 10, -1.0, &HashSet::new())
            .is_empty());
    }

    #[test]
    fn ties_are_ordered_by_id() {
        let mut s = VectorStore::default();
        for id in ["c", "a", "b"] {
            s.upsert(id.into(), &[1.0, 1.0]);
        }
        let r = s.search(&[1.0, 1.0], 2, -1.0);
        assert_eq!(
            r.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn retain_drops_ids_the_caller_rejects() {
        let mut s = VectorStore::default();
        for i in 0..10 {
            s.upsert(format!("{i}"), &[i as f32 + 1.0, 1.0]);
        }
        let keep: HashSet<&str> = ["1", "4", "9"].into_iter().collect();
        assert_eq!(s.retain(|id| keep.contains(id)), 7);
        let mut left: Vec<&str> = s.ids().collect();
        left.sort();
        assert_eq!(left, ["1", "4", "9"]);
    }

    #[test]
    fn engine_reflection_side_is_independent_of_chunks() {
        let mut e = SearchEngine::new(10);
        e.insert_chunk("x".into(), vec![1.0, 0.0]);
        e.insert_reflection("x".into(), vec![0.0, 1.0]);
        assert!(e.has_chunk("x") && e.has_reflection("x"));
        e.remove_reflection("x");
        assert!(e.has_chunk("x") && !e.has_reflection("x"));
        assert_eq!((e.chunk_count(), e.reflection_count()), (1, 0));
    }
}
