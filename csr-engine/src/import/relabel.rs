//! One label per session for chunks already stored.
//!
//! Two import paths used to label a session's chunks differently. The hooks
//! used the project of the session's working directory (`visilaunch`); the
//! file watcher used the transcript folder's name, which it only shortened
//! when the folder contained a lowercase `projects`, so a `PhpstormProjects`
//! session became `-Users-u-PhpstormProjects-visilaunch`. Whichever wrote
//! last won. On 2026-11-10, 636k of 913k chunks carried the raw folder form,
//! and a scoped search (which resolves `visilaunch`) missed them unless the
//! git-based folder rule in `hooks::scope_folder` happened to admit them.
//! Folders that are not git repositories (the home folder, Google Drive)
//! never were.
//!
//! Every import path now labels through
//! [`crate::import::project_label_for_transcript`]. This pass applies the
//! same rule to what is already stored: for each Claude Code transcript in
//! `import_state`, read the directory the session started in and relabel
//! that conversation's chunks to its project. When the transcript is gone
//! (Claude Code deletes old ones), the start directory is decoded from the
//! dash-encoded folder in its recorded path, against the filesystem as it is
//! now (`hooks::scope_folder::decoded_start_dir`). Conversations where neither
//! works are left as they are.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Result;
use rusqlite::params;

use crate::storage::Storage;

#[derive(Debug, Default)]
pub struct RelabelStats {
    pub conversations_checked: usize,
    /// Transcript missing on disk.
    pub transcripts_missing: usize,
    /// Transcript records no start directory (Codex rollouts and the like).
    pub without_start_dir: usize,
    /// One conversation id, transcripts whose start directories disagree.
    pub ambiguous: usize,
    /// Labelled from the decoded transcript folder (transcript gone or silent).
    pub decoded_from_folder: usize,
    pub conversations_relabeled: usize,
    pub chunks_relabeled: usize,
    /// (old label, new label) to chunk count.
    pub changes: BTreeMap<(String, String), usize>,
}

impl RelabelStats {
    pub fn render(&self, applied: bool) -> String {
        let verb = if applied {
            "relabeled"
        } else {
            "would relabel"
        };
        let mut out = format!(
            "checked {} conversations: {} labelled from their folder, {} left (transcript gone and folder undecodable), {} without a start directory, {} ambiguous\n{verb} {} chunks in {} conversations\n",
            self.conversations_checked,
            self.decoded_from_folder,
            self.transcripts_missing,
            self.without_start_dir,
            self.ambiguous,
            self.chunks_relabeled,
            self.conversations_relabeled,
        );
        let mut top: Vec<_> = self.changes.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        for ((from, to), n) in top.into_iter().take(15) {
            out.push_str(&format!("  {n:>7}  {from}  ->  {to}\n"));
        }
        out
    }
}

/// Relabel stored chunks to the label their transcript resolves to. With
/// `apply` false nothing is written.
pub fn relabel_projects(
    storage: &Storage,
    projects_dir: &Path,
    apply: bool,
) -> Result<RelabelStats> {
    relabel_projects_in(storage, Path::new("/"), projects_dir, apply)
}

/// [`relabel_projects`] with the filesystem root folder names decode against.
fn relabel_projects_in(
    storage: &Storage,
    fs_root: &Path,
    projects_dir: &Path,
    apply: bool,
) -> Result<RelabelStats> {
    let rows: Vec<(String, String)> = storage.with_connection(|conn| {
        let mut stmt = conn.prepare(
            "SELECT conversation_id, file_path FROM import_state
             WHERE conversation_id IS NOT NULL AND file_path LIKE '%.jsonl'",
        )?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    })?;

    let mut by_conv: HashMap<String, Vec<String>> = HashMap::new();
    for (conv, path) in rows {
        by_conv.entry(conv).or_default().push(path);
    }

    let mut stats = RelabelStats::default();
    let mut folder_labels: HashMap<String, Option<String>> = HashMap::new();
    let mut convs: Vec<_> = by_conv.into_iter().collect();
    convs.sort();
    for (conv, paths) in convs {
        stats.conversations_checked += 1;
        let mut labels: Vec<String> = Vec::new();
        let mut missing = 0;
        let mut no_dir = 0;
        for p in &paths {
            let path = Path::new(p);
            if !path.exists() {
                missing += 1;
                continue;
            }
            match super::transcript_start_dir(path)
                .and_then(|cwd| crate::search::cross_project::resolve_project_from_cwd(&cwd))
            {
                Some(label) => labels.push(label),
                None => no_dir += 1,
            }
        }
        let mut from_folder = false;
        if labels.is_empty() {
            for p in &paths {
                let label = folder_labels
                    .entry(p.clone())
                    .or_insert_with(|| {
                        crate::hooks::scope_folder::decoded_start_dir(fs_root, projects_dir, p)
                            .and_then(|dir| {
                                crate::search::cross_project::resolve_project_from_cwd(
                                    &dir.to_string_lossy(),
                                )
                            })
                    })
                    .clone();
                if let Some(l) = label {
                    labels.push(l);
                    from_folder = true;
                }
            }
        }
        labels.sort();
        labels.dedup();
        if from_folder && labels.len() == 1 {
            stats.decoded_from_folder += 1;
        }
        let label = match labels.as_slice() {
            [one] => one.clone(),
            [] if missing > 0 && no_dir == 0 => {
                stats.transcripts_missing += 1;
                continue;
            }
            [] => {
                stats.without_start_dir += 1;
                continue;
            }
            _ => {
                stats.ambiguous += 1;
                continue;
            }
        };

        let changes: Vec<(String, usize)> = storage.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT project_name, COUNT(*) FROM chunks
                 WHERE conversation_id = ?1 AND project_name != ?2 AND source = 'conversation'
                 GROUP BY project_name",
            )?;
            let rows = stmt
                .query_map(params![conv, label], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as usize))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })?;
        if changes.is_empty() {
            continue;
        }
        let n: usize = changes.iter().map(|(_, c)| c).sum();
        if apply {
            let written = storage.relabel_conversation(&conv, &label)?;
            stats.chunks_relabeled += written;
        } else {
            stats.chunks_relabeled += n;
        }
        stats.conversations_relabeled += 1;
        for (from, c) in changes {
            *stats.changes.entry((from, label.clone())).or_default() += c;
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::ConversationChunk;
    use crate::provenance::Speaker;

    fn chunk(id: &str, conv: &str, label: &str) -> ConversationChunk {
        ConversationChunk {
            id: id.into(),
            conversation_id: conv.into(),
            project_name: label.into(),
            timestamp: "2026-11-10T00:00:00Z".into(),
            content: format!("text {id}"),
            message_count: 1,
            summary: None,
            author: Speaker::User,
            seq: 0,
            is_sidechain: false,
        }
    }

    fn transcript(dir: &Path, conv: &str, cwd: Option<&str>) -> String {
        let f = dir.join(format!("{conv}.jsonl"));
        let line = match cwd {
            Some(c) => format!(
                r#"{{"type":"user","cwd":{},"message":{{"content":"x"}}}}"#,
                serde_json::to_string(c).unwrap()
            ),
            None => r#"{"type":"user","message":{"content":"x"}}"#.to_string(),
        };
        std::fs::write(&f, line + "\n").unwrap();
        f.to_string_lossy().into_owned()
    }

    fn label_of(storage: &Storage, id: &str) -> String {
        storage.get_chunks_by_ids(&[id.to_string()]).unwrap()[0]
            .project_name
            .clone()
    }

    #[test]
    fn relabels_to_the_start_directory_and_leaves_the_unknowable_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Storage::open_memory().unwrap();
        // Raw folder label, transcript says visilaunch.
        s.insert_chunk(
            &chunk("a1", "conv-a", "-Users-u-PhpstormProjects-visilaunch"),
            &[0.1; 384],
        )
        .unwrap();
        s.insert_chunk(&chunk("a2", "conv-a", "visilaunch"), &[0.1; 384])
            .unwrap();
        let pa = transcript(
            tmp.path(),
            "conv-a",
            Some("/Users/u/PhpstormProjects/visilaunch"),
        );
        s.mark_file_imported(Path::new(&pa), 2).unwrap();
        // No start directory recorded: untouched.
        s.insert_chunk(&chunk("b1", "conv-b", "-Users-u-somewhere"), &[0.1; 384])
            .unwrap();
        let pb = transcript(tmp.path(), "conv-b", None);
        s.mark_file_imported(Path::new(&pb), 1).unwrap();
        // Transcript gone: untouched.
        s.insert_chunk(&chunk("c1", "conv-c", "-Users-u-gone"), &[0.1; 384])
            .unwrap();
        s.mark_file_imported(&tmp.path().join("conv-c.jsonl"), 1)
            .unwrap();

        let dry = relabel_projects_in(&s, tmp.path(), &tmp.path().join("projects"), false).unwrap();
        assert_eq!(dry.chunks_relabeled, 1);
        assert_eq!(
            label_of(&s, "a1"),
            "-Users-u-PhpstormProjects-visilaunch",
            "dry run writes nothing"
        );

        let done = relabel_projects_in(&s, tmp.path(), &tmp.path().join("projects"), true).unwrap();
        assert_eq!(
            (done.conversations_relabeled, done.chunks_relabeled),
            (1, 1)
        );
        assert_eq!(done.without_start_dir, 1);
        assert_eq!(done.transcripts_missing, 1);
        assert_eq!(label_of(&s, "a1"), "visilaunch");
        assert_eq!(label_of(&s, "a2"), "visilaunch");
        assert_eq!(label_of(&s, "b1"), "-Users-u-somewhere");
        assert_eq!(label_of(&s, "c1"), "-Users-u-gone");

        let again =
            relabel_projects_in(&s, tmp.path(), &tmp.path().join("projects"), true).unwrap();
        assert_eq!(again.chunks_relabeled, 0, "idempotent");
    }

    #[test]
    fn a_gone_transcript_is_labelled_from_its_decoded_folder() {
        let tmp = tempfile::tempdir().unwrap();
        // The session's start directory still exists; its transcript does not.
        std::fs::create_dir_all(tmp.path().join("Users/u/PhpstormProjects/latina.web")).unwrap();
        let projects = tmp.path().join("projects");
        let folder = projects.join("-Users-u-PhpstormProjects-latina-web");
        std::fs::create_dir_all(&folder).unwrap();
        let s = Storage::open_memory().unwrap();
        s.insert_chunk(
            &chunk("m1", "conv-m", "-Users-u-PhpstormProjects-latina-web"),
            &[0.1; 384],
        )
        .unwrap();
        s.mark_file_imported(&folder.join("conv-m.jsonl"), 1)
            .unwrap();
        // A subagent transcript, also gone, labelled by its folder name.
        s.insert_chunk(&chunk("g1", "agent-1", "subagents"), &[0.1; 384])
            .unwrap();
        s.mark_file_imported(&folder.join("conv-m/subagents/agent-1.jsonl"), 1)
            .unwrap();
        // A folder whose directory no longer exists stays as it is.
        s.insert_chunk(&chunk("x1", "conv-x", "-Users-u-gone"), &[0.1; 384])
            .unwrap();
        s.mark_file_imported(&projects.join("-Users-u-gone/conv-x.jsonl"), 1)
            .unwrap();

        let done = relabel_projects_in(&s, tmp.path(), &projects, true).unwrap();
        assert_eq!(done.decoded_from_folder, 2);
        assert_eq!(label_of(&s, "m1"), "latina.web");
        assert_eq!(label_of(&s, "g1"), "latina.web");
        assert_eq!(label_of(&s, "x1"), "-Users-u-gone");
    }

    #[test]
    fn a_label_change_does_not_touch_the_full_text_index() {
        let s = Storage::open_memory().unwrap();
        s.insert_chunk(&chunk("a1", "conv-a", "-Users-u-x"), &[0.1; 384])
            .unwrap();
        let fts_rows = |s: &Storage| -> i64 {
            s.with_connection(|c| {
                Ok(c.query_row(
                    "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'text'",
                    [],
                    |r| r.get(0),
                )?)
            })
            .unwrap()
        };
        assert_eq!(fts_rows(&s), 1);
        let trigger: String = s
            .with_connection(|c| {
                Ok(c.query_row(
                    "SELECT sql FROM sqlite_master WHERE name = 'chunks_fts_au'",
                    [],
                    |r| r.get(0),
                )?)
            })
            .unwrap();
        assert!(trigger.contains("UPDATE OF content"), "{trigger}");
        assert_eq!(s.relabel_conversation("conv-a", "x").unwrap(), 1);
        assert_eq!(fts_rows(&s), 1, "index entry still there and still matches");
    }
}
