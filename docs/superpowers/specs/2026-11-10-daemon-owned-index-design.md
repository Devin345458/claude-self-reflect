# Daemon-owned index: design

Date: 2026-11-10. Branch: `rearch/daemon-owned-index` (from f65fc361). Hard fork of
claude-self-reflect: upstream compatibility is a non-goal.

## Why

Every Claude Code hook ran as a fresh process that built an `Engine`, loaded its own copy of
the HNSW index (1.9 GB on disk, about 4 GB resident, 4 s idle, 20 to 32 s under load), and then
competed for `index.lock` to save it. Everything patched since October follows from that:
rebuild storms, negative drift, compaction, 124 GB of index writes, stop hooks starving on the
exclusive lock, and prompt-submit hooks timing out and losing their injection. On 2026-11-10 a
burst of 10 to 11 hook processes a minute pushed stop, prompt-submit and session-start past
their 55 s deadline repeatedly.

## Goal

CSR is boring: no hook errors, no waiting on it, no storms, no feature removed.

Non-goals: upstream compatibility; npm installer; Windows.
Anti-goals: a hook that blocks Claude Code; a failure that is silent to the user.

## Decisions

1. **Exact search replaces HNSW.** Embeddings are normalized 384-dim vectors; cosine is a dot
   product. A threaded scan of 908,650 vectors takes 15 ms at 4 to 12 threads (30 ms on one),
   measured while the load average was 334 to 594. The store lives in memory only; SQLite is the
   single source of truth. Deleted with it: index files, manifest, `index.lock`, generations,
   tombstones, compaction, drift reconcile, and the hnsw_rs stdout leak.
2. **The daemon is the only process holding vectors.** It binds its endpoints first and loads
   vectors from SQLite in the background (about 3 s warm, 22 s cold). MCP tools already run in
   it over HTTP.
3. **Hooks are thin clients.** `csr-engine hook <name>` forwards stdin, cwd and relevant env to
   the daemon over a Unix socket (`~/.claude-self-reflect/daemon.sock`) and prints the reply. No
   Engine, no embedding model, no SQLite in the hook process.
   - Output hooks (prompt-submit, session-start, session-briefing) wait for the reply within a
     budget under their Claude Code timeout.
   - Write-only hooks (stop, post-tool-use, precompact, session-end) return as soon as the daemon
     acknowledges; the daemon does the work on a queue.
   - Daemon unreachable: the hook exits 0 at once and logs to hook-timing.log. session-start also
     prints a `systemMessage` so the user sees CSR is down (never silent).
4. **Other writers stay possible.** CLI commands may still write SQLite directly; the daemon
   catches up new embedding rows by rowid and reconciles deletions by count.
5. **Daemon priority.** The LaunchAgent no longer runs at background priority, since hooks now
   wait on it.
6. **Two embedding models in the daemon.** Each model serializes its callers. With one, every
   prompt-submit queued behind the watcher's chunk embeds (0.9 to 1.9 s live). Background loops
   get their own model; hooks and MCP keep the other. Live prompt-submit: 54 to 104 ms.
7. **Install path.** `setup` registers MCP over HTTP only. A local install script builds, signs,
   and swaps the binary with bootout and bootstrap.

## Project labels (added 2026-11-10)

The watcher labelled chunks by transcript folder name and only shortened names containing a
lowercase `projects`; the hooks labelled by working directory. With hooks moved into the daemon
only the raw form remained, and scoped search (which resolves the short form) missed it outside
git repositories. Now every import path labels by the transcript's start directory, label-only
changes skip re-embedding and the full-text trigger, and `relabel-projects` fixed 682,589 stored
chunks (589,025 then 93,564 sidechain). About 151k chunks stay raw: their directories are gone.

## Known gaps

- (Fixed 2026-11-10) MCP project scoping read the daemon's environment. Each session now
  scopes to its first MCP root; see `mcp/session.rs`.
- The session-start `systemMessage` for a down daemon is emitted (tested) but has not been
  seen rendering in Claude Code.

## Left to the builder

Protocol framing, queue size, per-hook budgets, thread counts, and how handler output is
captured, as long as the anti-goals hold.
