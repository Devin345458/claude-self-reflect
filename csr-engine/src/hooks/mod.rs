//! Hook handlers for Claude Code lifecycle events.
//!
//! Claude Code runs `csr-engine hook <name>` at lifecycle events, with JSON on
//! stdin. That process is a thin client: it forwards the input to the daemon
//! (`crate::daemon::hook_server`), prints what the daemon sends back (the
//! injection, for prompt-submit and session-start), and exits 0. The handlers
//! in this module run inside the daemon, against the one loaded engine.

pub mod context;
pub mod install;
pub mod intent;
pub mod post_tool_use;
pub mod precompact;
pub mod prompt_submit;
pub(crate) mod scope_folder;
pub mod session_briefing;
pub mod session_end;
pub mod session_start;
pub mod stop;

use std::io::Read;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use crate::engine::Engine;
use crate::search::cross_project::resolve_project_from_cwd;

/// Input received from Claude Code via stdin JSON.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HookInput {
    #[serde(alias = "sessionId")]
    pub session_id: Option<String>,
    #[serde(alias = "transcriptPath")]
    pub transcript_path: Option<String>,
    pub cwd: Option<String>,
    pub reason: Option<String>,
    /// Tool name for PostToolUse hook
    #[serde(alias = "toolName")]
    pub tool_name: Option<String>,
    /// Tool input for PostToolUse hook (contains file_path, content, etc.)
    #[serde(alias = "toolInput")]
    pub tool_input: Option<serde_json::Value>,
    /// Whether the Stop hook is re-entering (to prevent infinite loops)
    #[serde(alias = "stopHookActive")]
    pub stop_hook_active: Option<bool>,
    /// User's prompt text for UserPromptSubmit hook
    pub prompt: Option<String>,
    /// Hook event source (e.g. "startup", "resume", "compact", "clear") for SessionStart
    pub source: Option<String>,
}

/// Claude Code timeout the installer writes for `name`. The hook client is
/// done well before it: 1 s to connect, then 15 s for an injection (reply
/// hooks) or 3 s for an acknowledgement (queued hooks); see
/// `crate::daemon::hook_server`.
pub fn installed_timeout_secs(name: &str) -> u64 {
    match crate::daemon::hook_server::hook_kind(name) {
        Some(crate::daemon::hook_server::HookKind::Reply) => 20,
        _ => 10,
    }
}

/// A hook stops itself this long before Claude Code would give up on it.
const DEADLINE_MARGIN_SECS: u64 = 2;

/// How long a hook process may run before it exits on its own.
///
/// A safety net: the client is already bounded by its socket timeouts. Claude
/// Code discards a hook's output at its timeout but can leave the process
/// running, so an unbounded hook could still stack up behind a wedged daemon.
///
/// `env_override` is `CSR_HOOK_DEADLINE_SECS`: a positive integer replaces the
/// table, `0` disables the deadline, anything else is ignored.
pub fn hook_deadline(name: &str, env_override: Option<&str>) -> Option<std::time::Duration> {
    if let Some(secs) = env_override.and_then(|raw| raw.trim().parse::<u64>().ok()) {
        return (secs > 0).then(|| std::time::Duration::from_secs(secs));
    }
    Some(std::time::Duration::from_secs(
        installed_timeout_secs(name) - DEADLINE_MARGIN_SECS,
    ))
}

/// Start the deadline for this hook process: a background thread that records
/// the overrun in hook-timing.log and exits the process with code 1, which
/// Claude Code treats as a non-blocking hook error (never 2, which blocks).
pub fn arm_hook_deadline(name: &str) {
    let env_override = std::env::var("CSR_HOOK_DEADLINE_SECS").ok();
    let Some(deadline) = hook_deadline(name, env_override.as_deref()) else {
        return;
    };
    let name = name.to_string();
    let _ = std::thread::Builder::new()
        .name("csr-hook-deadline".into())
        .spawn(move || {
            std::thread::sleep(deadline);
            let line = format!(
                "CSR hook {name}: exceeded {}s deadline, exiting",
                deadline.as_secs()
            );
            crate::telemetry::append_timing_line(&line);
            eprintln!("{line} (see ~/.claude-self-reflect/hook-timing.log)");
            std::process::exit(1);
        });
}

pub fn read_stdin_json() -> HookInput {
    use std::io::IsTerminal;

    // If stdin is a TTY (not piped), don't block waiting for input
    if std::io::stdin().is_terminal() {
        return HookInput::default();
    }

    let mut buf = String::new();
    match std::io::stdin().read_to_string(&mut buf) {
        Ok(_) if !buf.trim().is_empty() => serde_json::from_str(&buf).unwrap_or_default(),
        _ => HookInput::default(),
    }
}

/// Shared helper: import the current transcript incrementally.
/// Used by stop, precompact, prompt_submit, and session_end hooks.
pub async fn import_current_transcript(input: &HookInput, engine: &Engine, cwd: &Path) {
    let Some(ref transcript) = input.transcript_path else {
        return;
    };
    let tp = std::path::PathBuf::from(transcript);
    if !tp.exists() {
        return;
    }

    // S-1 fix: validate transcript path is a .jsonl file.
    // The .jsonl extension check prevents importing arbitrary files (e.g. /etc/passwd)
    // via crafted hook JSON. Claude Code only produces .jsonl transcripts.
    let Ok(canonical) = tp.canonicalize() else {
        return;
    };
    if canonical.extension().and_then(|e| e.to_str()) != Some("jsonl") {
        eprintln!(
            "CSR: refusing non-JSONL transcript: {}",
            canonical.display()
        );
        return;
    }

    // Inside the daemon the file watcher owns transcript imports. Hand it the
    // path (it de-duplicates within its debounce window) instead of parsing the
    // whole transcript again on every hook.
    if context::in_daemon() && crate::import::watcher::nudge(canonical.clone()) {
        return;
    }

    let _ = cwd;
    let project = crate::import::project_label_for_transcript(&canonical);

    match engine.import_file(&canonical, &project).await {
        Ok(0) => {} // no new content, silent
        Ok(n) => eprintln!("CSR: indexed {} new chunks from active session", n),
        Err(e) => eprintln!("CSR: import failed (non-fatal): {}", e),
    }
}

/// Save `target` and point it at `sink`, returning the saved descriptor.
/// Shared by [`StdoutQuarantine`] and [`ClaimedStdout`] so the fcntl/dup2 pair
/// exists in exactly one place.
///
/// All-or-nothing: a non-negative return means both steps landed, `target`
/// now points at `sink`, and the caller owns the returned descriptor (it must
/// eventually be restored or closed). A negative return means `target` was
/// left exactly as it was and there is nothing to close. Reporting success
/// after only the `fcntl` half worked would leave `target` on the *original*
/// stream while callers believe it is diverted — for [`ClaimedStdout`] that
/// means JSON-RPC frames written through the saved descriptor and library
/// `println!`s would both land on the same real fd 1, corrupting the
/// protocol stream, which is the exact failure this diversion exists to
/// prevent. A half-applied diversion is worse than none, so on a failed
/// `dup2` the saved copy is closed and the caller is told nothing happened:
/// the failure exit is a single unconditional `close` on the only path where
/// `dup2` did not succeed, so there is nowhere for the saved descriptor to
/// leak. The all-or-nothing contract, including that close-on-failure, is
/// covered deterministically through [`divert_fd_with`]'s injected effects —
/// what remains untestable is *observing* the leak from outside via the
/// process's open-fd high-water mark, which is process-global state that
/// other threads move arbitrarily under the parallel test harness.
#[cfg(unix)]
fn divert_fd(target: libc::c_int, sink: libc::c_int) -> libc::c_int {
    // SAFETY: fcntl/dup2 on this process's own descriptors. CLOEXEC keeps the
    // saved copy out of any child spawned while the guard is held.
    divert_fd_with(
        target,
        sink,
        |fd| unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) },
        |s, t| unsafe { libc::dup2(s, t) },
        |fd| unsafe {
            libc::close(fd);
        },
    )
}

/// The all-or-nothing decision behind [`divert_fd`], with every syscall
/// injected as a closure. Pure logic on top of the three effects — save,
/// dup2, close — so the failure-path cleanup can be proven deterministically
/// in a unit test without touching any real descriptor or depending on
/// process-global allocation state.
#[cfg(unix)]
fn divert_fd_with(
    target: libc::c_int,
    sink: libc::c_int,
    save: impl Fn(libc::c_int) -> libc::c_int,
    dup2: impl Fn(libc::c_int, libc::c_int) -> libc::c_int,
    close: impl Fn(libc::c_int),
) -> libc::c_int {
    let saved = save(target);
    if saved < 0 {
        return saved;
    }
    if dup2(sink, target) < 0 {
        // `target` is unchanged; the saved copy is now useless, close it
        // rather than leak it.
        close(saved);
        return -1;
    }
    saved
}

/// Claims real fd 1 for the JSON-RPC transport, for the lifetime of the MCP
/// server process, and sends everything else that would have landed on stdout
/// to stderr instead.
///
/// Unlike [`StdoutQuarantine`], which is scoped to a single hook invocation,
/// this has to hold for as long as the process serves MCP: the server is
/// long-lived, `Engine::new` rebuilds or backfills the HNSW index at startup
/// (the same `hnsw_rs` `println!` covered by `StdoutQuarantine`'s doc comment),
/// and `spawn_enrichment_loops` plus `--watch` importers keep inserting into
/// that index *while the server is running*, so the 50,000-point boundary can
/// fire mid-session, long after any scoped guard would have been dropped.
/// stdout is the MCP JSON-RPC channel, so any of those prints reaching it
/// corrupts the protocol stream. Claim it once, before `Engine::new` runs, and
/// hold it for the rest of the process; there is deliberately no restore.
///
/// Everything that used to reach stdout now reaches stderr instead, which
/// Claude Code captures into its own MCP server logs — the noise moves to
/// where it belongs rather than disappearing. The saved descriptor is
/// `CLOEXEC`, so children spawned after the claim (narrator `claude -p`
/// subprocesses, `--watch` helpers) never inherit the real JSON-RPC channel.
pub struct ClaimedStdout {
    #[cfg(unix)]
    saved: libc::c_int,
}

impl ClaimedStdout {
    /// Flushes Rust's stdout buffer, then diverts fd 1 to fd 2 and hands back a
    /// handle to the saved original. Returns `None` (nothing redirected, the
    /// caller falls back to the existing `rmcp::transport::io::stdio()` path)
    /// when `divert_fd` fails to fully apply the diversion, or unconditionally
    /// on non-unix, where Rust's stdout does not go through a CRT fd `dup2`
    /// can redirect.
    pub fn claim() -> Option<Self> {
        let _ = std::io::Write::flush(&mut std::io::stdout());
        #[cfg(unix)]
        {
            Self::claim_on(libc::STDOUT_FILENO, libc::STDERR_FILENO)
        }
        #[cfg(not(unix))]
        {
            None
        }
    }

    #[cfg(unix)]
    fn claim_on(target: libc::c_int, sink: libc::c_int) -> Option<Self> {
        let saved = divert_fd(target, sink);
        if saved < 0 {
            return None;
        }
        Some(Self { saved })
    }

    /// The saved raw descriptor, for tests that claim on pipes instead of the
    /// real fd 1.
    #[cfg(all(unix, test))]
    fn saved_fd(&self) -> libc::c_int {
        self.saved
    }

    /// Hands ownership of the saved descriptor to a `tokio::fs::File` so it can
    /// back the MCP transport's write half. Consumes `self` on purpose: the
    /// claim is meant to last for the process lifetime, so there is no `Drop`
    /// impl to race with this handoff, and the descriptor must not be closed
    /// out from under the file that now owns it.
    #[cfg(unix)]
    pub fn into_async_writer(self) -> tokio::fs::File {
        use std::os::unix::io::FromRawFd;
        // SAFETY: `saved` came from `F_DUPFD_CLOEXEC` in `divert_fd` and is
        // owned solely by this `ClaimedStdout` (no `Drop` impl closes it), so
        // handing it to `std::fs::File` does not create a double-close or use
        // of a descriptor still held elsewhere.
        tokio::fs::File::from_std(unsafe { std::fs::File::from_raw_fd(self.saved) })
    }
}

/// Run a hook in this process, reading its input from stdin, against an
/// engine this process built. The installed hooks never do this (they forward
/// to the daemon); it remains for debugging a handler without a daemon.
pub async fn dispatch_hook(hook_name: &str, engine: &Engine) -> Result<()> {
    let input = read_stdin_json();
    run_hook(hook_name, &input, engine, &std::env::current_dir()?).await
}

/// Resolve the working directory a hook should act in: the `cwd` Claude Code
/// sent, if it is a real directory under `$HOME` (S-3 fix), else `fallback`
/// (the hook process's own working directory).
pub fn resolve_hook_cwd(input: &HookInput, fallback: &Path) -> Result<std::path::PathBuf> {
    if let Some(ref dir) = input.cwd {
        let p = std::path::PathBuf::from(dir);
        if p.is_dir() {
            let canonical = p.canonicalize()?;
            if let Some(home) = dirs::home_dir() {
                if !canonical.starts_with(&home) {
                    anyhow::bail!("cwd {} is outside home directory", canonical.display());
                }
            }
            return Ok(canonical);
        }
    }
    Ok(fallback.to_path_buf())
}

/// Run one hook handler. Shared by the daemon, for forwarded hooks, and by
/// [`dispatch_hook`]. Logs one timing line to hook-timing.log.
pub async fn run_hook(
    hook_name: &str,
    input: &HookInput,
    engine: &Engine,
    fallback_cwd: &Path,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let cwd = resolve_hook_cwd(input, fallback_cwd)?;

    let result = match hook_name {
        "session-start" => session_start::handle(input, engine, &cwd).await,
        "session-briefing" => session_briefing::handle(input, engine, &cwd).await,
        "session-end" => session_end::handle(input, engine, &cwd).await,
        "precompact" => precompact::handle(input, engine, &cwd).await,
        "stop" => stop::handle(input, engine, &cwd).await,
        "post-tool-use" => post_tool_use::handle(input, engine, &cwd).await,
        "prompt-submit" => prompt_submit::handle(input, engine, &cwd).await,
        _ => {
            eprintln!("unknown hook: {}", hook_name);
            Ok(())
        }
    };

    let cwd_str = cwd.to_string_lossy();
    let project = resolve_project_from_cwd(&cwd_str).unwrap_or_else(|| "unknown".to_string());
    let where_ = if context::in_daemon() {
        "daemon"
    } else {
        "local"
    };
    crate::telemetry::append_timing_line(&format!(
        "CSR hook {hook_name} [{project}]: hook={ms}ms total={ms}ms ({where_})",
        ms = t0.elapsed().as_millis()
    ));

    result
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The MCP claim must behave like the hook quarantine's diversion half:
    /// once claimed, library writes to the target land on the sink, and the
    /// caller can still reach the original stream through the saved
    /// descriptor. Run on pipes so the test process's real fd 1 is untouched.
    #[cfg(unix)]
    #[test]
    fn claimed_stdout_diverts_target_to_sink() {
        fn pipe() -> (libc::c_int, libc::c_int) {
            let mut fds = [0 as libc::c_int; 2];
            assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
            (fds[0], fds[1])
        }
        fn write_fd(fd: libc::c_int, bytes: &[u8]) {
            let n = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
            assert_eq!(n, bytes.len() as isize);
        }
        fn read_fd(fd: libc::c_int) -> String {
            let mut buf = [0u8; 64];
            let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
            assert!(n > 0, "nothing arrived on fd {fd}");
            String::from_utf8_lossy(&buf[..n as usize]).into_owned()
        }

        let (out_r, out_w) = pipe();
        let (sink_r, sink_w) = pipe();

        let claimed = ClaimedStdout::claim_on(out_w, sink_w).expect("claim_on should succeed");

        // Library code still writing to the (now-diverted) target lands on the sink.
        write_fd(out_w, b"library noise");
        assert_eq!(read_fd(sink_r), "library noise");

        // The saved descriptor still reaches the original reader.
        let saved = claimed.saved_fd();
        write_fd(saved, b"jsonrpc");
        assert_eq!(read_fd(out_r), "jsonrpc");

        unsafe { libc::close(saved) };
        for fd in [out_r, out_w, sink_r, sink_w] {
            unsafe { libc::close(fd) };
        }
    }

    /// A failed `dup2` half of `divert_fd` must not be reported as a
    /// successful claim: `sink_w` is closed up front, which is a bad fd for
    /// `dup2`, so the `fcntl` save succeeds but the `dup2` fails with
    /// `EBADF`. `claim_on` must refuse the claim rather than hand back a
    /// `ClaimedStdout` whose saved descriptor still points at the *original*
    /// stream while `target` was never actually repointed — and `target`
    /// itself must be provably untouched: still readable by its original
    /// reader, not silently left on a stream nobody can read.
    #[cfg(unix)]
    #[test]
    fn claimed_stdout_refuses_a_partial_diversion() {
        fn pipe() -> (libc::c_int, libc::c_int) {
            let mut fds = [0 as libc::c_int; 2];
            assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
            (fds[0], fds[1])
        }
        fn write_fd(fd: libc::c_int, bytes: &[u8]) {
            let n = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
            assert_eq!(n, bytes.len() as isize);
        }
        fn read_fd(fd: libc::c_int) -> String {
            let mut buf = [0u8; 64];
            let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
            assert!(n > 0, "nothing arrived on fd {fd}");
            String::from_utf8_lossy(&buf[..n as usize]).into_owned()
        }

        let (out_r, out_w) = pipe();
        // A sentinel that is definitionally invalid as a dup2 target, rather
        // than a real descriptor number recycled from a closed pipe: under
        // the parallel test harness other threads churn fds constantly, and
        // `fcntl(.., F_DUPFD_CLOEXEC, 3)` hands out the lowest free number,
        // so a closed-and-reused sink could collide with a fd another thread
        // just opened, making `dup2` succeed and the test flaky. `c_int::MAX`
        // can never be allocated by that call, so `dup2` fails with `EBADF`
        // deterministically.
        let sink_w = libc::c_int::MAX;

        let claimed = ClaimedStdout::claim_on(out_w, sink_w);
        assert!(
            claimed.is_none(),
            "a partial diversion (fcntl ok, dup2 EBADF) must not be reported as a successful claim"
        );

        // `target` must be exactly as it was: still reachable by its original
        // reader, not silently repointed at a stream nobody can read.
        write_fd(out_w, b"jsonrpc");
        assert_eq!(read_fd(out_r), "jsonrpc");

        for fd in [out_r, out_w] {
            unsafe { libc::close(fd) };
        }
    }

    /// `divert_fd`'s failure path (`fcntl`/save succeeds, `dup2` fails) must
    /// close the descriptor it saved rather than leak it — the property
    /// `claimed_stdout_refuses_a_partial_diversion` cannot observe from
    /// outside. Proven here through `divert_fd_with`'s injected effects: a
    /// fake `save` that hands back a fixed, unmistakable descriptor number,
    /// a fake `dup2` that always fails, and a `close` that records every
    /// descriptor it was asked to close. No real syscalls, no dependence on
    /// process-global descriptor state.
    #[cfg(unix)]
    #[test]
    fn divert_fd_with_closes_the_saved_copy_on_a_failed_dup2() {
        let closed = std::cell::RefCell::new(Vec::new());
        let result = divert_fd_with(
            1,
            2,
            |_target| 4242,
            |_sink, _target| -1,
            |fd| {
                closed.borrow_mut().push(fd);
            },
        );

        assert!(
            result < 0,
            "a failed dup2 must be reported as a failed diversion"
        );
        assert_eq!(
            closed.into_inner(),
            vec![4242],
            "the saved descriptor must be closed exactly once on the failed dup2 path"
        );
    }

    /// The mirror case: when `dup2` succeeds, the saved descriptor is handed
    /// back to the caller (who now owns it), not closed.
    #[cfg(unix)]
    #[test]
    fn divert_fd_with_does_not_close_the_saved_copy_on_success() {
        let closed = std::cell::RefCell::new(Vec::new());
        let result = divert_fd_with(
            1,
            2,
            |_target| 4242,
            |_sink, _target| 0,
            |fd| {
                closed.borrow_mut().push(fd);
            },
        );

        assert_eq!(
            result, 4242,
            "a successful diversion must return the saved descriptor"
        );
        assert!(
            closed.into_inner().is_empty(),
            "a successful diversion must not close the saved descriptor"
        );
    }

    #[test]
    fn hook_deadline_stays_under_each_hooks_installed_timeout() {
        use std::time::Duration;
        for (name, deadline) in [
            ("prompt-submit", 18),
            ("session-start", 18),
            ("stop", 8),
            ("post-tool-use", 8),
            ("precompact", 8),
            ("session-end", 8),
            ("session-briefing", 8),
            ("some-future-hook", 8),
        ] {
            let d = hook_deadline(name, None).unwrap();
            assert_eq!(d, Duration::from_secs(deadline), "{name}");
            assert!(
                d < Duration::from_secs(installed_timeout_secs(name)),
                "{name}"
            );
        }
    }

    #[test]
    fn hook_deadline_env_override() {
        use std::time::Duration;
        assert_eq!(
            hook_deadline("stop", Some("7")),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            hook_deadline("stop", Some(" 7 ")),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            hook_deadline("stop", Some("0")),
            None,
            "0 disables the deadline"
        );
        assert_eq!(
            hook_deadline("stop", Some("soon")),
            Some(Duration::from_secs(8)),
            "an unparseable override falls back to the table"
        );
    }
}
