//! Claude Code hooks, served by the daemon over a Unix socket.
//!
//! `csr-engine hook <name>` is a thin client. It sends the hook's stdin, its
//! working directory and its `CSR_*` environment to the daemon, prints what
//! comes back, and exits 0. It never opens SQLite, never loads the embedding
//! model and never holds vectors, so a hook costs milliseconds and a few MB no
//! matter how many sessions are running.
//!
//! Protocol: one request per connection. The client writes a JSON
//! [`HookRequest`] and shuts down its write half; the daemon reads to EOF,
//! writes a JSON [`HookReply`] and closes.
//!
//! Two kinds of hook:
//! - **Reply hooks** (prompt-submit, session-start) print an injection, so the
//!   daemon runs them before replying.
//! - **Queued hooks** (stop, post-tool-use, precompact, session-end,
//!   session-briefing) print nothing. The daemon acknowledges at once and runs
//!   them in the background, so Claude Code never waits on them.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;

use crate::engine::Engine;
use crate::hooks::context::HookContext;
use crate::hooks::HookInput;

/// Largest request the daemon accepts. post-tool-use carries the tool's input
/// (a whole file for Write), so this is generous.
const MAX_REQUEST_BYTES: u64 = 64 * 1024 * 1024;
/// How long the daemon waits for a client to finish sending.
const READ_TIMEOUT: Duration = Duration::from_secs(10);
/// A reply hook that arrives while the vectors are still loading (only right
/// after a daemon restart) waits this long, then answers empty.
const VECTORS_READY_WAIT: Duration = Duration::from_secs(5);
/// Upper bound on one reply hook inside the daemon.
const REPLY_HOOK_LIMIT: Duration = Duration::from_secs(30);
/// Queued hooks waiting or running beyond this are dropped and logged.
const MAX_QUEUED: usize = 1000;
/// Concurrent queued hooks of the quick kind (stop, post-tool-use, precompact).
const FAST_LANE: usize = 4;
/// Concurrent queued hooks of the slow kind (session-end, session-briefing),
/// which can take minutes (V3 extraction, a nested `claude -p`).
const SLOW_LANE: usize = 2;

/// Client: connect timeout. A local socket either accepts at once or the
/// daemon is not there.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
/// Client: how long a reply hook waits for its injection.
const REPLY_BUDGET: Duration = Duration::from_secs(15);
/// Client: how long a queued hook waits for the acknowledgement.
const ACK_BUDGET: Duration = Duration::from_secs(3);

/// What the client sends.
#[derive(Debug, Serialize, Deserialize)]
pub struct HookRequest {
    /// Hook name, or an admin command (`ping`, `reload-vectors`).
    pub hook: String,
    /// The JSON Claude Code wrote to the hook's stdin, unparsed.
    #[serde(default)]
    pub input: String,
    /// The hook process's working directory, used when the input has no cwd.
    #[serde(default)]
    pub cwd: String,
    /// The hook process's forwarded environment (see
    /// [`crate::hooks::context::forwarded_env_var`]).
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// What the daemon answers.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct HookReply {
    /// Text for the client to print on stdout.
    #[serde(default)]
    pub stdout: String,
    /// The daemon accepted the hook and will run it in the background.
    #[serde(default)]
    pub queued: bool,
    /// Set when the daemon could not take the request.
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookKind {
    Reply,
    QueuedFast,
    QueuedSlow,
}

/// How the daemon treats `hook`, or `None` for an unknown name.
pub fn hook_kind(hook: &str) -> Option<HookKind> {
    match hook {
        "prompt-submit" | "session-start" => Some(HookKind::Reply),
        "stop" | "post-tool-use" | "precompact" => Some(HookKind::QueuedFast),
        "session-end" | "session-briefing" => Some(HookKind::QueuedSlow),
        _ => None,
    }
}

/// The daemon's socket for the database at `db_path`: `CSR_DAEMON_SOCKET` if
/// set, else `daemon.sock` beside the database. The hook client derives it
/// from the same default database path the daemon opens, so they always meet,
/// and a daemon run against a copy of the database never collides with the
/// live one.
pub fn socket_path_for(db_path: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("CSR_DAEMON_SOCKET") {
        return PathBuf::from(p);
    }
    db_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("daemon.sock")
}

// ─── Daemon side ───

/// Bind the hook socket, replacing a stale one. Only one daemon runs at a time
/// (it holds `daemon.lock` before calling this), so an existing socket file is
/// always left over from a previous run.
pub fn bind(path: &Path) -> Result<UnixListener> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).context("removing stale hook socket"),
    }
    let listener = UnixListener::bind(path)
        .with_context(|| format!("binding hook socket {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(listener)
}

struct Lanes {
    fast: Semaphore,
    slow: Semaphore,
    queued: AtomicUsize,
}

/// Accept hook connections until `shutdown` is set.
pub async fn serve(listener: UnixListener, engine: Engine, shutdown: Arc<AtomicBool>) {
    let lanes = Arc::new(Lanes {
        fast: Semaphore::new(FAST_LANE),
        slow: Semaphore::new(SLOW_LANE),
        queued: AtomicUsize::new(0),
    });
    while !shutdown.load(Ordering::SeqCst) {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(e) => {
                tracing::warn!(error = %e, "hook socket accept failed");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let engine = engine.clone();
        let lanes = lanes.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, engine, lanes).await {
                tracing::debug!(error = %e, "hook connection ended with an error");
            }
        });
    }
}

async fn handle_connection(
    mut stream: UnixStream,
    engine: Engine,
    lanes: Arc<Lanes>,
) -> Result<()> {
    let mut raw = Vec::new();
    tokio::time::timeout(
        READ_TIMEOUT,
        (&mut stream)
            .take(MAX_REQUEST_BYTES + 1)
            .read_to_end(&mut raw),
    )
    .await
    .context("client did not finish sending")??;
    let reply = if raw.len() as u64 > MAX_REQUEST_BYTES {
        HookReply {
            error: Some("request too large".into()),
            ..Default::default()
        }
    } else {
        match serde_json::from_slice::<HookRequest>(&raw) {
            Ok(req) => answer(req, &engine, &lanes).await,
            Err(e) => HookReply {
                error: Some(format!("bad request: {e}")),
                ..Default::default()
            },
        }
    };
    let bytes = serde_json::to_vec(&reply)?;
    stream.write_all(&bytes).await?;
    stream.shutdown().await?;
    Ok(())
}

async fn answer(req: HookRequest, engine: &Engine, lanes: &Arc<Lanes>) -> HookReply {
    match req.hook.as_str() {
        "ping" => {
            // Never wait on the vectors: a load holds the write lock for
            // seconds, and ping is how callers tell "loading" from "down".
            let (chunks, reflections) = match engine.search().try_read() {
                Ok(idx) => (
                    serde_json::json!(idx.chunk_count()),
                    serde_json::json!(idx.reflection_count()),
                ),
                Err(_) => (serde_json::Value::Null, serde_json::Value::Null),
            };
            return HookReply {
                stdout: serde_json::json!({
                    "vectors_ready": engine.vectors_ready(),
                    "chunks": chunks,
                    "reflections": reflections,
                    "queued": lanes.queued.load(Ordering::SeqCst),
                    "version": env!("CARGO_PKG_VERSION"),
                })
                .to_string(),
                ..Default::default()
            };
        }
        "reload-vectors" => {
            return match engine.load_vectors().await {
                Ok(()) => {
                    let idx = engine.search().read().await;
                    HookReply {
                        stdout: format!(
                            "reloaded vectors: {} chunks, {} reflections",
                            idx.chunk_count(),
                            idx.reflection_count()
                        ),
                        ..Default::default()
                    }
                }
                Err(e) => HookReply {
                    error: Some(format!("reload failed: {e}")),
                    ..Default::default()
                },
            };
        }
        _ => {}
    }

    let Some(kind) = hook_kind(&req.hook) else {
        return HookReply {
            error: Some(format!("unknown hook: {}", req.hook)),
            ..Default::default()
        };
    };

    match kind {
        HookKind::Reply => {
            if !engine.wait_vectors_ready(VECTORS_READY_WAIT).await {
                crate::telemetry::append_timing_line(&format!(
                    "CSR hook {}: vectors still loading, answered empty",
                    req.hook
                ));
                return HookReply::default();
            }
            let hook = req.hook.clone();
            match tokio::time::timeout(REPLY_HOOK_LIMIT, run(req, engine.clone())).await {
                Ok(stdout) => HookReply {
                    stdout,
                    ..Default::default()
                },
                Err(_) => {
                    crate::telemetry::append_timing_line(&format!(
                        "CSR hook {hook}: gave up after {}s in the daemon",
                        REPLY_HOOK_LIMIT.as_secs()
                    ));
                    HookReply::default()
                }
            }
        }
        HookKind::QueuedFast | HookKind::QueuedSlow => {
            if lanes.queued.fetch_add(1, Ordering::SeqCst) >= MAX_QUEUED {
                lanes.queued.fetch_sub(1, Ordering::SeqCst);
                crate::telemetry::append_timing_line(&format!(
                    "CSR hook {}: dropped, {MAX_QUEUED} hooks already queued",
                    req.hook
                ));
                return HookReply {
                    error: Some("daemon queue full".into()),
                    ..Default::default()
                };
            }
            let engine = engine.clone();
            let lanes = lanes.clone();
            let enqueued = Instant::now();
            tokio::spawn(async move {
                let lane = if kind == HookKind::QueuedSlow {
                    &lanes.slow
                } else {
                    &lanes.fast
                };
                let permit = lane.acquire().await;
                let waited = enqueued.elapsed();
                if waited > Duration::from_secs(1) {
                    crate::telemetry::append_timing_line(&format!(
                        "CSR hook {}: waited {}ms in the daemon queue",
                        req.hook,
                        waited.as_millis()
                    ));
                }
                let _ = run(req, engine).await;
                drop(permit);
                lanes.queued.fetch_sub(1, Ordering::SeqCst);
            });
            HookReply {
                queued: true,
                ..Default::default()
            }
        }
    }
}

/// Run one hook with the request's context. Returns what it printed.
async fn run(req: HookRequest, engine: Engine) -> String {
    let input: HookInput = if req.input.trim().is_empty() {
        HookInput::default()
    } else {
        serde_json::from_str(&req.input).unwrap_or_default()
    };
    let fallback_cwd = if req.cwd.is_empty() {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
    } else {
        PathBuf::from(&req.cwd)
    };
    let ctx = HookContext::forwarded(req.env);
    let result = ctx
        .scope(crate::hooks::run_hook(
            &req.hook,
            &input,
            &engine,
            &fallback_cwd,
        ))
        .await;
    if let Err(e) = result {
        crate::telemetry::append_timing_line(&format!("CSR hook {}: error: {e}", req.hook));
    }
    ctx.take_output()
}

// ─── Client side ───

/// Send `request` to the daemon at `socket` and wait up to `budget` for the
/// reply. Blocking: the hook process has nothing else to do.
pub fn send(socket: &Path, request: &HookRequest, budget: Duration) -> Result<HookReply> {
    let mut stream = connect(socket)?;
    stream.set_write_timeout(Some(budget))?;
    stream.set_read_timeout(Some(budget))?;
    let body = serde_json::to_vec(request)?;
    stream.write_all(&body)?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .context("daemon did not answer in time")?;
    let reply: HookReply = serde_json::from_slice(&raw).context("unreadable daemon reply")?;
    Ok(reply)
}

fn connect(socket: &Path) -> Result<std::os::unix::net::UnixStream> {
    // std has no connect timeout for Unix sockets. A listening socket accepts
    // immediately and a missing one fails immediately; the thread only
    // guards against a wedged accept queue.
    let path = socket.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(std::os::unix::net::UnixStream::connect(&path));
    });
    match rx.recv_timeout(CONNECT_TIMEOUT) {
        Ok(Ok(stream)) => Ok(stream),
        Ok(Err(e)) => {
            Err(e).with_context(|| format!("cannot reach the CSR daemon at {}", socket.display()))
        }
        Err(_) => anyhow::bail!(
            "CSR daemon at {} did not accept within 1s",
            socket.display()
        ),
    }
}

/// The client half of `csr-engine hook <name>`: forward stdin to the daemon
/// serving `db_path` and print the reply. Always returns normally: a hook must
/// never fail or block Claude Code.
pub fn run_client(hook: &str, db_path: &Path) {
    let t0 = Instant::now();
    let mut input = String::new();
    {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            let _ = std::io::stdin().read_to_string(&mut input);
        }
    }
    let request = HookRequest {
        hook: hook.to_string(),
        input,
        cwd: std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default(),
        env: std::env::vars()
            .filter(|(k, _)| crate::hooks::context::forwarded_env_var(k))
            .collect(),
    };
    let budget = match hook_kind(hook) {
        Some(HookKind::Reply) => REPLY_BUDGET,
        _ => ACK_BUDGET,
    };
    let socket = socket_path_for(db_path);
    match send(&socket, &request, budget) {
        Ok(reply) => {
            if let Some(err) = &reply.error {
                crate::telemetry::append_timing_line(&format!(
                    "CSR hook {hook}: daemon refused: {err}"
                ));
            }
            if !reply.stdout.is_empty() {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(reply.stdout.as_bytes());
                let _ = out.flush();
            }
            let ms = t0.elapsed().as_millis();
            if ms >= 1000 {
                crate::telemetry::append_timing_line(&format!(
                    "CSR hook {hook}: client waited {ms}ms"
                ));
            }
        }
        Err(e) => {
            crate::telemetry::append_timing_line(&format!(
                "CSR hook {hook}: daemon unreachable: {e:#}"
            ));
            if hook == "session-start" {
                // Never silent: tell the user once per session that CSR is off.
                let msg = format!(
                    "CSR memory is offline: the csr-engine daemon is not answering on {}. \
                     No past-session context this session. Check: launchctl print gui/$(id -u)/com.devin.csr-daemon",
                    socket.display()
                );
                println!("{}", serde_json::json!({ "systemMessage": msg }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_installed_hook_has_a_kind() {
        for h in [
            "prompt-submit",
            "session-start",
            "stop",
            "post-tool-use",
            "precompact",
            "session-end",
            "session-briefing",
        ] {
            assert!(hook_kind(h).is_some(), "{h}");
        }
        assert_eq!(hook_kind("prompt-submit"), Some(HookKind::Reply));
        assert_eq!(hook_kind("stop"), Some(HookKind::QueuedFast));
        assert_eq!(hook_kind("session-briefing"), Some(HookKind::QueuedSlow));
        assert_eq!(hook_kind("nope"), None);
    }

    fn engine_in(tmp: &tempfile::TempDir) -> Engine {
        let projects = tmp.path().join("projects");
        std::fs::create_dir_all(&projects).unwrap();
        Engine::new(&tmp.path().join("csr.db"), &projects).unwrap()
    }

    /// Start a server on a temp socket; returns the socket path.
    async fn start(tmp: &tempfile::TempDir, engine: Engine) -> PathBuf {
        let sock = tmp.path().join("d.sock");
        let listener = bind(&sock).unwrap();
        tokio::spawn(serve(listener, engine, Arc::new(AtomicBool::new(false))));
        sock
    }

    async fn send_async(sock: PathBuf, req: HookRequest, budget: Duration) -> Result<HookReply> {
        tokio::task::spawn_blocking(move || send(&sock, &req, budget)).await?
    }

    fn req(hook: &str, input: serde_json::Value) -> HookRequest {
        HookRequest {
            hook: hook.into(),
            input: input.to_string(),
            cwd: String::new(),
            env: HashMap::new(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ping_reports_loaded_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = start(&tmp, engine_in(&tmp)).await;
        let reply = send_async(
            sock,
            req("ping", serde_json::json!({})),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&reply.stdout).unwrap();
        assert_eq!(v["vectors_ready"], true);
        assert_eq!(v["chunks"], 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queued_hooks_are_acknowledged_without_waiting_for_the_work() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = start(&tmp, engine_in(&tmp)).await;
        let t = Instant::now();
        let reply = send_async(
            sock,
            req("stop", serde_json::json!({"session_id": "s1"})),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert!(reply.queued);
        assert!(reply.error.is_none());
        assert!(t.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unknown_hooks_and_garbage_get_an_error_not_a_hang() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = start(&tmp, engine_in(&tmp)).await;
        let reply = send_async(
            sock.clone(),
            req("nope", serde_json::json!({})),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert!(reply.error.unwrap().contains("unknown hook"));

        let raw = tokio::task::spawn_blocking(move || {
            let mut s = std::os::unix::net::UnixStream::connect(&sock).unwrap();
            s.write_all(b"not json").unwrap();
            s.shutdown(std::net::Shutdown::Write).unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            out
        })
        .await
        .unwrap();
        let reply: HookReply = serde_json::from_str(&raw).unwrap();
        assert!(reply.error.unwrap().starts_with("bad request"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reply_hook_answers_empty_while_vectors_load_instead_of_hanging() {
        let tmp = tempfile::tempdir().unwrap();
        let projects = tmp.path().join("projects");
        std::fs::create_dir_all(&projects).unwrap();
        // Never loaded: vectors_ready stays false.
        let engine = Engine::new_unloaded(&tmp.path().join("csr.db"), &projects).unwrap();
        let sock = start(&tmp, engine).await;
        let t = Instant::now();
        let reply = send_async(
            sock,
            req(
                "prompt-submit",
                serde_json::json!({"prompt": "how did we fix the daemon socket"}),
            ),
            Duration::from_secs(15),
        )
        .await
        .unwrap();
        assert!(reply.stdout.is_empty());
        let waited = t.elapsed();
        assert!(
            waited >= VECTORS_READY_WAIT && waited < VECTORS_READY_WAIT + Duration::from_secs(3)
        );
    }

    #[test]
    fn a_missing_daemon_fails_fast() {
        let tmp = tempfile::tempdir().unwrap();
        let t = Instant::now();
        let err = send(
            &tmp.path().join("absent.sock"),
            &req("stop", serde_json::json!({})),
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert!(t.elapsed() < Duration::from_millis(500));
        assert!(format!("{err:#}").contains("cannot reach"));
    }
}
