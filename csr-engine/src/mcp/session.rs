//! Per-session client directory for MCP over HTTP.
//!
//! Over stdio, Claude Code starts one engine per session and exports
//! `CLAUDE_PROJECT_DIR` into it, so "the current project" came from the
//! process environment. Over HTTP one daemon serves every session, and its
//! environment belongs to none of them. Each session's directory now comes
//! from the client itself: MCP roots. On the first tool call of a session the
//! server asks the client for its roots (`roots/list`) and keeps the first
//! `file://` root for the rest of the session. A `roots/list_changed`
//! notification clears it, so the next call asks again.
//!
//! [`crate::search::cross_project::with_session_client_dir`] makes the
//! directory visible to scope resolution for the duration of one tool call.
//!
//! Verified 2026-11-10 against Claude Code 2.1.322: its initialize declares
//! `"roots":{"listChanged":true}`, and `roots/list` answers with the session's
//! working directory, e.g. `{"roots":[{"uri":"file:///Users/x/PhpstormProjects/visilaunch"}]}`.
//! Each session logs one `CSR mcp session: client=... dir=...` line.

use std::time::Duration;

use rmcp::model::Root;
use rmcp::service::Peer;
use rmcp::RoleServer;

/// How long a tool call waits for the client to answer `roots/list`.
const ROOTS_TIMEOUT: Duration = Duration::from_secs(3);

/// One MCP session's client directory, fetched once.
#[derive(Default)]
pub struct SessionDir {
    /// `None`: not asked yet. `Some(dir)`: the answer (`dir` may be `None`
    /// when the client has no roots or did not answer).
    state: tokio::sync::Mutex<Option<Option<String>>>,
}

impl SessionDir {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    /// The session's directory, asking `peer` for its roots the first time.
    pub async fn resolve(&self, peer: &Peer<RoleServer>) -> Option<String> {
        self.resolve_with(|| fetch_from_peer(peer)).await
    }

    /// [`Self::resolve`] with the fetch passed in, for tests.
    pub async fn resolve_with<F, Fut>(&self, fetch: F) -> Option<String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Option<String>>,
    {
        let mut state = self.state.lock().await;
        if let Some(dir) = &*state {
            return dir.clone();
        }
        let dir = fetch().await;
        *state = Some(dir.clone());
        dir
    }

    /// Forget the cached answer (the client's roots changed).
    pub async fn invalidate(&self) {
        *self.state.lock().await = None;
    }
}

async fn fetch_from_peer(peer: &Peer<RoleServer>) -> Option<String> {
    let info = peer.peer_info();
    let client = info
        .map(|i| format!("{} {}", i.client_info.name, i.client_info.version))
        .unwrap_or_else(|| "unknown".to_string());
    let declares_roots = info.is_some_and(|i| i.capabilities.roots.is_some());
    let (dir, how) = if declares_roots {
        match tokio::time::timeout(ROOTS_TIMEOUT, peer.list_roots()).await {
            Ok(Ok(result)) => {
                let dir = dir_from_roots(&result.roots);
                let how = if dir.is_some() {
                    "roots"
                } else {
                    "roots (no file root)"
                };
                (dir, how.to_string())
            }
            Ok(Err(e)) => (None, format!("roots/list failed: {e}")),
            Err(_) => (
                None,
                format!("roots/list timed out after {}s", ROOTS_TIMEOUT.as_secs()),
            ),
        }
    } else {
        (None, "client declares no roots".to_string())
    };
    crate::telemetry::append_timing_line(&format!(
        "CSR mcp session: client={client} dir={} ({how})",
        dir.as_deref().unwrap_or("none")
    ));
    dir
}

/// The path of the first `file://` root, percent-decoded.
pub fn dir_from_roots(roots: &[Root]) -> Option<String> {
    roots.iter().find_map(|r| file_uri_to_path(&r.uri))
}

/// `file:///Users/x/My%20App` to `/Users/x/My App`. Only local file URIs
/// (empty or `localhost` host) are accepted.
fn file_uri_to_path(uri: &str) -> Option<String> {
    let rest = uri.strip_prefix("file://")?;
    let path = if rest.starts_with('/') {
        rest
    } else {
        rest.strip_prefix("localhost")?
    };
    if !path.starts_with('/') {
        return None;
    }
    let decoded = percent_decode(path)?;
    let trimmed = decoded.trim_end_matches('/');
    Some(if trimmed.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
    })
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_uris_become_paths() {
        assert_eq!(
            file_uri_to_path("file:///Users/x/proj").as_deref(),
            Some("/Users/x/proj")
        );
        assert_eq!(
            file_uri_to_path("file:///Users/x/My%20App/").as_deref(),
            Some("/Users/x/My App")
        );
        assert_eq!(
            file_uri_to_path("file://localhost/tmp/a").as_deref(),
            Some("/tmp/a")
        );
        assert_eq!(file_uri_to_path("file://otherhost/tmp/a"), None);
        assert_eq!(file_uri_to_path("https://example.com/x"), None);
        assert_eq!(file_uri_to_path("file:///bad%zz"), None);
    }

    #[test]
    fn the_first_file_root_wins() {
        let roots = vec![
            Root::new("https://example.com"),
            Root::new("file:///Users/x/a"),
            Root::new("file:///Users/x/b"),
        ];
        assert_eq!(dir_from_roots(&roots).as_deref(), Some("/Users/x/a"));
        assert_eq!(dir_from_roots(&[]), None);
    }

    #[tokio::test]
    async fn fetches_once_until_invalidated() {
        let s = SessionDir::default();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let fetch = || async {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Some("/Users/x/a".to_string())
        };
        assert_eq!(s.resolve_with(fetch).await.as_deref(), Some("/Users/x/a"));
        assert_eq!(s.resolve_with(fetch).await.as_deref(), Some("/Users/x/a"));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        // A failed answer is cached too, until the client says its roots changed.
        s.invalidate().await;
        assert_eq!(s.resolve_with(|| async { None }).await, None);
        assert_eq!(s.resolve_with(fetch).await, None);
        s.invalidate().await;
        assert_eq!(s.resolve_with(fetch).await.as_deref(), Some("/Users/x/a"));
    }
}
