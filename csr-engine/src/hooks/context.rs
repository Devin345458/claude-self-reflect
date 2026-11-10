//! Per-request context for a hook handler.
//!
//! A hook used to be its own process: it read settings from its own
//! environment, printed its injection to stdout, and imported its transcript
//! itself. Inside the daemon one process serves every session, so each
//! forwarded request carries the session's environment, and the handler's
//! output is collected here and sent back to the hook client to print.
//!
//! Outside a context (tests, the local session-briefing path) every function
//! falls back to the process: `std::env` and real stdout.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

/// Environment variables a hook client forwards to the daemon: the `CSR_*`
/// switches and the two variables that locate the session's project. Nothing
/// else, so no credential in the session's environment leaves the hook.
pub fn forwarded_env_var(name: &str) -> bool {
    name.starts_with("CSR_") || name == "CLAUDE_PROJECT_DIR" || name == "MCP_CLIENT_CWD"
}

pub struct HookContext {
    env: HashMap<String, String>,
    out: Mutex<String>,
    in_daemon: bool,
}

tokio::task_local! {
    static CONTEXT: Arc<HookContext>;
}

impl HookContext {
    /// A context for a hook forwarded to the daemon, with the session's
    /// environment.
    pub fn forwarded(env: HashMap<String, String>) -> Arc<Self> {
        Arc::new(Self {
            env,
            out: Mutex::new(String::new()),
            in_daemon: true,
        })
    }

    /// Run `fut` with this context active.
    pub async fn scope<F: Future>(self: &Arc<Self>, fut: F) -> F::Output {
        CONTEXT.scope(self.clone(), fut).await
    }

    /// Everything the handler printed.
    pub fn take_output(&self) -> String {
        std::mem::take(&mut *self.out.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

/// The session's value of `name` for a forwarded hook (falling back to the
/// daemon's own environment when the session did not set it), or this
/// process's value outside a context.
pub fn env_var(name: &str) -> Option<String> {
    CONTEXT
        .try_with(|c| c.env.get(name).cloned())
        .ok()
        .flatten()
        .or_else(|| std::env::var(name).ok())
}

/// Whether this handler is running inside the daemon for a forwarded hook.
pub fn in_daemon() -> bool {
    CONTEXT.try_with(|c| c.in_daemon).unwrap_or(false)
}

/// Write hook output: into the request's buffer inside a context, to stdout
/// otherwise.
pub fn emit(text: &str) {
    let captured = CONTEXT
        .try_with(|c| {
            c.out
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push_str(text)
        })
        .is_ok();
    if !captured {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }
}

/// `print!` for hook output. See [`emit`].
#[macro_export]
macro_rules! hook_print {
    ($($arg:tt)*) => {
        $crate::hooks::context::emit(&format!($($arg)*))
    };
}

/// `println!` for hook output. See [`emit`].
#[macro_export]
macro_rules! hook_println {
    () => {
        $crate::hooks::context::emit("\n")
    };
    ($($arg:tt)*) => {
        $crate::hooks::context::emit(&format!("{}\n", format_args!($($arg)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn output_and_env_are_per_request() {
        let a = HookContext::forwarded(HashMap::from([("CSR_X".to_string(), "a".to_string())]));
        let b = HookContext::forwarded(HashMap::new());
        a.scope(async {
            assert!(in_daemon());
            assert_eq!(env_var("CSR_X").as_deref(), Some("a"));
            crate::hook_print!("one ");
            crate::hook_println!("two {}", 2);
        })
        .await;
        b.scope(async {
            assert_eq!(env_var("CSR_X"), std::env::var("CSR_X").ok());
            crate::hook_println!("other");
        })
        .await;
        assert_eq!(a.take_output(), "one two 2\n");
        assert_eq!(b.take_output(), "other\n");
        assert!(!in_daemon(), "no context outside a scope");
    }

    #[test]
    fn forwards_csr_switches_and_project_location_only() {
        assert!(forwarded_env_var("CSR_NO_ANTI_PATTERNS"));
        assert!(forwarded_env_var("CLAUDE_PROJECT_DIR"));
        assert!(forwarded_env_var("MCP_CLIENT_CWD"));
        assert!(!forwarded_env_var("PATH"));
        assert!(!forwarded_env_var("ANTHROPIC_API_KEY"));
        assert!(!forwarded_env_var("CLAUDE_CODE_OAUTH_TOKEN"));
    }
}
