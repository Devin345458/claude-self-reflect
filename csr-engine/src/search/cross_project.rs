use crate::import;

/// Resolve project name from a CWD path string.
/// Pure function — testable without environment variable manipulation.
pub fn resolve_project_from_cwd(cwd: &str) -> Option<String> {
    if cwd.is_empty() {
        return None;
    }

    let path = std::path::Path::new(cwd);
    let dir_name = path.file_name()?.to_string_lossy().to_string();

    // If it looks like a Claude projects directory (dash-separated), normalize it
    if dir_name.starts_with('-') && dir_name.contains("projects") {
        return Some(import::normalize_project_name(&dir_name));
    }

    // Walk up path components to find one after "projects"
    // This handles subdirectories like /Users/name/projects/my-app/src/engine
    let components: Vec<&str> = cwd.split('/').filter(|s| !s.is_empty()).collect();
    for (i, comp) in components.iter().enumerate() {
        if *comp == "projects" && i + 1 < components.len() {
            return Some(components[i + 1].to_string());
        }
    }

    // Fallback: last path component
    Some(dir_name)
}

tokio::task_local! {
    /// The calling MCP session's directory, set for the duration of one tool
    /// call by [`with_session_client_dir`].
    static SESSION_CLIENT_DIR: String;
}

/// Run `fut` with `dir` as the client directory every scope lookup inside it
/// sees. `None` runs `fut` unchanged (lookups fall back to the environment).
/// The daemon's MCP server wraps every tool call in this with the session's
/// MCP root (see `crate::mcp::session`).
pub async fn with_session_client_dir<F: std::future::Future>(
    dir: Option<String>,
    fut: F,
) -> F::Output {
    match dir.filter(|d| !d.trim().is_empty()) {
        Some(d) => SESSION_CLIENT_DIR.scope(d, fut).await,
        None => fut.await,
    }
}

/// The project of the client's directory (see [`resolve_client_dir`]).
pub fn resolve_current_project() -> Option<String> {
    resolve_project_from_cwd(&resolve_client_dir()?)
}

/// The directory the MCP client is working in, first match wins:
/// 1. the calling MCP session's root ([`with_session_client_dir`]);
/// 2. `MCP_CLIENT_CWD`, then `CLAUDE_PROJECT_DIR`, from the session's
///    forwarded environment inside a hook, else this process's environment.
///
/// Claude Code exports `CLAUDE_PROJECT_DIR` to stdio MCP servers and has never
/// set `MCP_CLIENT_CWD`. Over HTTP neither reaches the daemon, whose own
/// environment belongs to no session; the session root is what scopes there.
pub fn resolve_client_dir() -> Option<String> {
    if let Ok(dir) = SESSION_CLIENT_DIR.try_with(|d| d.clone()) {
        return Some(dir);
    }
    client_dir_from(
        crate::hooks::context::env_var("MCP_CLIENT_CWD"),
        crate::hooks::context::env_var("CLAUDE_PROJECT_DIR"),
    )
}

/// Env-free core of [`resolve_client_dir`]: the first value that is not blank.
fn client_dir_from(
    mcp_client_cwd: Option<String>,
    claude_project_dir: Option<String>,
) -> Option<String> {
    [mcp_client_cwd, claude_project_dir]
        .into_iter()
        .flatten()
        .find(|dir| !dir.trim().is_empty())
}

/// Normalize a project scope parameter.
///
/// - `None` → auto-detect from the client directory ([`resolve_client_dir`])
/// - `Some("all")` (any case) → `None` (search all projects)
/// - `Some(name)` → `Some(name)` (specific project)
///
/// Returns `(effective_project, scope_label)` where scope_label is for display.
pub fn normalize_project_scope(project: Option<&str>) -> (Option<String>, String) {
    match project {
        Some(p) if p.eq_ignore_ascii_case("all") => (None, "all".to_string()),
        Some(p) if !p.is_empty() => (Some(p.to_string()), p.to_string()),
        _ => {
            // Auto-detect from environment
            match resolve_current_project() {
                Some(p) => {
                    let label = p.clone();
                    (Some(p), label)
                }
                None => (None, "all".to_string()),
            }
        }
    }
}

/// Encode a path the way Claude Code names its `~/.claude/projects` folders.
///
/// Claude Code does `path.replace(/[^a-zA-Z0-9]/g, "-")` (verified in the shipped
/// binary, 2.1.226). That regex carries no `u` flag, so it runs over UTF-16 code
/// units, not scalar values: an accented BMP character costs one dash, but a
/// non-BMP one (emoji, rarer CJK) is a surrogate pair and costs *two*.
pub(crate) fn encode_project_folder(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else {
            for _ in 0..c.len_utf16() {
                out.push('-');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_scope_all() {
        let (project, label) = normalize_project_scope(Some("all"));
        assert!(project.is_none());
        assert_eq!(label, "all");
    }

    #[test]
    fn test_normalize_scope_all_case_insensitive() {
        let (project, _) = normalize_project_scope(Some("ALL"));
        assert!(project.is_none());
        let (project, _) = normalize_project_scope(Some("All"));
        assert!(project.is_none());
    }

    #[test]
    fn test_normalize_scope_specific() {
        let (project, label) = normalize_project_scope(Some("my-project"));
        assert_eq!(project, Some("my-project".to_string()));
        assert_eq!(label, "my-project");
    }

    #[test]
    fn test_normalize_scope_none_no_env() {
        // When MCP_CLIENT_CWD is not set, falls back to "all"
        let (_project, label) = normalize_project_scope(None);
        assert!(!label.is_empty());
    }

    #[test]
    fn test_normalize_scope_empty() {
        let (project, label) = normalize_project_scope(Some(""));
        assert_eq!(
            label,
            if project.is_some() {
                project.as_deref().unwrap()
            } else {
                "all"
            }
        );
    }

    #[test]
    fn the_client_directory_prefers_mcp_client_cwd_and_falls_back_to_claude_project_dir() {
        let dir = |s: &str| Some(s.to_string());
        assert_eq!(client_dir_from(dir("/a"), dir("/b")), dir("/a"));
        assert_eq!(client_dir_from(None, dir("/b")), dir("/b"));
        assert_eq!(client_dir_from(dir("  "), dir("/b")), dir("/b"));
        assert_eq!(client_dir_from(None, dir("")), None);
        assert_eq!(client_dir_from(None, None), None);
    }

    // Pure function tests — no env var manipulation needed
    #[test]
    fn test_resolve_from_cwd_simple_project() {
        let result = resolve_project_from_cwd("/Users/name/projects/claude-self-reflect");
        assert_eq!(result, Some("claude-self-reflect".to_string()));
    }

    #[test]
    fn test_resolve_from_cwd_subdirectory() {
        let result =
            resolve_project_from_cwd("/Users/name/projects/claude-self-reflect/src/engine");
        assert_eq!(result, Some("claude-self-reflect".to_string()));
    }

    #[test]
    fn test_resolve_from_cwd_claude_dir_format() {
        let result = resolve_project_from_cwd(
            "/Users/name/.claude/projects/-Users-name-projects-claude-self-reflect",
        );
        assert_eq!(result, Some("claude-self-reflect".to_string()));
    }

    #[test]
    fn test_resolve_from_cwd_empty() {
        assert_eq!(resolve_project_from_cwd(""), None);
    }

    #[test]
    fn test_resolve_from_cwd_no_projects_segment() {
        let result = resolve_project_from_cwd("/tmp/something/mydir");
        assert_eq!(result, Some("mydir".to_string()));
    }

    #[tokio::test]
    async fn a_session_directory_wins_and_stays_inside_its_call() {
        let a = with_session_client_dir(Some("/Users/x/PhpstormProjects/alpha".into()), async {
            (
                resolve_client_dir(),
                resolve_current_project(),
                normalize_project_scope(None),
            )
        })
        .await;
        assert_eq!(a.0.as_deref(), Some("/Users/x/PhpstormProjects/alpha"));
        assert_eq!(a.1.as_deref(), Some("alpha"));
        assert_eq!(a.2, (Some("alpha".to_string()), "alpha".to_string()));

        // An explicit project still wins over the session's own.
        let explicit =
            with_session_client_dir(Some("/Users/x/PhpstormProjects/alpha".into()), async {
                normalize_project_scope(Some("beta"))
            })
            .await;
        assert_eq!(explicit.0.as_deref(), Some("beta"));

        // Outside the call the session directory is gone.
        let outside = resolve_client_dir();
        assert_ne!(outside.as_deref(), Some("/Users/x/PhpstormProjects/alpha"));
    }

    #[tokio::test]
    async fn concurrent_sessions_never_see_each_others_directory() {
        let run = |dir: &'static str| {
            tokio::spawn(with_session_client_dir(Some(dir.to_string()), async move {
                let mut seen = Vec::new();
                for _ in 0..50 {
                    seen.push(resolve_current_project());
                    tokio::task::yield_now().await;
                }
                seen
            }))
        };
        let (a, b) = (run("/Users/x/p/alpha"), run("/Users/x/p/beta"));
        let (a, b) = (a.await.unwrap(), b.await.unwrap());
        assert!(a.iter().all(|p| p.as_deref() == Some("alpha")));
        assert!(b.iter().all(|p| p.as_deref() == Some("beta")));
    }

    #[tokio::test]
    async fn no_session_directory_falls_back_to_the_environment() {
        let inside = with_session_client_dir(None, async { resolve_client_dir() }).await;
        assert_eq!(inside, resolve_client_dir());
        let blank =
            with_session_client_dir(Some("  ".into()), async { resolve_client_dir() }).await;
        assert_eq!(blank, resolve_client_dir());
    }
}
