//! The macOS LaunchAgent that keeps the daemon running.
//!
//! The daemon is the only process holding the vectors, and every hook and MCP
//! call goes through it, so it has to be up whenever Claude Code is. launchd
//! starts it at login and restarts it if it exits.
//!
//! The agent runs as `ProcessType Interactive`, the scheduling footing of an
//! app. It used to run as `Background` with `LowPriorityIO` (startup 60 to
//! 75 s against 4 s at normal priority). `Standard` still left it at priority
//! 20 against 31 for ordinary processes, so during a load spike (iOS
//! simulators booting, load average 340 on 2026-11-10) a prompt-submit took
//! up to 8.5 s. Hooks wait on this process, and it is idle between bursts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// launchd label. Kept from the hand-written agent this replaces, so an
/// install updates that job instead of starting a second daemon beside it.
pub const LABEL: &str = "com.devin.csr-daemon";

pub fn plist_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join("Library/LaunchAgents")
        .join(format!("{LABEL}.plist"))
}

fn log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join("Library/Logs/claude-self-reflect/csr-daemon.log")
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The agent's plist. `env` becomes its `EnvironmentVariables`.
pub fn render_plist(
    binary: &Path,
    home: &Path,
    log: &Path,
    env: &BTreeMap<String, String>,
) -> String {
    let mut env_xml = String::new();
    for (k, v) in env {
        env_xml.push_str(&format!(
            "\t\t<key>{}</key>\n\t\t<string>{}</string>\n",
            xml_escape(k),
            xml_escape(v)
        ));
    }
    let bin = xml_escape(&binary.to_string_lossy());
    let log = xml_escape(&log.to_string_lossy());
    let home = xml_escape(&home.to_string_lossy());
    let addr = crate::setup::DAEMON_MCP_ADDR;
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{LABEL}</string>
	<key>ProgramArguments</key>
	<array>
		<string>{bin}</string>
		<string>daemon</string>
		<string>--serve-http</string>
		<string>{addr}</string>
	</array>
	<key>EnvironmentVariables</key>
	<dict>
{env_xml}	</dict>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>ThrottleInterval</key>
	<integer>10</integer>
	<key>ProcessType</key>
	<string>Interactive</string>
	<key>WorkingDirectory</key>
	<string>{home}</string>
	<key>StandardOutPath</key>
	<string>{log}</string>
	<key>StandardErrorPath</key>
	<string>{log}</string>
</dict>
</plist>
"#
    )
}

/// The `EnvironmentVariables` of the plist already installed, if it has any.
/// Read with `plutil` so any valid plist encoding works.
fn existing_env(path: &Path) -> Option<BTreeMap<String, String>> {
    if !path.exists() {
        return None;
    }
    let out = std::process::Command::new("plutil")
        .args(["-extract", "EnvironmentVariables", "json", "-o", "-"])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// Environment for a first install: HOME and PATH from this process, plus
/// `ANTHROPIC_BASE_URL` when set (a local API proxy the AI narratives use).
fn fresh_env() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    for k in ["HOME", "PATH", "ANTHROPIC_BASE_URL"] {
        if let Ok(v) = std::env::var(k) {
            env.insert(k.to_string(), v);
        }
    }
    env
}

/// Write the agent's plist for `binary`. Keeps the environment of a plist that
/// is already there. Returns the path written. Does not touch launchd: the
/// install script boots the job out and back in around the binary swap.
pub fn write_plist(binary: &Path) -> Result<PathBuf> {
    let path = plist_path();
    let env = existing_env(&path).unwrap_or_else(fresh_env);
    let home = dirs::home_dir().context("no home directory")?;
    let log = log_path();
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let body = render_plist(binary, &home, &log, &env);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if path.exists() {
        let backup = path.with_extension("plist.bak");
        std::fs::copy(&path, &backup).context("backing up the existing plist")?;
    }
    let tmp = path.with_extension("plist.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_runs_the_daemon_at_interactive_priority_with_the_given_env() {
        let env = BTreeMap::from([
            ("PATH".to_string(), "/a:/b".to_string()),
            (
                "ANTHROPIC_BASE_URL".to_string(),
                "http://localhost:3456".to_string(),
            ),
        ]);
        let p = render_plist(
            Path::new("/Users/x/.local/bin/csr-engine"),
            Path::new("/Users/x"),
            Path::new("/Users/x/Library/Logs/claude-self-reflect/csr-daemon.log"),
            &env,
        );
        assert!(p.contains(
            "<string>/Users/x/.local/bin/csr-engine</string>\n\t\t<string>daemon</string>"
        ));
        assert!(p.contains(&format!(
            "<string>{}</string>",
            crate::setup::DAEMON_MCP_ADDR
        )));
        assert!(p.contains("<key>ProcessType</key>\n\t<string>Interactive</string>"));
        assert!(!p.contains("Background"));
        assert!(!p.contains("LowPriorityIO"));
        assert!(
            p.contains("<key>ANTHROPIC_BASE_URL</key>\n\t\t<string>http://localhost:3456</string>")
        );
        assert!(p.contains(&format!("<string>{LABEL}</string>")));
    }

    #[test]
    fn plist_escapes_xml() {
        let env = BTreeMap::from([("X".to_string(), "a&b<c>".to_string())]);
        let p = render_plist(Path::new("/b"), Path::new("/h"), Path::new("/l"), &env);
        assert!(p.contains("<string>a&amp;b&lt;c&gt;</string>"));
    }

    #[test]
    fn plist_is_valid_for_plutil() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("t.plist");
        std::fs::write(
            &f,
            render_plist(
                Path::new("/b"),
                Path::new("/h"),
                Path::new("/l"),
                &fresh_env(),
            ),
        )
        .unwrap();
        let ok = std::process::Command::new("plutil")
            .arg("-lint")
            .arg(&f)
            .output();
        if let Ok(out) = ok {
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stdout)
            );
        }
        let env = existing_env(&f).expect("env readable back");
        assert_eq!(env, fresh_env());
    }
}
