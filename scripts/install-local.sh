#!/bin/zsh
# Build csr-engine from this checkout and install it as the running daemon.
#
#   scripts/install-local.sh            build, back up, swap, restart, verify
#   scripts/install-local.sh --rollback <backup dir>
#
# Order matters. launchd caches a binary's code signature by path, so replacing
# the file under a running job gets the next launch killed with "Code Signature
# Invalid". The job is booted out first, the new binary is staged beside the
# target, cleared and ad-hoc signed, renamed into place (hooks never see a
# missing binary), and only then is the job bootstrapped again.
set -euo pipefail

LABEL=com.devin.csr-daemon
DOMAIN="gui/$(id -u)"
BIN="$HOME/.local/bin/csr-engine"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"

stop_daemon() {
  # The PID launchd reports, not a pgrep pattern: any shell whose command line
  # mentions the daemon would match a pattern.
  local pid
  pid=$(launchctl print "$DOMAIN/$LABEL" 2>/dev/null | awk '$1 == "pid" {print $3; exit}')
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  [[ -z "$pid" ]] && return 0
  for _ in {1..60}; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  echo "daemon pid $pid did not exit after bootout" >&2
  return 1
}

swap_in() { # $1 = binary to install
  local stage="$BIN.new"
  cp "$1" "$stage"
  xattr -cr "$stage"
  codesign --force --sign - "$stage"
  mv -f "$stage" "$BIN"
}

wait_ready() {
  for _ in {1..120}; do
    if out=$("$BIN" daemon-ping 2>/dev/null); then
      echo "$out"
      return 0
    fi
    sleep 0.5
  done
  echo "daemon did not answer within 60s; log: ~/Library/Logs/claude-self-reflect/csr-daemon.log" >&2
  return 1
}

if [[ "${1:-}" == "--rollback" ]]; then
  dir="${2:?usage: install-local.sh --rollback <backup dir>}"
  stop_daemon
  swap_in "$dir/csr-engine"
  [[ -f "$dir/$LABEL.plist" ]] && cp "$dir/$LABEL.plist" "$PLIST"
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "rolled back to $dir"
  exit 0
fi

cd "$REPO/csr-engine"
cargo build --release
NEW="$REPO/csr-engine/target/release/csr-engine"
"$NEW" --version

BACKUP="$HOME/.claude-self-reflect/backups/$(date +%Y-%m-%d-%H%M%S)-install"
mkdir -p "$BACKUP"
[[ -f "$BIN" ]] && cp -p "$BIN" "$BACKUP/csr-engine"
[[ -f "$PLIST" ]] && cp -p "$PLIST" "$BACKUP/"
echo "backup: $BACKUP"

stop_daemon
swap_in "$NEW"
"$BIN" agent-plist --binary "$BIN"
launchctl bootstrap "$DOMAIN" "$PLIST"
wait_ready
echo "installed $("$BIN" --version)"
echo "rollback: $0 --rollback $BACKUP"
