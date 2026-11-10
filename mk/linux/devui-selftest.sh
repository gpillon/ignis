#!/usr/bin/env bash
# The behavioural test of mk/linux/devui.py, the DEV_UI hook behind `make
# run-ui` / `make dev-ui` on Linux (GitHub #226). It copies the hook
# (mk/os/linux.mk) and its scripts into a temporary tree, beside a throwaway
# Makefile that gives the hook the variables the real one does, and runs
# them on stand-ins: a server whose /ready answers 503 and then 200, and a
# `node` first on PATH in place of Vite. Each stand-in records its arguments,
# its process group and every signal it gets. A case starts make as the
# leader of a session of its own, as a terminal starts a job, and ends it the
# way a user or make would: Ctrl+C to the process group, SIGTERM, SIGHUP,
# kill -9 of a make or of the script, a child exiting on its own; then
# nothing of that session may be left running. No build, no GPU, nothing beyond 127.0.0.1.
# Linux only (/proc, process groups and sessions). It writes only under a
# temporary directory it removes.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1

failed=0
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

python="$(command -v python3)" || { echo "devui-selftest: python3 not found" >&2; exit 1; }
tmp="$(mktemp -d)"
sessions=()

# The live processes of session $1, its id being /proc/<pid>/stat's sixth
# field (state ppid pgrp session after the command name); a zombie is gone.
session_left() {
    local stat line fields
    for stat in /proc/[0-9]*/stat; do
        IFS= read -r line 2>/dev/null < "$stat" || continue
        read -r -a fields <<< "${line##*) }"
        [ "${fields[3]}" = "$1" ] || continue
        case "${fields[0]}" in Z|X) continue ;; esac
        stat="${stat#/proc/}"
        printf '%s ' "${stat%/stat}"
    done
}
cleanup() {
    local s pid
    for s in "${sessions[@]}"; do
        for pid in $(session_left "$s"); do kill -KILL "$pid" 2>/dev/null; done
    done
    rm -rf "$tmp"
}
trap cleanup EXIT

running() { # <pid>: alive and not a zombie
    local line
    IFS= read -r line 2>/dev/null < "/proc/$1/stat" || return 1
    case "${line##*) }" in Z\ *|X\ *) return 1 ;; esac
}
parent_of() { local line fields; IFS= read -r line < "/proc/$1/stat" && read -r -a fields <<< "${line##*) }" && echo "${fields[1]}"; }
wait_until() { # <seconds> <command ...>
    local deadline=$((SECONDS + $1))
    shift
    until "$@"; do
        [ "$SECONDS" -lt "$deadline" ] || return 1
        sleep 0.1
    done
}
free_port() { "$python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
field() { "$python" -c 'import json, sys; v = json.load(open(sys.argv[1]))[sys.argv[2]]; print(v if isinstance(v, str) else json.dumps(v))' "$1" "$2"; }
has() { # <label> <file> <text>
    grep -qF -- "$3" "$2" || fail "$1: no line with '$3' in:
$(cat "$2")"
}
lacks() { # <label> <file> <text>
    ! grep -qF -- "$3" "$2" || fail "$1: '$3' in $2"
}

# The stand-ins and the tree: the hook and its scripts as they are in the repo.
bin="$tmp/bin"
tree="$tmp/repo"
run="$tmp/run"
state="$tmp/state"
mkdir -p "$bin" "$tree/mk/os" "$tree/mk/linux" "$tree/web/node_modules/vite/bin" "$run" "$state" "$tmp/empty"
cp mk/os/linux.mk "$tree/mk/os/"
cp mk/linux/devui.py mk/linux/server.py "$tree/mk/linux/"
: > "$tree/web/node_modules/vite/bin/vite.js"
export STUB_DIR="$state"
port="$(free_port)"
url="http://127.0.0.1:$port"

cat > "$bin/stub-server" <<'EOF'
#!/usr/bin/env python3
# The server's stand-in. STUB_SERVER: never-ready, exit-early (code 7 before
# it serves), ignore-term (records SIGTERM and keeps serving). A signal
# drains for 0.3 s before the exit, so a second one is recorded too.
import http.server, json, os, signal, sys, time
state, mode = os.environ["STUB_DIR"], os.environ.get("STUB_SERVER", "")
def on_signal(signum, _frame):
    with open(os.path.join(state, "server.signals"), "a") as f:
        f.write(signal.Signals(signum).name + "\n")
    if mode != "ignore-term":
        time.sleep(0.3)
        os._exit(0)
signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
info = {"pid": os.getpid(), "pgid": os.getpgrp(), "argv": sys.argv[1:], "log_level": os.environ.get("IGNIS_LOG_LEVEL", "")}
with open(os.path.join(state, "server.tmp"), "w") as f:
    json.dump(info, f)
os.replace(os.path.join(state, "server.tmp"), os.path.join(state, "server.json"))
if mode == "exit-early":
    time.sleep(0.3)
    print("stub: no artifact", file=sys.stderr)
    sys.exit(7)
print("generated API key: sk-selftest", flush=True)
host, port = sys.argv[sys.argv.index("--server-bind") + 1].rsplit(":", 1)
began, after = time.monotonic(), float("inf") if mode == "never-ready" else 0.5
class Ready(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/ready" and time.monotonic() - began >= after else 503)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer((host, int(port)), Ready).serve_forever(poll_interval=0.05)
EOF

cat > "$bin/node" <<'EOF'
#!/usr/bin/env python3
# Vite's stand-in, run as `node vite.js ...`. STUB_VITE=exit-3: exits on its own.
import json, os, signal, sys, time
state = os.environ["STUB_DIR"]
def on_signal(signum, _frame):
    with open(os.path.join(state, "vite.signals"), "a") as f:
        f.write(signal.Signals(signum).name + "\n")
    os._exit(0 if signum == signal.SIGTERM else 128 + signum)
signal.signal(signal.SIGTERM, on_signal)
signal.signal(signal.SIGINT, on_signal)
info = {"pid": os.getpid(), "ppid": os.getppid(), "pgid": os.getpgrp(), "cwd": os.getcwd(),
        "argv": sys.argv[1:], "url": os.environ.get("IGNIS_URL", "")}
with open(os.path.join(state, "vite.tmp"), "w") as f:
    json.dump(info, f)
os.replace(os.path.join(state, "vite.tmp"), os.path.join(state, "vite.json"))
if os.environ.get("STUB_VITE") == "exit-3":
    time.sleep(0.3)
    sys.exit(3)
while True:
    time.sleep(0.05)
EOF
chmod +x "$bin/stub-server" "$bin/node"
export PATH="$bin:$PATH"

# What the real Makefile gives the hook, and its run-ui / dev-ui recipes.
cat > "$tree/Makefile" <<'EOF'
SERVER_BIN := $(SELFTEST)/bin/stub-server
SERVER_URL := http://127.0.0.1:$(PORT)
SERVER_FLAGS = --server-bind 127.0.0.1:$(PORT) --selftest-arg 'two words'
SERVER_PID := $(SELFTEST)/run/ignis-server.pid
SERVER_LOG := $(SELFTEST)/run/ignis-server.log
READY_TIMEOUT ?= 10
SERVER_ENV = IGNIS_LOG_LEVEL=debug
include mk/os/linux.mk
.RECIPEPREFIX := >
.PHONY: run-ui dev-ui
run-ui:
> @$(SERVER_ENV) $(DEV_UI)
dev-ui:
> @$(MAKE) --no-print-directory run-ui
EOF

# launch <case> <command ...>: the command as the leader of a session of its
# own, with SIGINT and SIGQUIT at their defaults (an & job of this script
# ignores both, and make would keep ignoring them), its output in $out, or
# in $console when that is set.
console=
launch() {
    rm -rf "$state"
    mkdir -p "$state"
    out="$tmp/out.$1"
    shift
    "$python" -c 'import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGQUIT, signal.SIG_DFL)
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])' "$@" > "${console:-$out}" 2>&1 &
    leader=$!
    sessions+=("$leader")
}
make_ui() { launch "$1" env ${2:-} make -C "$tree" --no-print-directory SELFTEST="$tmp" PORT="$port" "${3:-run-ui}"; }
served() { [ -f "$state/vite.json" ] && grep -qF "Ctrl+C stops the server and Vite together" "$out"; }
ended() { ! running "$leader"; }
# settle <case>: the session's leader has exited and nothing of the session is
# left; whatever is, is killed so the next case starts clean.
settle() {
    wait_until 30 ended || fail "$1: the session's leader (pid $leader) is still running"
    wait_until 10 eval '[ -z "$(session_left "$leader")" ]' || fail "$1: still running: $(session_left "$leader")"
    kill -KILL $(session_left "$leader") 2>/dev/null
    wait "$leader" 2>/dev/null
    rc=$?
}
# abort <case>: it never served; say why, and kill what it started.
abort() {
    fail "$1: the session never served:
$(cat "$out")"
    kill -KILL $(session_left "$leader") 2>/dev/null
    wait "$leader" 2>/dev/null
}
signals() { tr '\n' ' ' 2>/dev/null < "$state/$1.signals" | sed 's/ $//'; }

echo "devui-selftest: Ctrl+C"
make_ui ctrl-c
if wait_until 20 served; then
    server_pid="$(field "$state/server.json" pid)"
    devui_pid="$(field "$state/vite.json" ppid)"
    [ "$(parent_of "$devui_pid")" = "$leader" ] || fail "ctrl-c: a process sits between make and the script (exec)"
    [ "$(field "$state/server.json" pgid)" = "$server_pid" ] || fail "ctrl-c: the server is not in a process group of its own"
    [ "$(field "$state/vite.json" pgid)" = "$leader" ] || fail "ctrl-c: vite is not in make's process group"
    [ "$(field "$state/server.json" log_level)" = debug ] || fail "ctrl-c: the server did not get SERVER_ENV"
    [ "$(field "$state/server.json" argv)" = '["--server-bind", "127.0.0.1:'"$port"'", "--selftest-arg", "two words"]' ] ||
        fail "ctrl-c: the server's arguments: $(field "$state/server.json" argv)"
    [ "$(field "$state/vite.json" cwd)" = "$tree/web" ] || fail "ctrl-c: vite runs in $(field "$state/vite.json" cwd)"
    [ "$(field "$state/vite.json" argv)" = '["'"$tree"'/web/node_modules/vite/bin/vite.js", "--clearScreen", "false"]' ] ||
        fail "ctrl-c: vite's arguments: $(field "$state/vite.json" argv)"
    [ "$(field "$state/vite.json" url)" = "$url" ] || fail "ctrl-c: IGNIS_URL is $(field "$state/vite.json" url)"
    has "ctrl-c" "$out" "API key (generated for this run): sk-selftest"
    has "ctrl-c" "$out" "ready: $url (pid $server_pid,"
    # make status and make start read the record the session wrote.
    "$python" "$tree/mk/linux/server.py" status --pid "$run/ignis-server.pid" --log "$run/ignis-server.log" --url "$url" > "$tmp/status" 2>&1
    has "ctrl-c: make status" "$tmp/status" "process: running pid $server_pid (managed)"
    # A second session while this one runs: the same pid file, then the same port.
    other="$(free_port)"
    STUB_DIR="$tmp/second" timeout 20 "$python" "$tree/mk/linux/devui.py" --pid "$run/ignis-server.pid" \
        --log "$tmp/second.log" --url "http://127.0.0.1:$other" -- "$bin/stub-server" --server-bind "127.0.0.1:$other" > "$tmp/second" 2>&1 &&
        fail "a second session on the same pid file started"
    has "second session, same pid file" "$tmp/second" "error: ignis-server is already running (pid $server_pid); make stop first"
    STUB_DIR="$tmp/second" timeout 20 "$python" "$tree/mk/linux/devui.py" --pid "$tmp/second.pid" \
        --log "$tmp/second.log" --url "$url" -- "$bin/stub-server" --server-bind "127.0.0.1:$port" > "$tmp/second" 2>&1 &&
        fail "a second session on the same port started"
    has "second session, same port" "$tmp/second" "error: $url is already listening; refusing to start a second server"
    [ ! -e "$tmp/second.log" ] || fail "a refused session started a server"
    # A terminal's Ctrl+C: SIGINT to the foreground process group.
    kill -INT -- "-$leader"
    settle ctrl-c
    [ "$(signals server)" = SIGTERM ] || fail "ctrl-c: the server got '$(signals server)', not one SIGTERM"
    [ "$(signals vite)" = SIGINT ] || [ "$(signals vite)" = "SIGINT SIGTERM" ] || fail "ctrl-c: vite got '$(signals vite)'"
    has "ctrl-c" "$out" "Ctrl+C: stopping vite and ignis-server"
    has "ctrl-c" "$out" "ignis-server (pid $server_pid) stopped"
    [ ! -e "$run/ignis-server.pid" ] || fail "ctrl-c: the pid record was left behind"
else
    abort "ctrl-c"
fi

echo "devui-selftest: SIGTERM to make"
make_ui sigterm
if wait_until 20 served; then
    ls "$run"/ignis-server.log.* > /dev/null 2>&1 || fail "sigterm: the previous session's log was not kept"
    kill -TERM "$leader"
    settle sigterm
    [ "$(signals server)" = SIGTERM ] || fail "sigterm: the server got '$(signals server)', not one SIGTERM"
    [ "$(signals vite)" = SIGTERM ] || fail "sigterm: vite got '$(signals vite)'"
    has "sigterm" "$out" "SIGTERM: stopping vite and ignis-server"
else
    abort "sigterm"
fi

echo "devui-selftest: the server dies"
make_ui server-dies
if wait_until 20 served; then
    kill -KILL "$(field "$state/server.json" pid)"
    settle server-dies
    [ "$rc" -ne 0 ] || fail "server-dies: make exited 0"
    [ "$(signals vite)" = SIGTERM ] || fail "server-dies: vite got '$(signals vite)'"
    has "server-dies" "$out" "ignis-server exited (signal SIGKILL); stopping vite"
    has "server-dies" "$out" "vite (pid $(field "$state/vite.json" pid)) stopped"
    has "server-dies: the log tail" "$out" "generated API key: sk-selftest"
    [ ! -e "$run/ignis-server.pid" ] || fail "server-dies: the pid record was left behind"
else
    abort "server-dies"
fi

echo "devui-selftest: vite exits"
make_ui vite-exits STUB_VITE=exit-3
if wait_until 20 served; then settle vite-exits; else abort vite-exits; fi
[ "$(signals server)" = SIGTERM ] || fail "vite-exits: the server got '$(signals server)', not one SIGTERM"
has "vite-exits" "$out" "vite exited (code 3); stopping the server"
has "vite-exits" "$out" "ignis-server (pid $(field "$state/server.json" pid)) stopped"
has "vite-exits: make" "$out" "Error 3"

# kill -9 of the make that runs the script, under a shell that outlives it
# (a script running `make run-ui`): the script's parent is gone, its process
# group's leader is not, until the case releases it.
echo "devui-selftest: kill -9 of make, its caller alive"
launch kill-parent bash -c 'make -C "$1" --no-print-directory SELFTEST="$2" PORT="$3" run-ui
until [ -e "$2/release" ]; do sleep 0.1; done' - "$tree" "$tmp" "$port"
if wait_until 20 served; then
    devui_pid="$(field "$state/vite.json" ppid)"
    kill -KILL "$(parent_of "$devui_pid")"
    wait_until 10 eval '! running "$devui_pid"' || fail "kill-parent: the script is still running"
    running "$(field "$state/server.json" pid)" && fail "kill-parent: the server is still running"
    running "$(field "$state/vite.json" pid)" && fail "kill-parent: vite is still running"
    running "$leader" || fail "kill-parent: the caller's shell is gone, so this case tests nothing"
    touch "$tmp/release"
    settle kill-parent
    rm -f "$tmp/release"
    [ "$(signals server)" = SIGTERM ] || fail "kill-parent: the server got '$(signals server)', not one SIGTERM"
    [ "$(signals vite)" = SIGTERM ] || fail "kill-parent: vite got '$(signals vite)'"
    has "kill-parent" "$out" "make is gone: stopping vite and ignis-server"
else
    abort kill-parent
fi

# kill -9 of the make the terminal started, `make dev-ui`, whose second make
# (run-ui's, the script's parent) lives on until the script exits.
echo "devui-selftest: kill -9 of the dev-ui make"
make_ui kill-leader "" dev-ui
if wait_until 20 served; then
    # Disowned first: bash would report its own job Killed.
    disown "$leader"
    kill -KILL "$leader"
    settle kill-leader
    [ "$(signals server)" = SIGTERM ] || fail "kill-leader: the server got '$(signals server)', not one SIGTERM"
    [ "$(signals vite)" = SIGTERM ] || fail "kill-leader: vite got '$(signals vite)'"
    has "kill-leader" "$out" "make is gone: stopping vite and ignis-server"
else
    abort kill-leader
fi

# The terminal closed: SIGHUP to the process group, and the console gone, so
# every line the script still writes fails (here a pipe nobody reads any
# more); the session stops both and drops its pid record all the same.
echo "devui-selftest: the terminal closes"
mkfifo "$tmp/console"
console="$tmp/console"
make_ui hup
console=
exec 4< "$tmp/console"
if wait_until 20 eval '[ -f "$state/vite.json" ]'; then
    exec 4<&-
    # Disowned first: bash would report its own job Hangup.
    disown "$leader"
    kill -HUP -- "-$leader"
    settle hup
    [ "$(signals server)" = SIGTERM ] || fail "hup: the server got '$(signals server)', not one SIGTERM"
    [ ! -e "$run/ignis-server.pid" ] || fail "hup: the pid record was left behind"
else
    exec 4<&-
    abort hup
fi

echo "devui-selftest: kill -9 of the script"
make_ui kill-script
if wait_until 20 served; then
    kill -KILL "$(field "$state/vite.json" ppid)"
    settle kill-script
    [ "$(signals server)" = SIGTERM ] || fail "kill-script: the server got '$(signals server)', not one SIGTERM"
    [ "$(signals vite)" = SIGTERM ] || fail "kill-script: vite got '$(signals vite)'"
    "$python" "$tree/mk/linux/server.py" status --pid "$run/ignis-server.pid" --log "$run/ignis-server.log" --url "$url" > "$tmp/status" 2>&1
    has "kill-script: make status" "$tmp/status" "process: not running (no managed server)"
    rm -f "$run/ignis-server.pid"
else
    abort "kill-script"
fi

# A server that drains past the grace: a second Ctrl+C is its second signal,
# and what still runs at the end of the grace is killed.
echo "devui-selftest: a second Ctrl+C"
launch second-ctrl-c env STUB_SERVER=ignore-term "$python" "$tree/mk/linux/devui.py" --pid "$run/ignis-server.pid" \
    --log "$run/ignis-server.log" --url "$url" --stop-timeout 3 -- "$bin/stub-server" --server-bind "127.0.0.1:$port"
if wait_until 20 served; then
    server_pid="$(field "$state/server.json" pid)"
    kill -INT -- "-$leader"
    wait_until 10 eval '[ "$(signals server)" = SIGTERM ]' || fail "second-ctrl-c: the server got '$(signals server)' after the first"
    kill -INT -- "-$leader"
    settle second-ctrl-c
    [ "$(signals server)" = "SIGTERM SIGTERM" ] || fail "second-ctrl-c: the server got '$(signals server)'"
    has "second-ctrl-c" "$out" "ignis-server (pid $server_pid) killed"
    [ "$rc" -eq 130 ] || fail "second-ctrl-c: the script exited $rc, not 130"
else
    abort "second-ctrl-c"
fi

echo "devui-selftest: not ready in time"
launch not-ready env STUB_SERVER=never-ready make -C "$tree" --no-print-directory SELFTEST="$tmp" PORT="$port" READY_TIMEOUT=1 run-ui
settle not-ready
[ "$rc" -ne 0 ] || fail "not-ready: make exited 0"
has "not-ready" "$out" "error: not ready after 1s"
[ "$(signals server)" = SIGTERM ] || fail "not-ready: the server got '$(signals server)', not one SIGTERM"
[ ! -e "$state/vite.json" ] || fail "not-ready: vite started"

echo "devui-selftest: the server exits before it is ready"
make_ui exit-early STUB_SERVER=exit-early
settle exit-early
[ "$rc" -ne 0 ] || fail "exit-early: make exited 0"
has "exit-early" "$out" "error: ignis-server exited with code 7 before it was ready"
has "exit-early: the log tail" "$out" "stub: no artifact"
[ ! -e "$state/vite.json" ] || fail "exit-early: vite started"

echo "devui-selftest: no node, no vite.js"
devui_args=(--pid "$tmp/refused.pid" --log "$tmp/refused.log" --url "$url" -- "$bin/stub-server" --server-bind "127.0.0.1:$port")
PATH="$tmp/empty" "$python" "$tree/mk/linux/devui.py" "${devui_args[@]}" > "$tmp/refused" 2>&1 && fail "no node: the script exited 0"
has "no node" "$tmp/refused" "error: node not found on PATH"
mkdir -p "$tmp/bare/mk/linux"
cp mk/linux/devui.py mk/linux/server.py "$tmp/bare/mk/linux/"
"$python" "$tmp/bare/mk/linux/devui.py" "${devui_args[@]}" > "$tmp/refused" 2>&1 && fail "no vite.js: the script exited 0"
has "no vite.js" "$tmp/refused" "$tmp/bare/web/node_modules/vite/bin/vite.js not found (make web-install)"
[ ! -e "$tmp/refused.log" ] || fail "a refused session started a server"
lacks "the tree" <(ls -a "$tree/mk/linux") "__pycache__"

if [ "$failed" -ne 0 ]; then
    printf 'devui-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "devui-selftest: ok"
