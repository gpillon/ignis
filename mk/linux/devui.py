#!/usr/bin/env python3
# `make run-ui` / `make dev-ui` on Linux (the DEV_UI hook, mk/os/linux.mk,
# GitHub #226): ignis-server plus the Playground's Vite dev server (hot
# reload, /v1 proxied to the server) as ONE foreground session, the
# counterpart of mk/windows/devui.ps1. Its arguments are server.py start's:
# --pid --log --url --timeout, then -- the server and its flags.
#
# Everything launched here dies with the session:
#   - Vite stays in make's process group, the terminal's foreground job: it
#     reads the terminal (its shortcuts) and a Ctrl+C reaches it. The server
#     gets a process group of its own, so the one signal it receives is this
#     script's SIGTERM, its graceful shutdown: it takes a second signal for
#     "exit now" (crates/server/src/shutdown.rs), and a Ctrl+C reaching it
#     beside a forwarded SIGTERM would be one. A second Ctrl+C is forwarded
#     as that second signal, as it reaches the server on Windows.
#   - Ctrl+C, SIGTERM (make passes its own on) and SIGHUP (the terminal
#     closed) end the session, and so does make dying of a signal it cannot
#     pass on (kill -9): the loops below watch this script's parent, the make
#     that runs run-ui (the hook execs this script, so no shell sits between),
#     and its process group's leader, the make the terminal started
#     (`make dev-ui` runs a second make for run-ui).
#   - both children get SIGTERM if this script is killed (PR_SET_PDEATHSIG),
#     the kill-on-close job object's part on Windows.
#   - if either child exits on its own, the other is stopped too.
#
# The pid record is server.py's, written with its helpers, so `make status`
# and `make stop` see a dev-ui server as they see `make start`'s; the
# refusals, the /ready wait and the kept logs are `make start`'s on Linux.
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
import uuid

# server.py's helpers, imported without leaving a __pycache__ in the tree.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import alive, control_lock, forget, identity, read_record  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VITE_GRACE = 5
POLL = 0.25
PR_SET_PDEATHSIG = 1
try:
    LIBC = ctypes.CDLL(None, use_errno=True)
except OSError:
    LIBC = None

# The signal that ended the session, and the server once it was asked to
# stop, which a second Ctrl+C then ends at once.
ended = None
stopping_server = None

# The lines `--server-api-key auto` and `--server-expose` print once into the
# server's log, repeated on the console as on Windows.
GENERATED = [("API key (generated for this run)", r"generated API key: (\S+)"),
             ("Public URL", r"public URL: (\S+)"),
             ("Public Playground", r"Playground: (\S+)")]


def say(*lines, error=False):
    try:
        print(*lines, sep="\n", file=sys.stderr if error else sys.stdout, flush=True)
    except OSError:
        # The terminal is gone (SIGHUP): the session still stops both.
        pass


def on_signal(signum, _frame):
    global ended
    if ended is None:
        ended = signum
        return
    # Only a second Ctrl+C: a terminal that closes sends SIGHUP twice (to the
    # job, and again from the shell), and that keeps the drain.
    if signum == signal.SIGINT and stopping_server is not None and stopping_server.poll() is None:
        stopping_server.send_signal(signal.SIGTERM)


def child_setup(own_group):
    def setup():
        if own_group:
            os.setpgid(0, 0)
        if LIBC is not None:
            LIBC.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM))
    return setup


class Session:
    """The make this script runs under: its parent, and its process group's leader."""

    def __init__(self):
        self.parent = os.getppid()
        group = os.getpgrp()
        self.leader = identity(group) if group != os.getpid() else None

    def gone(self):
        if os.getppid() != self.parent:
            return True
        return self.leader is not None and identity(self.leader["pid"]) != self.leader


def describe(code):
    if code >= 0:
        return f"code {code}"
    try:
        return f"signal {signal.Signals(-code).name}"
    except ValueError:
        return f"signal {-code}"


def listening(url):
    endpoint = urlsplit(url)
    try:
        socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=1).close()
    except OSError:
        return False
    return True


def show_log_tail(log, count=40):
    for path in [log, Path(str(log) + ".err")]:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except FileNotFoundError:
            continue
        if lines:
            say(f"--- {path} (last {count} lines)", *lines[-count:])


def show_generated(log):
    try:
        text = log.read_text(errors="replace")
    except FileNotFoundError:
        return
    lines = [f"{label}: {found[-1]}" for label, pattern in GENERATED if (found := re.findall(pattern, text))]
    if lines:
        say("", *lines, "")


def stop(process, grace, name, server=False):
    global stopping_server
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    if server:
        stopping_server = process
    try:
        process.wait(grace)
        say(f"{name} (pid {process.pid}) stopped")
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait()
    say(f"{name} (pid {process.pid}) killed")


def start_server(args, command):
    """The refusals, log rotation and pid record of server.py's start, for a child of this session."""
    with control_lock(args.pid):
        record = read_record(args.pid)
        if alive(record):
            raise ValueError(f"ignis-server is already running (pid {record['pid']}); make stop first")
        if listening(args.url):
            raise ValueError(f"{args.url} is already listening; refusing to start a second server")
        args.pid.unlink(missing_ok=True)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        stderr = Path(str(args.log) + ".err")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        for log in [args.log, stderr]:
            if log.exists():
                log.rename(Path(str(log) + "." + stamp))
        with args.log.open("w") as out, stderr.open("w") as err:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       preexec_fn=child_setup(own_group=True))
        record = identity(process.pid)
        if record is not None:
            temporary = Path(str(args.pid) + "." + uuid.uuid4().hex)
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(args.pid)
    say(f"started ignis-server pid {process.pid}: {' '.join(command)}")
    return process, record


def wait_ready(args, server, session):
    """'ready', 'ended' (a signal or make gone), 'exited' (log tail shown) or 'timeout'."""
    url = args.url.rstrip("/") + "/ready"
    say(f"log {args.log} ; waiting up to {args.timeout:g}s for {url}")
    # /ready is public even when the API requires a key, and is 503 during warmup.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    began = time.monotonic()
    while True:
        if ended is not None or session.gone():
            return "ended"
        if server.poll() is not None:
            say(f"error: ignis-server exited with {describe(server.returncode)} before it was ready", error=True)
            show_log_tail(args.log)
            return "exited"
        try:
            with opener.open(url, timeout=1) as response:
                if response.status == 200 and server.poll() is None:
                    say(f"ready: {args.url} (pid {server.pid}, {time.monotonic() - began:.1f}s)")
                    show_generated(args.log)
                    return "ready"
        except OSError:
            pass
        if time.monotonic() - began >= args.timeout:
            say(f"error: not ready after {args.timeout:g}s (pid {server.pid})", error=True)
            return "timeout"
        time.sleep(POLL)


def run(args, command):
    vite_js = REPO / "web" / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite_js.is_file():
        raise ValueError(f"{vite_js} not found (make web-install)")
    node = shutil.which("node")
    if not node:
        raise ValueError("node not found on PATH")
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, on_signal)
    session = Session()
    server, record = start_server(args, command)
    vite = None
    try:
        state = wait_ready(args, server, session)
        if state == "ended":
            return end_code()
        if state != "ready":
            return 1
        # --clearScreen false: Vite would otherwise wipe the generated API key
        # printed just above.
        vite = subprocess.Popen([node, str(vite_js), "--clearScreen", "false"], cwd=REPO / "web",
                                env=dict(os.environ, IGNIS_URL=args.url), preexec_fn=child_setup(own_group=False))
        say("", f"Playground (Vite, hot reload) -> {args.url} ; server log {args.log}",
            "Ctrl+C stops the server and Vite together", "")
        while True:
            if ended is not None or session.gone():
                return end_code()
            if vite.poll() is not None:
                say(f"vite exited ({describe(vite.returncode)}); stopping the server")
                return vite.returncode if vite.returncode >= 0 else 128 - vite.returncode
            if server.poll() is not None:
                say(f"ignis-server exited ({describe(server.returncode)}); stopping vite")
                show_log_tail(args.log)
                return 1
            time.sleep(POLL)
    finally:
        stop(vite, VITE_GRACE, "vite")
        stop(server, args.stop_timeout, "ignis-server", server=True)
        if record is not None:
            with control_lock(args.pid):
                forget(args.pid, record)


def end_code():
    """Say why the session ended (a signal, or make gone) and return the exit code for it."""
    if ended is None:
        say("make is gone: stopping vite and ignis-server")
        return 1
    name = "Ctrl+C" if ended == signal.SIGINT else signal.Signals(ended).name
    say("", f"{name}: stopping vite and ignis-server")
    return 128 + ended


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description="make run-ui / dev-ui on Linux: the server and Vite as one session")
    parser.add_argument("--pid", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--stop-timeout", type=float, default=20)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        parser.error("requires -- server-binary [arguments ...]")
    if args.timeout <= 0 or args.stop_timeout <= 0:
        parser.error("timeouts must be positive")
    try:
        return run(args, command)
    except (OSError, ValueError) as error:
        say(f"error: {error}", error=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
