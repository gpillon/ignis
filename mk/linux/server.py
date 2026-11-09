#!/usr/bin/env python3
# Vendored from the RTX 5080 fork, github.com/xzzqqq/ignis-5080 at 9259978846 (GitHub #493);
# patched: the readiness loop retries on any OSError (a reset or a slow answer mid-load), not only URLError.
"""Linux daemon hooks for make; only the recorded process is stopped by default."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_time": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_record(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def alive(record):
    return bool(record and identity(record["pid"]) == record)


@contextmanager
def control_lock(pid_file):
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(pid_file) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def stop_process(record, timeout):
    if not alive(record):
        return
    # A daemon starts its own session. Do not signal a reused PID or another group.
    pid = record["pid"]
    try:
        group = os.getpgid(pid)
        if group == pid:
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while alive(record) and time.monotonic() < deadline:
        time.sleep(0.05)
    if alive(record):
        raise ValueError(f"server pid {pid} did not stop within {timeout:g}s; PID record retained")


def forget(pid_file, record):
    if read_record(pid_file) == record:
        pid_file.unlink(missing_ok=True)


def other_servers():
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid == os.getuid() and (path / "comm").read_text().strip() == "ignis-server":
                record = identity(int(path.name))
                if record:
                    yield record
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def stop(args):
    with control_lock(args.pid):
        record = read_record(args.pid)
        targets = [record] if alive(record) else []
        if args.force:
            targets.extend(r for r in other_servers() if r not in targets)
        for target in targets:
            stop_process(target, args.stop_timeout)
            print(f"stopped server pid {target['pid']}")
        forget(args.pid, record)
        if not targets:
            print("process: not running (no managed server)")


def start(args, command):
    if not command:
        raise ValueError("start requires -- server-binary [arguments ...]")
    process = None
    record = None
    try:
        with control_lock(args.pid):
            if alive(read_record(args.pid)):
                raise ValueError("managed server is already running; use make stop or redeploy")
            endpoint = urlsplit(args.url)
            try:
                connection = socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=1)
            except OSError:
                pass
            else:
                connection.close()
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
                                           start_new_session=True)
            record = identity(process.pid)
            if record is None:
                raise ValueError(f"server exited {process.poll()} before its PID could be recorded; see {stderr}")
            temporary = Path(str(args.pid) + "." + uuid.uuid4().hex)
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(args.pid)
        # Do not hold the control lock while waiting: status and stop remain usable.
        deadline = time.monotonic() + args.timeout
        # /ready is public even when the API requires a key, and is 503 during warmup.
        url = args.url.rstrip("/") + "/ready"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ValueError(f"server exited {process.returncode}; see {args.log} and {stderr}")
            try:
                with opener.open(url, timeout=min(1, max(0.01, deadline - time.monotonic()))) as response:
                    if response.status == 200 and process.poll() is None:
                        print(f"server pid {process.pid} ready at {args.url}; log {args.log}")
                        return
            except OSError:
                pass
            time.sleep(0.05)
        raise ValueError(f"server was not ready within {args.timeout:g}s; stopped; see {args.log} and {stderr}")
    except BaseException:
        if record is not None:
            with control_lock(args.pid):
                stop_process(record, args.stop_timeout)
                forget(args.pid, record)
        if process is not None:
            process.wait(timeout=args.stop_timeout)
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--pid", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--stop-timeout", type=float, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv[:split])
    if args.timeout <= 0 or args.stop_timeout <= 0:
        parser.error("timeouts must be positive")
    try:
        if args.action == "start":
            start(args, argv[split + 1:])
        elif args.action == "stop":
            stop(args)
        else:
            record = read_record(args.pid)
            print(f"process: running pid {record['pid']} (managed)" if alive(record) else "process: not running (no managed server)")
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"server: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
