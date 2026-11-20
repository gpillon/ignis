#!/usr/bin/env python3
"""The GPU lock on Linux (ADR 0006, GitHub #707), the counterpart of mk/windows/gpulock.ps1.

One lock per card, keyed by IGNIS_GPU_CARD ("0", the one card today, GitHub #387), a JSON file
in IGNIS_GPU_LOCK_DIR (default /tmp/ignis-gpu-lock, one place for the whole machine) that names
its holder: the pid, the process's start time and the boot it ran in, the worktree, the command
and the time the hold began. A record whose pid is gone, or now names another process (another
start time or another boot), is free, so a holder killed before it could release leaves nothing
standing. Every change of the record (take, hand over, release) is made under an flock beside
it, so two that find one stale record cannot both take it.

It is added beside the preflight marker and the process check of mk/linux/gpu.sh, not in place
of them: ignis_core::gpu_profile reads the marker, and a worktree on an older commit knows only
those two.

    gpulock.py acquire --pid PID --wait SECONDS --command TEXT   hold it for PID
    gpulock.py release --pid PID                                  only if PID holds it
    gpulock.py status [--except PID]                              exit 1 while another holds it

acquire waits up to SECONDS (0: refuses at once) and fails naming the holder; status prints the
holder, and exits 0 when the card is free or held by PID itself.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
POLL = 1.0


def lock_path():
    folder = os.environ.get("IGNIS_GPU_LOCK_DIR") or "/tmp/ignis-gpu-lock"
    return Path(folder) / f"gpu-{os.environ.get('IGNIS_GPU_CARD') or '0'}.lock"


def identity(pid):
    """server.py's: the pid, its start time and the boot, or None for a process gone or a zombie."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_time": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(record, holder):
    return all(record.get(key) == holder.get(key) for key in ("pid", "start_time", "boot_id"))


def live(record):
    """Whether the record's holder still runs: the same pid, start time and boot."""
    if not record:
        return False
    now = identity(record["pid"])
    return now is not None and same(record, now)


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except ValueError:
        # Written by a temporary file and a rename, so never half a record: an unreadable
        # one is no holder anyone can name.
        return None


def describe(record):
    return (f"pid {record['pid']} since {record.get('since', '?')}, worktree "
            f"{record.get('worktree', '?')}: {record.get('command', '?')}")


class Held(ValueError):
    def __init__(self, path, record):
        super().__init__(f"GPU lock: the card is held ({path}) by {describe(record)}")
        self.record = record


@contextmanager
def changing(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".mutex").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        yield


def write(path, record):
    temporary = Path(f"{path}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(record) + "\n")
    temporary.replace(path)


def acquire(pid, command, wait, say=print):
    """Hold the lock for `pid` (a live process), waiting up to `wait` seconds for it: the
    record written, or Held naming the holder once the wait is over."""
    path = lock_path()
    me = identity(pid)
    if me is None:
        raise ValueError(f"GPU lock: pid {pid} is not running")
    record = dict(me, worktree=str(REPO), command=command,
                  since=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    deadline = time.monotonic() + wait
    told = False
    while True:
        with changing(path):
            held = read(path)
            if live(held) and same(held, me):
                return held
            if not live(held):
                write(path, record)
                return record
        left = deadline - time.monotonic()
        if left <= 0:
            raise Held(path, held)
        if not told:
            say(f"GPU lock: waiting up to {wait:g}s for the card, held ({path}) by {describe(held)}",
                flush=True)
            told = True
        time.sleep(min(POLL, left))


def hand(holder, to, command=None):
    """Hand the lock `holder` holds to the process `to` (an identity), keeping the hold's
    worktree and start; False if `holder` no longer holds it."""
    path = lock_path()
    with changing(path):
        held = read(path)
        if not held or not same(held, holder):
            return False
        write(path, dict(held, **to, **({"command": command} if command else {})))
        return True


def release(holder):
    """Free the lock if `holder` (an identity, alive or not) holds it; nothing otherwise."""
    path = lock_path()
    if not path.exists():
        return False
    with changing(path):
        held = read(path)
        if not held or not same(held, holder):
            return False
        path.unlink()
        return True


def holder(path=None):
    """The live record, or None when the card is free."""
    record = read(path or lock_path())
    return record if live(record) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description="the GPU lock (one per card)")
    parser.add_argument("action", choices=["acquire", "release", "status"])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--wait", type=float, default=0)
    parser.add_argument("--command", default="")
    parser.add_argument("--except", dest="exempt", type=int)
    args = parser.parse_args(argv)
    if args.action != "status" and args.pid is None:
        parser.error(f"{args.action} requires --pid")
    try:
        if args.action == "acquire":
            acquire(args.pid, args.command, args.wait)
            return 0
        if args.action == "release":
            me = identity(args.pid)
            if me is not None:
                release(me)
            return 0
        path = lock_path()
        record = holder(path)
        if record is None:
            print(f"GPU lock: free ({path})")
            return 0
        print(f"GPU lock: held ({path}) by {describe(record)}")
        return 0 if args.exempt is not None and record["pid"] == args.exempt else 1
    except Held as error:
        print(error, file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"GPU lock: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
