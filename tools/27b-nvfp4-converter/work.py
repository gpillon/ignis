"""The work tree's DONE protocol (docs/formats/flash-next-layout.md §2), which the packer checks.

Copied from tools/image-converter/work.py (itself tools/flash-next-converter/layout.py's `file_digest`,
`write_json_atomic`, `_fsync`, `mark_done`), unchanged: a unit directory is complete only when
its DONE, written last, lists every other file of the directory with its size and sha256.
"""
import hashlib
import json
import os


def file_digest(path, block=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json_atomic(path, obj):
    """Writes JSON through a flushed temporary file and a rename."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync(path):
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mark_done(directory, digests=None):
    """Writes DONE last: every other file of the directory with its size and sha256, each file
    flushed to disk first. `digests` (name -> sha256) skips re-hashing known files."""
    digests = digests or {}
    files = {}
    for name in sorted(os.listdir(directory)):
        p = os.path.join(directory, name)
        if name in ("DONE", "DONE.tmp") or not os.path.isfile(p):
            continue
        _fsync(p)
        files[name] = {"bytes": os.path.getsize(p), "sha256": digests.get(name) or file_digest(p)}
    write_json_atomic(os.path.join(directory, "DONE"), {"files": files})
