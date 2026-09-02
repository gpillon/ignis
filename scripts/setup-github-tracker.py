#!/usr/bin/env python3
"""One-shot: set up the GitHub issue tracker for ignis (gpillon/ignis).

Creates the five triage labels and the three kernel-port issues
(#01 closed/resolved, #02 and #03 open with the ready-for-agent label).

The token is read at runtime from the git remote URL (your local .git/config)
and never leaves this machine. Safe to re-run: existing labels/issues are
skipped (matched by name/title).

Run:  python scripts/setup-github-tracker.py
"""
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "gpillon/ignis"
API = f"https://api.github.com/repos/{REPO}"

LABELS = ["needs-triage", "needs-info", "ready-for-agent", "ready-for-human", "wontfix"]

# (title, body source file, close-after-create, labels)
ISSUES = [
    (
        "kernel-port 01: build skeleton (CMake+ninja+nvcc 120a, C ABI, Rust FFI)",
        ".scratch/kernel-port/issues/01-kernel-build-skeleton.md",
        True,
        [],
    ),
    (
        "kernel-port 02: artifact reader port (src/artifact/* -> ignis-artifact)",
        ".scratch/kernel-port/issues/02-artifact-reader.md",
        False,
        ["ready-for-agent"],
    ),
    (
        "kernel-port 03: C ABI surface - decode step (GEMM + attention)",
        ".scratch/kernel-port/issues/03-c-abi-surface.md",
        False,
        ["ready-for-agent"],
    ),
]


def token() -> str:
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True, cwd=ROOT
    ).stdout
    m = re.search(r"://[^@/]+:([^@]+)@", url)
    if not m:
        sys.exit("token not found in the origin URL (expected https://user:TOKEN@github.com/...)")
    return m.group(1)


TOK = token()


def api(method: str, path: str, payload=None, ignore_422: bool = False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path,
        data=data,
        headers={
            "Authorization": f"Bearer {TOK}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ignis-tracker-setup",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r) if r.status != 204 else None
    except urllib.error.HTTPError as e:
        if ignore_422 and e.code == 422:
            return None  # already exists
        raise


def setup_labels() -> None:
    existing = {l["name"] for l in api("GET", "/labels?per_page=100")}
    for name in LABELS:
        if name in existing:
            print(f"label exists:  {name}")
        else:
            api("POST", "/labels", {"name": name}, ignore_422=True)
            print(f"label created: {name}")


def setup_issues() -> None:
    titles = [i["title"] for i in api("GET", "/issues?state=all&per_page=100")]
    for title, body_file, close, labels in ISSUES:
        if title in titles:
            print(f"issue exists:  {title}")
            continue
        body = (ROOT / body_file).read_text()
        issue = api("POST", "/issues", {"title": title, "body": body, "labels": labels})
        print(f"issue created: #{issue['number']} {title}")
        if close:
            api("PATCH", f"/issues/{issue['number']}", {"state": "closed"})
            print(f"issue closed:  #{issue['number']} (resolved locally, see .scratch/)")


if __name__ == "__main__":
    setup_labels()
    setup_issues()
    print(f"done - tracker: https://github.com/{REPO}/issues")