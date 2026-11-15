"""Reader for mtp_replay's per-round records (crates/runtime/examples/mtp_replay.rs, GitHub #626).

The manifest and the token streams are the DFlash2 replay's (tools/dflash2-replay); the records
are the MTP head's: per round, the drafts the head's chain proposed, the head's probability of
each, its 16 leading candidates, the drafts the recording accepted and the committed run.
"""
import json
import os

import numpy as np

K = 7  # the record's draft positions: the widest window; `window` says how many a round read
C = K + 1
TOPK = 16

RECORD = np.dtype([
    ("job", "<u4"), ("turn", "<u4"), ("anchor", "<u4"), ("window", "<u4"), ("extent", "<u4"),
    ("accepted", "<u4"), ("committed", "<u4"),
    ("run", "<i4", (C,)),  # the committed run, the anchor first, -1 past it
    ("drafts", "<i4", (K,)),  # -1 past the window
    ("probability", "<f4", (K,)),  # the head's, softmax at T = 1 over the whole vocabulary; NaN past
    ("cand_ids", "<i4", (K, TOPK)),  # descending, the draft first
    ("cand_probs", "<f4", (K, TOPK)),
    ("agreed", "<u4"),
])
assert RECORD.itemsize == 1016, RECORD.itemsize


def load(out_dir, manifest):
    """All rounds of every finished job (one with its `.done`)."""
    parts = []
    for job in manifest["jobs"]:
        if not os.path.exists(os.path.join(out_dir, job["job"] + ".done")):
            continue
        parts.append(np.fromfile(os.path.join(out_dir, job["job"] + ".rounds"), dtype=RECORD))
    return np.concatenate(parts) if parts else np.zeros(0, RECORD)


def read_manifest(path):
    m = json.load(open(path))
    m["_path"] = path
    return m


def replay(out_dir):
    """The directory's `replay.json`: the drafter and the window its records were replayed at."""
    return json.load(open(os.path.join(out_dir, "replay.json")))


def accepted_at(rows):
    """Per round and draft position s (0-based), whether the recording accepted draft s + 1:
    a bool [rounds, K], False past the round's window."""
    return np.arange(K)[None, :] < rows["accepted"][:, None]
