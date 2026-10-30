"""An override directory (GitHub #420, spec #386): EXL3 records that replace the quant's records of
the same name, so a variant made from another BF16 (the abliterated one) is built like the quant.

The directory is what `replay.py run` writes: an EXL3 checkpoint in the quant's layout (index, shards,
`quantization_config.json`) holding only the matrices it replaces, and `replay.json`, its
provenance. The provenance gives the override's source repository and revision (the BF16 it was made
from), the quant revision it was made against and each record's sha256; the override's files are
pinned by their bytes and sha256 when the conversion runs. A record swaps like for like: its name
must be one of the quant's text matrices, with the same K and the same shape. Its reconstruction
error is measured against `reference_root`, the BF16 it was made from: the partial shards `replay.py
fetch` wrote (only the changed tensors) and its `fetch.json`, whose per-tensor sha256 pin each tensor.
"""
import hashlib
import json
import os
import struct

import exl3
import work

REPLAY_RECORD = "replay.json"
FETCH_RECORD = "fetch.json"
INDEX = "model.safetensors.index.json"


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tensor_sha256(path, name):
    """The sha256 of tensor `name`'s bytes in the safetensors file `path`."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        start, end = json.loads(f.read(n))[name]["data_offsets"]
        f.seek(8 + n + start)
        h, left = hashlib.sha256(), end - start
        while left:
            chunk = f.read(min(left, 1 << 24))
            if not chunk:
                raise ValueError(f"{path} ends inside {name}")
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def provenance(root):
    """The replay record of `root`, checked for what the conversion reads from it; ValueError names
    what is missing."""
    path = os.path.join(root, REPLAY_RECORD)
    if not os.path.isfile(path):
        raise ValueError(f"{root} has no {REPLAY_RECORD}: the override's source repository and revision "
                         f"cannot be pinned (`replay.py run` writes it)")
    record = _json(path)
    missing = []
    if record.get("status") != "complete":
        missing.append(f"status complete (it is {record.get('status')!r})")
    sources = record.get("sources") or {}
    for part, fields in (("quant", ("revision",)), ("abliterated", ("repo", "revision", "tensors"))):
        for field in fields:
            if not (sources.get(part) or {}).get(field):
                missing.append(f"sources.{part}.{field}")
    if not record.get("matrices"):
        missing.append("matrices")
    elif not all((m or {}).get("record_sha256") for m in record["matrices"].values()):
        missing.append("matrices.*.record_sha256")
    if missing:
        raise ValueError(f"{path} lacks {', '.join(missing)}")
    return record


class Override:
    """`root`: the override's EXL3 directory, with its `replay.json`; `reference_root`: the partial
    BF16 checkpoint it was made from (`fetch.json` and shards), needed to measure its errors."""

    def __init__(self, root, reference_root=None):
        self.root, self.reference_root = str(root), reference_root
        self.provenance = provenance(self.root)
        abliterated = self.provenance["sources"]["abliterated"]
        self.repo, self.revision = abliterated["repo"], abliterated["revision"]
        self.quant_revision = self.provenance["sources"]["quant"]["revision"]
        self.quant = exl3.Quant(self.root)
        self.matrices = {key: self.quant.matrix(key) for key in self.quant.keys()}

    def problems(self, base):
        """What keeps this override from swapping into the quant's matrices `base` (key -> Matrix):
        one line per record, naming it and what differs."""
        out = []
        if not self.matrices:
            out.append("the override holds no EXL3 record")
        stated = self.provenance["matrices"]
        for key in sorted(set(self.matrices) - set(stated)):
            out.append(f"{key}: {REPLAY_RECORD} does not name that record")
        for key in sorted(set(stated) - set(self.matrices)):
            out.append(f"{key}: {REPLAY_RECORD} names it, the override holds no such record")
        for key, m in sorted(self.matrices.items()):
            b = base.get(key)
            if b is None:
                out.append(f"{key}: the quant has no record of that name")
            elif m.k2 != b.k2:
                out.append(f"{key}: K {m.k2 / 2:g}, the quant's is {b.k2 / 2:g}")
            elif m.shape != b.shape:
                out.append(f"{key}: shape {m.shape}, the quant's is {b.shape}")
            elif key in stated and hashlib.sha256(self.record(key)).hexdigest() != stated[key]["record_sha256"]:
                out.append(f"{key}: the record's sha256 is not {REPLAY_RECORD}'s")
        return out

    def record(self, key):
        return self.quant.record(self.matrices[key])

    def pin_reference(self):
        """The reference BF16 pinned for the conversion record: repository and revision from its
        `fetch.json`, which must be the replay's; each override matrix's `.weight` read from its
        shard and hashed against `fetch.json` and against `replay.json`. ValueError names what
        differs."""
        if self.reference_root is None:
            raise ValueError("the override's reference BF16 (the partial shards `replay.py fetch` wrote) "
                             "is not given")
        path = os.path.join(self.reference_root, FETCH_RECORD)
        if not os.path.isfile(path):
            raise ValueError(f"{self.reference_root} has no {FETCH_RECORD}: its tensors cannot be pinned "
                             f"(`replay.py fetch` writes it)")
        fetched = _json(path)
        if (fetched.get("repo"), fetched.get("revision")) != (self.repo, self.revision):
            raise ValueError(f"{path} is {fetched.get('repo')}@{fetched.get('revision')}, the override was "
                             f"made from {self.repo}@{self.revision}")
        index_path = os.path.join(self.reference_root, INDEX)
        weight_map = _json(index_path)["weight_map"]
        stated = self.provenance["sources"]["abliterated"]["tensors"]
        tensors, shards, problems = {}, {}, []
        for key in sorted(self.matrices):
            name = f"{key}.weight"
            t = (fetched.get("tensors") or {}).get(name)
            if t is None or name not in weight_map:
                problems.append(f"{name}: not in the reference")
                continue
            if weight_map[name] != t["shard"]:
                problems.append(f"{name}: the index has it in {weight_map[name]}, {FETCH_RECORD} in {t['shard']}")
                continue
            if stated.get(name) != t["sha256"]:
                problems.append(f"{name}: {FETCH_RECORD}'s sha256 is not {REPLAY_RECORD}'s")
                continue
            shard = os.path.join(self.reference_root, t["shard"])
            if not os.path.isfile(shard):
                problems.append(f"{name}: {t['shard']} is missing")
                continue
            digest = tensor_sha256(shard, name)
            if digest != t["sha256"]:
                problems.append(f"{name}: hashes to {digest}, {FETCH_RECORD} has {t['sha256']}")
                continue
            tensors[name] = {"shard": t["shard"], "bytes": t["bytes"], "sha256": digest}
            shards[t["shard"]] = {"lfs_oid": fetched["shards"][t["shard"]]["lfs_oid"]}
        if problems:
            raise ValueError("the override's reference BF16 is not the one it was made from:\n  "
                             + "\n  ".join(problems))
        return {"repo": self.repo, "revision": self.revision, "index_sha256": fetched.get("index_sha256"),
                "files": {FETCH_RECORD: {"bytes": os.path.getsize(path), "sha256": work.file_digest(path)},
                          INDEX: {"bytes": os.path.getsize(index_path), "sha256": work.file_digest(index_path)}},
                "shards": dict(sorted(shards.items())), "tensors": tensors}
