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

# The file names the replay writes and the override and the overlay read.
REPLAY_RECORD = "replay.json"
FETCH_RECORD = "fetch.json"
INDEX = "model.safetensors.index.json"


def read_json(path):
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
    record = read_json(path)
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


def mismatch(key, record, other, whose):
    """The line naming how the EXL3 record `key` (its `k2` and `shape`) is not like `other`, `whose`
    it stands for: its K first, then its shape; None when it is like for like."""
    if record.k2 != other.k2:
        return f"{key}: K {record.k2 / 2:g}, {whose} is {other.k2 / 2:g}"
    if record.shape != other.shape:
        return f"{key}: shape {record.shape}, {whose} is {other.shape}"
    return None


class Fetched:
    """A directory `replay.py fetch` wrote: partial BF16 shards, their index (`weight_map`) and
    `fetch.json` (`record`), whose `tensors` give each tensor's shard and sha256. ValueError when it
    has no `fetch.json`. Both of its readers, `overlay.Reference` and `Override.pin_reference`, check
    a tensor with `misplaced` and `digest`."""

    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, FETCH_RECORD)
        if not os.path.isfile(self.path):
            raise ValueError(f"{root} has no {FETCH_RECORD}: its tensors cannot be pinned "
                             f"(`replay.py fetch` writes it)")
        self.record = read_json(self.path)
        self.tensors = self.record.get("tensors") or {}
        self.index_path = os.path.join(root, INDEX)
        self.weight_map = read_json(self.index_path)["weight_map"]

    def misplaced(self, name):
        """The line naming both shards when the index and `fetch.json` put tensor `name` in
        different ones, else None."""
        mapped, stated = self.weight_map[name], self.tensors[name]["shard"]
        if mapped != stated:
            return f"{name}: the index has it in {mapped}, {FETCH_RECORD} in {stated}"
        return None

    def digest(self, name):
        """(the sha256 of tensor `name`'s bytes in its shard, the line naming it when that is not
        `fetch.json`'s, else None)."""
        digest = tensor_sha256(os.path.join(self.root, self.weight_map[name]), name)
        stated = self.tensors[name]["sha256"]
        return digest, None if digest == stated else f"{name}: hashes to {digest}, {FETCH_RECORD} has {stated}"


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
            unlike = None if b is None else mismatch(key, m, b, "the quant's")
            if b is None:
                out.append(f"{key}: the quant has no record of that name")
            elif unlike:
                out.append(unlike)
            elif key in stated and hashlib.sha256(self.record(key)).hexdigest() != stated[key]["record_sha256"]:
                out.append(f"{key}: the record's sha256 is not {REPLAY_RECORD}'s")
        return out

    def replaces(self, key):
        """Whether the override holds a record for the quant's matrix `key`, which it replaces."""
        return key in self.matrices

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
        fetched = Fetched(self.reference_root)
        record, path, index_path = fetched.record, fetched.path, fetched.index_path
        if (record.get("repo"), record.get("revision")) != (self.repo, self.revision):
            raise ValueError(f"{path} is {record.get('repo')}@{record.get('revision')}, the override was "
                             f"made from {self.repo}@{self.revision}")
        stated = self.provenance["sources"]["abliterated"]["tensors"]
        tensors, shards, problems = {}, {}, []
        for key in sorted(self.matrices):
            name = f"{key}.weight"
            t = fetched.tensors.get(name)
            if t is None or name not in fetched.weight_map:
                problems.append(f"{name}: not in the reference")
                continue
            misplaced = fetched.misplaced(name)
            if misplaced:
                problems.append(misplaced)
                continue
            if stated.get(name) != t["sha256"]:
                problems.append(f"{name}: {FETCH_RECORD}'s sha256 is not {REPLAY_RECORD}'s")
                continue
            if not os.path.isfile(os.path.join(self.reference_root, t["shard"])):
                problems.append(f"{name}: {t['shard']} is missing")
                continue
            digest, tampered = fetched.digest(name)
            if tampered:
                problems.append(tampered)
                continue
            tensors[name] = {"shard": t["shard"], "bytes": t["bytes"], "sha256": digest}
            shards[t["shard"]] = {"lfs_oid": record["shards"][t["shard"]]["lfs_oid"]}
        if problems:
            raise ValueError("the override's reference BF16 is not the one it was made from:\n  "
                             + "\n  ".join(problems))
        return {"repo": self.repo, "revision": self.revision, "index_sha256": record.get("index_sha256"),
                "files": {FETCH_RECORD: {"bytes": os.path.getsize(path), "sha256": work.file_digest(path)},
                          INDEX: {"bytes": os.path.getsize(index_path), "sha256": work.file_digest(index_path)}},
                "shards": dict(sorted(shards.items())), "tensors": tensors}
