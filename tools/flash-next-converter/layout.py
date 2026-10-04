"""The converter's output bytes: expert records, the per-layer index, the DONE protocol.

The contract is `docs/specs/flash-next/layout.md`; section numbers below refer to it.
Everything here is pure CPU and NumPy, so the packer's expectations are unit-tested
without a GPU or the checkpoint.
"""
import hashlib
import json
import os
import struct
from dataclasses import dataclass

import numpy as np

PAGE = 4096
# (in, out) of each expert projection as exllamav3 takes it (§3)
SHAPES = {"gu": (2560, 1280), "dn": (640, 2560)}
PROJ_CODE = {"gu": 0, "dn": 1}
PROJ_NAME = {0: "gu", 1: "dn"}
K2_SET = (4, 5, 6, 8)  # 2K for K in {2, 2.5, 3, 4}
INDEX_ENTRY = struct.Struct("<HBBIQ")  # expert, proj, k2, record bytes, offset


def _check_k2(k2):
    if k2 not in K2_SET:
        raise ValueError(f"k2 {k2} is not one of {K2_SET} (K in 2, 2.5, 3, 4)")


def trellis_bytes(proj, k2):
    _check_k2(k2)
    i, o = SHAPES[proj]
    return i * o * k2 // 16


def record_bytes(proj, k2):
    """Bytes of one record: trellis, suh, svh, zero padding to a 4096 multiple (§3)."""
    i, o = SHAPES[proj]
    data = trellis_bytes(proj, k2) + 2 * (i + o)
    return -(-data // PAGE) * PAGE


def rate(proj, k2):
    """Acceptance 2's bits per weight: K plus the fp16 channel scales, padding excluded (§5)."""
    i, o = SHAPES[proj]
    return k2 / 2 + 16 * (i + o) / (i * o)


def stored_rate(proj, k2):
    i, o = SHAPES[proj]
    return record_bytes(proj, k2) * 8 / (i * o)


@dataclass(frozen=True)
class IndexEntry:
    expert: int
    proj: str
    k2: int
    size: int
    offset: int


def record_payload(proj, k2, rec):
    """The record's bytes from exllamav3's tensors (NumPy arrays), checked against the class."""
    i, o = SHAPES[proj]
    trellis, suh, svh = rec["trellis"], rec["suh"], rec["svh"]
    want = (i // 16, o // 16, 8 * k2)
    if trellis.dtype != np.int16 or tuple(trellis.shape) != want:
        raise ValueError(f"{proj} k2={k2}: trellis {trellis.dtype} {tuple(trellis.shape)}, want int16 {want}")
    if suh.dtype != np.float16 or suh.shape != (i,) or svh.dtype != np.float16 or svh.shape != (o,):
        raise ValueError(f"{proj}: suh/svh must be fp16 ({i},) / ({o},)")
    body = np.ascontiguousarray(trellis).tobytes() + np.ascontiguousarray(suh).tobytes() \
        + np.ascontiguousarray(svh).tobytes()
    return body + bytes(record_bytes(proj, k2) - len(body))


def write_experts(directory, n_experts, record_of):
    """Writes experts.bin and experts.idx for experts 0..n-1, gate/up before down (§4).

    `record_of(expert, proj)` returns (k2, {"trellis", "suh", "svh"} as NumPy arrays).
    Returns the index entries and the sha256 of each record's bytes."""
    entries, digests, offset = [], [], 0
    with open(os.path.join(directory, "experts.bin"), "wb") as f:
        for e in range(n_experts):
            for proj in ("gu", "dn"):
                k2, rec = record_of(e, proj)
                payload = record_payload(proj, k2, rec)
                f.write(payload)
                entries.append(IndexEntry(e, proj, k2, len(payload), offset))
                digests.append(hashlib.sha256(payload).hexdigest())
                offset += len(payload)
    with open(os.path.join(directory, "experts.idx"), "wb") as f:
        for en in entries:
            f.write(INDEX_ENTRY.pack(en.expert, PROJ_CODE[en.proj], en.k2, en.size, en.offset))
    return entries, digests


def read_index(path):
    raw = open(path, "rb").read()
    if len(raw) % INDEX_ENTRY.size:
        raise ValueError(f"{path}: {len(raw)} bytes is not a whole number of index entries")
    out, offset = [], 0
    for n in range(len(raw) // INDEX_ENTRY.size):
        e, p, k2, size, off = INDEX_ENTRY.unpack_from(raw, n * INDEX_ENTRY.size)
        proj = PROJ_NAME[p]
        if size != record_bytes(proj, k2) or off != offset:
            raise ValueError(f"{path}: entry {n} (expert {e} {proj}) size {size} offset {off}, want "
                             f"{record_bytes(proj, k2)} at {offset}")
        out.append(IndexEntry(e, proj, k2, size, off))
        offset += size
    return out


def read_record(path, entry):
    """exllamav3's tensors of one record, as NumPy arrays (§3)."""
    i, o = SHAPES[entry.proj]
    with open(path, "rb") as f:
        f.seek(entry.offset)
        raw = f.read(entry.size)
    t = trellis_bytes(entry.proj, entry.k2)
    trellis = np.frombuffer(raw, dtype=np.int16, count=t // 2).reshape(i // 16, o // 16, 8 * entry.k2)
    suh = np.frombuffer(raw, dtype=np.float16, count=i, offset=t)
    svh = np.frombuffer(raw, dtype=np.float16, count=o, offset=t + 2 * i)
    return {"trellis": trellis, "suh": suh, "svh": svh}


# ---------------------------------------------------------------- the DONE protocol (§2)

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
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mark_done(directory, digests=None):
    """Writes DONE last: every other file of the directory with its size and sha256.
    `digests` (name -> sha256) skips re-hashing files whose digest is already known."""
    digests = digests or {}
    files = {}
    for name in sorted(os.listdir(directory)):
        p = os.path.join(directory, name)
        if name in ("DONE", "DONE.tmp") or not os.path.isfile(p):
            continue
        files[name] = {"bytes": os.path.getsize(p), "sha256": digests.get(name) or file_digest(p)}
    write_json_atomic(os.path.join(directory, "DONE"), {"files": files})


def is_done(directory):
    return os.path.isfile(os.path.join(directory, "DONE"))


# ---------------------------------------------------------------- non-expert encodings (§6.2)

# the projections the study measured in FP8 (run 6's NONEXPERT list)
STUDY_FP8 = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
             "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "self_attn.indexer.index_qk_proj", "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
             "mlp.shared_expert.down_proj")
ROUTER = "mlp.gate.weight"
FP8_MIN_ROWS = 16


def encoding_of(name, shape):
    """'expert', 'fp8' or 'bf16' for a checkpoint tensor (name without the model prefix's
    `layers.N.`): every 2-D linear weight with at least 16 rows and columns is FP8 except
    the router; the experts are trellis records; everything else stays bf16."""
    if name.startswith("mlp.experts."):
        return "expert"
    if name == ROUTER or name.startswith("ple.ple_embedding."):
        return "bf16"
    if name.endswith(".weight") and len(shape) == 2 and min(shape) >= FP8_MIN_ROWS and "norm" not in name:
        return "fp8"
    return "bf16"


def measured_by_study(name):
    return name[:-len(".weight")].endswith(STUDY_FP8) if name.endswith(".weight") else False
