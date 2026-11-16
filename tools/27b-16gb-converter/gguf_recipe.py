"""A recipe translated from a public GGUF file's per-tensor allocation (GitHub #634, spec #630): the 12 GB card
class's second candidate, `ig28b`, from the dynamic 2-bit quant of the 27B nearest the ceiling.

    header: the file's header read by HTTP ranges, never its weights: every tensor's name, dims and ggml type,
            with the file's repository, revision, size and LFS oid, as a table (JSON).
    recipe: the table translated to a named recipe (`recipes/<name>.json`) under a ceiling in bytes.

The rule (`fit`):

1. Each matrix's bits are its ggml type's bits per weight, block scales included: the bytes of a block times 8
   over its weights (`GGML_TYPES`, from llama.cpp's `ggml-common.h`).
2. Its K is the integer nearest its bits plus one shift for the whole body, held to 2 to 5. The head is at 4
   whatever its type: the load binds it there, so it is outside the order.
3. The shift is the largest at which the text scope's device weights, counted as the converter counts them, are
   at or under the ceiling. K moves only where a class of bits crosses a half, so the shift is one of those
   points, and every matrix with more bits than another has a K at least as high.
4. The class that crosses next (every matrix whose K the next point raises) does not fit whole: its matrices are
   promoted in rank order, while the next one fits. A class's matrices tie on bits, so the rank is the source's
   own allocation one grain coarser: the mean bits of the matrix's layer, weighted by weights, higher first;
   then the key.

A matrix of the record maps to one GGUF tensor (`GGUF_NAMES`): the source splits the body as exllamav3 does, so
no K is shared, and gate/up or q/k/v keep their own (the load binds each matrix at its own K).
"""
import argparse
import collections
import json
import math
import os
import re
import struct
import sys
import urllib.parse
import urllib.request
from fractions import Fraction

import recipe as recipe_mod

SCHEMA = "qwen3.8-27b-gguf-allocation-v1"
# ggml type id -> (name, bytes per block, weights per block): the `block_*` structs' sizes, which
# llama.cpp's ggml/src/ggml-common.h pins with static_asserts (tag v1.0.0, b1b770d90d94cad9e9ea5b993bdbc383a522fb28).
GGML_TYPES = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 2: ("Q4_0", 18, 32), 3: ("Q4_1", 20, 32), 6: ("Q5_0", 22, 32),
              7: ("Q5_1", 24, 32), 8: ("Q8_0", 34, 32), 10: ("Q2_K", 84, 256), 11: ("Q3_K", 110, 256),
              12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256), 16: ("IQ2_XXS", 66, 256),
              17: ("IQ2_XS", 74, 256), 18: ("IQ3_XXS", 98, 256), 19: ("IQ1_S", 50, 256), 20: ("IQ4_NL", 18, 32),
              21: ("IQ3_S", 110, 256), 22: ("IQ2_S", 82, 256), 23: ("IQ4_XS", 136, 256), 29: ("IQ1_M", 56, 256),
              30: ("BF16", 2, 1)}
# The 27B's matrices under GGUF's names (llama.cpp's `qwen35` architecture) -> exllamav3's.
GGUF_NAMES = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z",
              "ssm_out": "linear_attn.out_proj", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
              "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj",
              "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
GGUF_HEAD = "output.weight"
CHUNK = 1 << 20
# The calibration files a local conversion reads: their pins are the base recipe's.
CALIBRATION = ("cal_trace.safetensors", "quantization_config.json")
LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")
HF = "https://huggingface.co"
AGENT = {"User-Agent": "ignis-gguf-header-reader"}

Refused = recipe_mod.Refused


# ---- the header ---------------------------------------------------------------------------------------------------

class _Ranged:
    """`read(start, n)`'s bytes from the file's start, asked for in `chunk`s as the parser needs them."""

    def __init__(self, read, chunk):
        self.read, self.chunk, self.buf, self.pos = read, chunk, bytearray(), 0

    def take(self, n):
        while len(self.buf) < self.pos + n:
            got = self.read(len(self.buf), self.chunk)
            if not got:
                raise Refused(f"the file ends at {len(self.buf):,} bytes, inside its header")
            self.buf += got
        b = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return b

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("Q")).decode("utf-8", errors="replace")


# GGUF's value types: the scalars' struct formats; 8 is a string, 9 an array.
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _value(r, t):
    if t in _SCALARS:
        return r.unpack(_SCALARS[t])
    if t == 8:
        return r.string()
    if t == 9:
        et, count = r.unpack("I"), r.unpack("Q")
        if et in _SCALARS:
            r.take(struct.calcsize(_SCALARS[et]) * count)
        else:
            for _ in range(count):
                _value(r, et)
        return {"array": et, "count": count}
    raise Refused(f"GGUF value type {t} is not one of the format's")


def read_header(read, chunk=CHUNK):
    """A GGUF file's header through `read(start, n)` (bytes from `start`): its metadata (an array as its element
    type and count), every tensor's name, dims, ggml type and data offset, the header's length and the bytes
    asked for, which stop within one `chunk` of the header's end."""
    r = _Ranged(read, chunk)
    if r.take(4) != b"GGUF":
        raise Refused("not a GGUF file: its first bytes are not GGUF's magic")
    version, n_tensors, n_kv = r.unpack("I"), r.unpack("Q"), r.unpack("Q")
    metadata = {}
    for _ in range(n_kv):
        key = r.string()
        metadata[key] = _value(r, r.unpack("I"))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.unpack("Q") for _ in range(r.unpack("I"))]
        tensors.append({"name": name, "dims": dims, "type": r.unpack("I"), "offset": r.unpack("Q")})
    return {"version": version, "metadata": metadata, "tensors": tensors, "header_bytes": r.pos,
            "fetched_bytes": len(r.buf)}


def http_reader(url):
    """`read(start, n)` by HTTP range requests on `url`, counting them in `.requests`."""
    def read(start, n):
        req = urllib.request.Request(url, headers=dict(AGENT, Range=f"bytes={start}-{start + n - 1}"))
        with urllib.request.urlopen(req) as resp:
            if resp.status != 206:
                raise Refused(f"{url} answered a range request with HTTP {resp.status}")
            read.requests += 1
            return resp.read()
    read.requests = 0
    return read


def hub_file(repo, revision, name):
    """The file's size and LFS oid at `revision` (the Hub's paths-info), and its last commit."""
    body = urllib.parse.urlencode({"paths": name, "expand": "true"}).encode()
    req = urllib.request.Request(f"{HF}/api/models/{repo}/paths-info/{revision}", data=body, headers=AGENT)
    with urllib.request.urlopen(req) as resp:
        info = json.load(resp)
    if len(info) != 1 or "lfs" not in info[0]:
        raise Refused(f"{repo} at {revision} has no LFS file {name}")
    return {"bytes": info[0]["size"], "lfs_oid": info[0]["lfs"]["oid"],
            "last_commit": (info[0].get("lastCommit") or {}).get("id")}


# ---- the translation ----------------------------------------------------------------------------------------------

def bits_of(t):
    """A ggml type's bits per weight, its block scales included."""
    name, size, weights = GGML_TYPES[t]
    return name, Fraction(size * 8, weights)


def allocation(tensors, shapes):
    """{key: {"type", "bits"}} for every matrix of `shapes` ({key: [out, in]}: the record's, the body's and the
    head `lm_head`) from the GGUF tensor infos `tensors`; GGUF's dims are [in, out]. Tensors the record does not
    have (norms, the GDN small set, the MTP layer, the token embedding) are not read. Refused names a matrix
    with no tensor, other dims, or a type outside the table."""
    by_name = {t["name"]: t for t in tensors}
    out, problems = {}, []
    for key, shape in sorted(shapes.items()):
        if key == recipe_mod.HEAD:
            name = GGUF_HEAD
        else:
            m = LAYER.match(key)
            gguf = {v: k for k, v in GGUF_NAMES.items()}.get(key[m.end():]) if m else None
            name = f"blk.{m[1]}.{gguf}.weight" if gguf else None
        t = by_name.get(name)
        if t is None:
            problems.append(f"{key}: no GGUF tensor ({name or 'no GGUF name'})")
        elif t["dims"] != list(reversed(shape)):
            problems.append(f"{key}: GGUF dims {t['dims']} ([in, out]), the record's shape is {shape} ([out, in])")
        elif t["type"] not in GGML_TYPES:
            problems.append(f"{key}: ggml type {t['type']}, not one of the table's (GGML_TYPES)")
        else:
            name, bits = bits_of(t["type"])
            out[key] = {"type": name, "bits": float(bits)}
    if problems:
        raise Refused("the GGUF header does not hold the record's matrices:\n  " + "\n  ".join(problems))
    return out


def _k(bits, delta):
    lo, hi = recipe_mod.K_RANGE
    return min(hi, max(lo, math.floor(bits + delta + Fraction(1, 2))))


def _numel(shape):
    return shape[0] * shape[1]


def fit(alloc, shapes, copied, ceiling):
    """The rule (the module's): `alloc` (`allocation`'s) to {key: K} under `ceiling`, the text scope counted as
    `recipe.text_scope_bytes` counts it with the objects `copied` ({name: bytes}). Returns the K, the shift
    (`delta`, None when every matrix is at K2), the next point (`next_delta`), the matrices of its class
    promoted and those left in rank order, and the bytes before and after the promotion."""
    body = sorted(k for k in shapes if k != recipe_mod.HEAD)
    bits = {k: Fraction(alloc[k]["bits"]) for k in body}

    def state(delta):
        ks = {k: (recipe_mod.K_RANGE[0] if delta is None else _k(bits[k], delta)) for k in body}
        ks[recipe_mod.HEAD] = recipe_mod.HEAD_BITS
        return ks

    def total(ks):
        return recipe_mod.text_scope_bytes({k: {"K": K, "shape": shapes[k]} for k, K in ks.items()}, copied)

    floor = state(None)
    if total(floor) > ceiling:
        raise Refused(f"at K2 the text scope is {total(floor):,} bytes, above the ceiling of {ceiling:,}")
    lo, hi = recipe_mod.K_RANGE
    points = sorted({Fraction(k) + Fraction(1, 2) - b for b in set(bits.values()) for k in range(lo, hi)})
    delta, ks, nxt = None, floor, None
    for p in points:
        s = state(p)
        if s == ks:
            continue
        if total(s) > ceiling:
            nxt = p
            break
        delta, ks = p, s
    class_bytes = total(ks)
    promoted, left = [], []
    if nxt is not None:
        raised = state(nxt)
        layer = collections.defaultdict(lambda: [0, 0])
        for k in body:
            i = int(LAYER.match(k)[1])
            layer[i][0] += bits[k] * _numel(shapes[k])
            layer[i][1] += _numel(shapes[k])
        mean = {i: s / n for i, (s, n) in layer.items()}
        ranked = sorted((k for k in body if raised[k] != ks[k]), key=lambda k: (-mean[int(LAYER.match(k)[1])], k))
        for i, k in enumerate(ranked):
            trial = dict(ks, **{k: raised[k]})
            if total(trial) > ceiling:
                left = ranked[i:]
                break
            ks = trial
            promoted.append(k)
    return {"K": dict(sorted(ks.items())), "delta": None if delta is None else str(delta),
            "next_delta": None if nxt is None else str(nxt), "promoted": promoted, "left": left,
            "class_bytes": class_bytes, "text_scope_bytes": total(ks)}


def exllamav3_bits(ks, shapes):
    """The `bits` exllamav3 1.5.3 writes in the quant's configuration from a recipe: the body's average K weighted
    by each matrix's weights, to two decimals (`create_q_strategy_from_recipe`, `round(final_bpw, 2)`)."""
    body = [k for k in ks if k != recipe_mod.HEAD]
    return round(float(sum(_numel(shapes[k]) * ks[k] for k in body)) / sum(_numel(shapes[k]) for k in body), 2)


def about(source, f):
    return (f"Candidate B of the 12 GB card class (#634): the per-tensor allocation of {source['repo']} "
            f"{source['file']} at revision {source['revision']} (LFS oid {source['lfs_oid']}), read from its GGUF "
            f"header, translated by gguf_recipe.py: each matrix's K the integer nearest its ggml type's bits per weight "
            f"(block scales included) plus one shift for the whole body, held to 2 to 5, the head at 4; the shift the "
            f"largest under the ceiling ({f['delta']}), then the next class to cross promoted by its layer's mean "
            f"bits, then its key, while it fits.")


def build(base, f, shapes, source, weights_id, ceiling):
    """The recipe: the base's configuration, job and calibration pins, the fit's K, exllamav3's bits."""
    return {"schema": recipe_mod.SCHEMA, "about": about(source, f), "weights_id": weights_id, "ceiling_bytes": ceiling,
            "quantization_config": dict(base["quantization_config"], bits=exllamav3_bits(f["K"], shapes)),
            "job": base["job"], "files": {n: base["files"][n] for n in CALIBRATION},
            "tensors": {k: K for k, K in f["K"].items() if k != recipe_mod.HEAD}}


def histogram(ks):
    return " ".join(f"K{K} x{n}" for K, n in sorted(collections.Counter(ks.values()).items()))


# ---- the commands -------------------------------------------------------------------------------------------------

def cmd_header(a, log):
    source = dict(repo=a.repo, file=a.file, revision=a.revision, **hub_file(a.repo, a.revision, a.file))
    read = http_reader(f"{HF}/{a.repo}/resolve/{a.revision}/{urllib.parse.quote(a.file)}")
    h = read_header(read, a.chunk)
    table = dict(schema=SCHEMA, source=source, range_requests=read.requests, **h)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(table, fh, indent=1)
        fh.write("\n")
    log(f"{a.repo} {a.file} at {a.revision}: GGUF v{h['version']}, {len(h['tensors'])} tensors, header "
        f"{h['header_bytes']:,} bytes, {h['fetched_bytes']:,} read in {read.requests} ranges of {source['bytes']:,}")
    return 0


def cmd_recipe(a, log):
    path = os.path.join(a.recipes, f"{a.name}.json")
    if os.path.exists(path):
        raise Refused(f"{path} exists: a recipe is not overwritten")
    with open(a.table, encoding="utf-8") as fh:
        table = json.load(fh)
    with open(a.record, encoding="utf-8") as fh:
        record = json.load(fh)
    shapes = {m["source"]: m["shape"] for m in record["trellis"]["matrices"].values()}
    copied = {n: c["bytes"] for n, c in record["copied"]["objects"].items()}
    f = fit(allocation(table["tensors"], shapes), shapes, copied, a.ceiling)
    data = build(recipe_mod.load(a.base), f, shapes, table["source"], a.weights_id, a.ceiling)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")
    r = recipe_mod.load(a.name, a.recipes)
    m = {k: {"K": K, "shape": shapes[k]} for k, K in recipe_mod.ks(r).items()}
    log(f"recipe {a.name}: {histogram(recipe_mod.ks(r))}, {recipe_mod.bpw(m)} bpw (exllamav3's bits "
        f"{r['quantization_config']['bits']}), text scope {f['text_scope_bytes']:,} bytes of {a.ceiling:,}; shift "
        f"{f['delta']}, {f['class_bytes']:,} bytes before the next class (shift {f['next_delta']}), "
        f"{len(f['promoted'])} of it promoted, {len(f['left'])} left")
    return 0


def main(argv=None, log=print):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    h = sub.add_parser("header", help="read a GGUF file's header by HTTP ranges into a table")
    h.add_argument("--repo", required=True)
    h.add_argument("--revision", required=True)
    h.add_argument("--file", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--chunk", type=int, default=CHUNK)
    r = sub.add_parser("recipe", help="translate a table to a named recipe under a ceiling")
    r.add_argument("--table", required=True)
    r.add_argument("--record", required=True, help="a 27B trellis conversion record: the shapes and the copied objects")
    r.add_argument("--name", required=True)
    r.add_argument("--weights-id", required=True)
    r.add_argument("--ceiling", type=int, required=True)
    r.add_argument("--base", default=recipe_mod.DEFAULT, help="the recipe whose configuration, job and pins it takes")
    r.add_argument("--recipes", default=recipe_mod.RECIPES)
    a = p.parse_args(argv)
    try:
        return (cmd_header if a.command == "header" else cmd_recipe)(a, log)
    except Refused as e:
        log(f"REFUSED: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
