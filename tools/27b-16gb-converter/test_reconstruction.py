"""`reconstruction.rel_errors`, the conversion's GPU step, run on the CPU: torch, exllamav3 and
safetensors are stubs, so nothing touches a GPU. The stub `safe_open` reads each tensor's shape from
the real file's header and fills the tensor with its source's value (the quant's records 1, the base
BF16 1, the override's records 2, its reference 4); the stub `LinearEXL3` reconstructs a weight of
its trellis's value. So a quant matrix measured against the base BF16 has error 0 and an override
matrix measured against its own reference 0.5, and any other pairing reads otherwise (0.75 or 1).
"""
import json
import os
import sys
import types
import weakref

import numpy as np

import container
import convert
import exl3
import override as override_mod
import plan
import reconstruction
from conftest import PREFIX, make_override, make_sources, write_safetensors

RECORD = ("trellis", "suh", "svh", "mul1")


class Tensor:
    """The stub torch's tensor: a numpy array with the methods `rel_errors` calls."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float64)

    @property
    def shape(self):
        return self.a.shape

    def to(self, device):
        return self

    def float(self):
        return self

    def double(self):
        return self

    def t(self):
        return Tensor(self.a.T)

    def pow(self, n):
        return Tensor(self.a ** n)

    def sum(self):
        return Tensor(self.a.sum())

    def __getitem__(self, index):
        return Tensor(self.a[index])

    def __sub__(self, other):
        return Tensor(self.a - other.a)

    def __float__(self):
        return float(self.a)


class Shard:
    """A stub safetensors handle on one real file: reads are logged, and a tensor holds its source's
    value in the shape the file's header gives it."""

    def __init__(self, library, path):
        self.library, self.path, self.header = library, path, exl3.read_header(path)

    def get_tensor(self, name):
        self.library.events.append(("read", self.path, name))
        return Tensor(np.full(self.header[name].shape, self.library.values[os.path.dirname(self.path)]))


class Library:
    """The stub torch, exllamav3 and safetensors, sharing one log of what the step did: `empty_cache`,
    each shard opened (with the shards still open then) and each tensor read."""

    def __init__(self, values):
        self.values, self.events, self.open = values, [], weakref.WeakSet()

    def safe_open(self, path, framework, device):
        assert (framework, device) == ("pt", "cpu")
        self.events.append(("open", path, sorted(s.path for s in self.open)))
        shard = Shard(self, path)
        self.open.add(shard)
        return shard

    def install(self, monkeypatch):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: self.events.append(("empty_cache",)))

        class LinearEXL3:
            """exllamav3's linear: the weight it reconstructs, `[in, out]`, holds its trellis's value."""

            def __init__(self, config, in_features, out_features, suh, svh, trellis, mul1, key):
                self.shape, self.value = (in_features, out_features), float(trellis.a.flat[0])

            def get_weight_tensor(self):
                return Tensor(np.full(self.shape, self.value))

        exl3_module = types.ModuleType("exllamav3.modules.quant.exl3")
        exl3_module.LinearEXL3 = LinearEXL3
        safetensors = types.ModuleType("safetensors")
        safetensors.safe_open = self.safe_open
        modules = {"torch": torch, "safetensors": safetensors, "exllamav3": types.ModuleType("exllamav3"),
                   "exllamav3.modules": types.ModuleType("exllamav3.modules"),
                   "exllamav3.modules.quant": types.ModuleType("exllamav3.modules.quant"),
                   "exllamav3.modules.quant.exl3": exl3_module}
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)


def _base_bf16(root, matrices, shared):
    """The base BF16 checkpoint: every matrix's `.weight` in two shards. Layers 2 and 3 are in the
    shard named `shared`, the name the override's reference shard also has, as `replay.py fetch`
    names its shards after the base's."""
    base = os.path.join(root, "base-bf16")
    os.makedirs(base)
    shards = {}
    for key, m in matrices.items():
        shard = shared if key.startswith((PREFIX + "2.", PREFIX + "3.")) else "model-00007-of-00018.safetensors"
        shards.setdefault(shard, {})[key + ".weight"] = ("BF16", m.shape, bytes(2 * m.rows * m.columns))
    weight_map = {}
    for shard, tensors in shards.items():
        write_safetensors(os.path.join(base, shard), tensors)
        weight_map.update({name: shard for name in tensors})
    with open(os.path.join(base, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": weight_map}, f)
    return base


def _measure(tmp_path, monkeypatch):
    """The step over the plan of the reduced sources, with the override of `make_override` (layer 2's
    gate projection and layer 3's v projection); the sources, the plan, the errors and the log."""
    src = make_sources(str(tmp_path))
    ov = make_override(str(tmp_path), src)
    quant = exl3.Quant(src["quant"])
    matrices, _ = convert.text_matrices(quant)
    objects, _ = plan.plan(container.Container(src["served"]).objects, matrices)
    run = types.SimpleNamespace(quant=quant.root, base=_base_bf16(str(tmp_path), matrices, ov["ref_shard"]),
                                override=ov["override"], reference=ov["bf16"], swapped=set(ov["keys"]),
                                trellis=[p for p in objects if p.key is not None])
    library = Library({run.quant: 1.0, run.base: 1.0, run.override: 2.0, run.reference: 4.0})
    library.install(monkeypatch)
    run.errors = reconstruction.rel_errors(quant, run.base, objects, log=lambda _: None,
                                           override=override_mod.Override(ov["override"], ov["bf16"]))
    run.events = library.events
    return run


def test_a_matrix_the_override_replaces_is_its_record_against_its_reference_and_every_other_the_quants(
        tmp_path, monkeypatch):
    run = _measure(tmp_path, monkeypatch)
    assert len(run.trellis) == 26 and {p.key for p in run.trellis} >= run.swapped
    assert run.errors == {p.name: 0.5 if p.key in run.swapped else 0.0 for p in run.trellis}
    read = {}
    for event in run.events:
        if event[0] == "read":
            key, part = event[2].rsplit(".", 1)
            read.setdefault(key, set()).add((part, os.path.dirname(event[1])))
    for p in run.trellis:
        records, reference = (run.override, run.reference) if p.key in run.swapped else (run.quant, run.base)
        assert read[p.key] == {(part, records) for part in RECORD} | {("weight", reference)}, p.key


def test_one_shard_of_each_source_is_open_at_a_time(tmp_path, monkeypatch):
    run = _measure(tmp_path, monkeypatch)
    opened = [(event[1], event[2]) for event in run.events if event[0] == "open"]
    for path, still_open in opened:
        assert [p for p in still_open if os.path.dirname(p) == os.path.dirname(path)] == [], path
    # the quant's and the base's records come from two shards each, so each source moved shard
    for source in (run.quant, run.base):
        assert len({path for path, _ in opened if os.path.dirname(path) == source}) == 2, source


def test_the_cuda_cache_is_emptied_before_each_matrix(tmp_path, monkeypatch):
    run = _measure(tmp_path, monkeypatch)
    steps = []
    for event in run.events:
        if event[0] == "open":
            continue
        step = event[2].rsplit(".", 1)[0] if event[0] == "read" else event[0]
        if not steps or steps[-1] != step:
            steps.append(step)
    assert steps == [s for p in run.trellis for s in ("empty_cache", p.key)] + ["empty_cache"]
