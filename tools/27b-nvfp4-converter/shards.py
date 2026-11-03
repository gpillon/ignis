"""Safetensors reads that keep at most one shard open, in calibration and in conversion.

On Windows each open `safe_open` handle costs its file's size in commit charge: the 27B's 18 shards
open at once are 55.6 GB, which is the failure GitHub #422 hit (`os error 1455`) and the leak ninfer's
calibration has (`StreamingLayerLoader._open` keeps every shard it touched). Every handle this tool opens
goes through `ShardReader`, which holds one and closes it before it opens the next; `safe_open` is
looked up on this module at each open, so a test can count the handles. Headers are read without a
handle (`header`).
"""
import json
import os
import struct

from safetensors import safe_open

INDEX = "model.safetensors.index.json"


def header(path):
    """A safetensors file's tensors: name -> (dtype, shape), from its header alone."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        raw = json.loads(f.read(n))
    raw.pop("__metadata__", None)
    return {name: (v["dtype"], tuple(v["shape"])) for name, v in raw.items()}


def weight_map(root):
    """A checkpoint's tensor -> shard map, from its index."""
    with open(os.path.join(root, INDEX), encoding="utf-8") as f:
        return dict(json.load(f)["weight_map"])


class ShardReader:
    """Tensors by name from the files of `files` (name -> file under `root`), one file open at a time."""

    def __init__(self, root, files):
        self.root = root
        self.files = files
        self._file = None
        self._context = None
        self._handle = None

    @classmethod
    def checkpoint(cls, root):
        return cls(root, weight_map(root))

    @classmethod
    def single(cls, path):
        root, name = os.path.split(path)
        return cls(root, {tensor: name for tensor in header(path)})

    def get(self, name):
        """The tensor `name`: safetensors copies it out of the file, so it outlives the handle."""
        file = self.files[name]
        if file != self._file:
            self.close()
            self._context = safe_open(os.path.join(self.root, file), framework="pt", device="cpu")
            self._handle = self._context.__enter__()
            self._file = file
        return self._handle.get_tensor(name)

    def close(self):
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._file = self._context = self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
