"""Read-only access to a `.ninfer` v2 container: its directory and its objects' payloads.

Copied from tools/27b-16gb-converter/container.py, unchanged but for this paragraph: `verify` reads
the packed artifact back with it, and the inventory's test reads the served artifact's directory.
The format is `crates/artifact`'s reader (`Reader::open`): an 8-byte magic `NINFER\\0\\2`, the
directory's length as a little-endian u64, the JSON directory, then the payload, which starts at the
next multiple of 4096; every object's `offset` is relative to that start. This module never writes a container: the packer does (`ignis-artifact-pack`).
"""
import json
import struct

MAGIC = b"NINFER\x00\x02"
PREFIX_BYTES = 16
PAYLOAD_ALIGNMENT = 4096


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


class Container:
    """One container's identity and directory (`objects`, in directory order), and its payloads."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            prefix = f.read(PREFIX_BYTES)
            if len(prefix) < PREFIX_BYTES or prefix[:8] != MAGIC:
                raise ValueError(f"{self.path} is not a NInfer v2 container")
            (json_bytes,) = struct.unpack("<Q", prefix[8:16])
            directory = json.loads(f.read(json_bytes))
        self.identity = directory["identity"]
        self.objects = directory["objects"]
        self.payload_offset = align_up(PREFIX_BYTES + json_bytes, PAYLOAD_ALIGNMENT)
        self.by_name = {o["name"]: o for o in self.objects}
        if len(self.by_name) != len(self.objects):
            raise ValueError(f"{self.path} has duplicate object names")

    def span(self, name):
        """(absolute file offset, bytes) of an object's payload."""
        o = self.by_name[name]
        return self.payload_offset + o["offset"], o["bytes"]

    def read(self, name):
        offset, size = self.span(name)
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        if len(data) != size:
            raise ValueError(f"{self.path}: {name} extends beyond the file")
        return data
