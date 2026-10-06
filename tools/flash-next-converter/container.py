"""Reading expert records back out of the packed `.ninfer` v2 container (for `convert.py verify`).

The framing is the reader's (`crates/artifact/src/lib.rs`): an 8-byte magic, the directory's
JSON length as a little-endian u64, the JSON directory, then the payload from the next
4096-byte boundary; object offsets are payload-relative. Expert projections are one tensor
each, named `layers.{L}.mlp.experts.{E}.gate_up_proj` / `.down_proj`, with the K in the
format code (layout.md §4).
"""
import json
import struct

import layout

MAGIC = b"NINFER\x00\x02"
PAYLOAD_ALIGNMENT = 4096
K2_OF_FORMAT = {"TRELLIS_MUL1_K2": 4, "TRELLIS_MUL1_K2P5": 5, "TRELLIS_MUL1_K3": 6, "TRELLIS_MUL1_K4": 8}
PROJ_NAME = {"gu": "gate_up_proj", "dn": "down_proj"}


def expert_name(layer, expert, proj):
    return f"layers.{layer}.mlp.experts.{expert}.{PROJ_NAME[proj]}"


class Container:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            prefix = f.read(16)
            if prefix[:8] != MAGIC:
                raise ValueError(f"{path}: not a .ninfer v2 container")
            n = struct.unpack("<Q", prefix[8:16])[0]
            self.directory = json.loads(f.read(n))
        end = 16 + n
        self.payload_start = -(-end // PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT
        self.objects = {o["name"]: o for o in self.directory["objects"]}

    def expert_record(self, layer, expert, proj):
        """(k2, exllamav3 tensors) of one expert projection, read from the container's bytes."""
        o = self.objects[expert_name(layer, expert, proj)]
        k2 = K2_OF_FORMAT[o["format"]]
        if o["bytes"] != layout.record_bytes(proj, k2):
            raise ValueError(f"{o['name']}: {o['bytes']} bytes, a {proj} K={k2 / 2:g} record is "
                             f"{layout.record_bytes(proj, k2)}")
        entry = layout.IndexEntry(expert, proj, k2, o["bytes"], self.payload_start + o["offset"])
        return k2, layout.read_record(self.path, entry)
