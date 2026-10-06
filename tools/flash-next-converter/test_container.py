"""Locating expert records in a packed container (the post-pack verify reads them)."""
import json
import struct

import numpy as np

import container
import layout
from test_layout import fake_record


def write_container(path, records):
    """A minimal v2 container: prefix, JSON directory, 4096-aligned payload of records."""
    objects, payload, off = [], b"", 0
    for (L, e, p), (k2, rec) in records.items():
        body = layout.record_payload(p, k2, rec)
        fmt = {v: k for k, v in container.K2_OF_FORMAT.items()}[k2]
        i, o = layout.SHAPES[p]
        objects.append({"name": container.expert_name(L, e, p), "kind": "tensor", "shape": [o, i], "format": fmt,
                        "layout": "trellis-tile16-v1", "offset": off, "bytes": len(body)})
        payload += body
        off += len(body)
    js = json.dumps({"identity": {"model_id": "m", "weights_id": "w"}, "objects": objects}).encode()
    head = container.MAGIC + struct.pack("<Q", len(js)) + js
    head += bytes(-len(head) % 4096)
    path.write_bytes(head + payload)


def test_records_come_back_from_the_container_bit_for_bit(tmp_path):
    recs = {(3, 0, "gu"): (5, fake_record("gu", 5, 1)), (3, 0, "dn"): (8, fake_record("dn", 8, 2)),
            (3, 1, "gu"): (4, fake_record("gu", 4, 3))}
    write_container(tmp_path / "a.ninfer", recs)
    c = container.Container(str(tmp_path / "a.ninfer"))
    assert c.payload_start % 4096 == 0
    for (L, e, p), (k2, want) in recs.items():
        got_k2, got = c.expert_record(L, e, p)
        assert got_k2 == k2
        for name in ("trellis", "suh", "svh"):
            assert np.array_equal(got[name].view(np.uint8), want[name].view(np.uint8))
