"""The inventory the converter writes to: the served artifact's directory at the 27B's widths."""
import collections
import os

import pytest

import container
import inventory

SERVED = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer"


def test_the_fixture_has_the_served_counts_at_reduced_widths():
    objs = inventory.objects(inventory.Geometry.fixture())
    assert len(objs) == 1325
    per_unit = collections.Counter(o.unit for o in objs)
    assert per_unit == {"frontend": 6, "text": 908, "mtp": 12, "vision": 333, "dflash2": 66}
    formats = collections.Counter(o.format for o in objs if o.kind == "tensor")
    # ninfer's validate_inventory: BF16 575, FP32 343, I32 1, Q4 55, Q5 54, Q6 1, W8 9, NVFP4 281.
    assert formats == {"BF16": 575, "FP32": 343, "I32": 1, "Q4G64_F16S": 55, "Q5G64_F16S": 54,
                       "Q6G64_F16S": 1, "W8G32_F16S": 9, "NVFP4": 281}
    text_nvfp4 = [o for o in objs if o.format == "NVFP4" and o.unit == "text"]
    assert len(text_nvfp4) == 247
    assert sum(o.name.endswith("/input_scale_divisor") for o in objs) == 247


@pytest.mark.skipif(not os.path.isfile(SERVED), reason=f"the served artifact is not on this machine ({SERVED})")
def test_the_27b_inventory_is_the_served_artifacts_directory():
    served = container.Container(SERVED)
    ours = inventory.objects(inventory.Geometry.qwen3_8_27b())
    assert [o["name"] for o in served.objects] == [o.name for o in ours]
    for theirs, o in zip(served.objects, ours):
        if o.kind == "resource":
            assert theirs["kind"] == "resource" and theirs["encoding"] == "raw-bytes-v1", o.name
            continue
        assert (theirs["format"], theirs["layout"], tuple(theirs["shape"]), theirs["bytes"]) == \
            (o.format, o.layout, o.shape, inventory.encoded_size(o)), o.name
