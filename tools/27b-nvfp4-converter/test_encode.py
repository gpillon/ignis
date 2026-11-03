"""The encoders' bytes: known words, and ninfer's encoders' bytes (the served artifact's reference) on the
same values when ninfer's checkout is on this machine."""
import os
import struct
import sys

import pytest
import torch

import encode
import inventory

NINFER = "F:/ai/q38/ninfer"


def test_e2m1_rounds_to_nearest_even_and_saturates():
    # ninfer nvfp4_encode.self_test's grid and codes.
    grid = torch.tensor([0.0, 0.24, 0.25, 0.26, 0.5, 0.74, 0.75, 0.76, 1.0, 1.24, 1.25, 1.26, 1.5, 1.74, 1.75, 1.76,
                         2.0, 2.49, 2.5, 2.51, 3.0, 3.49, 3.5, 3.51, 4.0, 4.99, 5.0, 5.01, 5.9, 6.0, 7.0, 100.0])
    codes = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7]
    assert encode.e2m1_codes(grid).tolist() == codes
    assert encode.e2m1_codes(-grid).tolist() == [c | 8 for c in codes]


def test_an_nvfp4_payload_is_codes_scales_and_the_divisor():
    # One 128 x 64 block of ones but w[5, 7] = 3: amax 3, so d_w = 2688 / 3 = 896 and y = 896 w.
    w = torch.ones(128, 64, dtype=torch.bfloat16)
    w[5, 7] = 3.0
    payload, divisor, _ = encode.nvfp4(w, "cpu")
    assert divisor == 896.0
    assert len(payload) == 4096 + 512 + 4
    assert struct.unpack("<f", payload[-4:])[0] == 896.0
    # Row 5's first group holds 2688: scale E4M3FN(2688 / 6 = 448) = 0x7E, so 896 -> 2.0 (code 4) and
    # 2688 -> 6.0 (code 7); elements 6 and 7 share byte 3 of the row, the lower element in the low nibble.
    assert payload[5 * 32 + 3] == (7 << 4) | 4
    # Every other group's amax is 896: scale E4M3FN(149.33) = 144 (0x71), and 896 / 144 = 6.2 saturates at 6.
    assert payload[0] == (7 << 4) | 7
    scales = payload[4096: 4096 + 512]
    # The swizzle puts row r = 32a + b, group g at 16b + 4a + g: row 5, group 0 is byte 80.
    assert scales[80] == 0x7E
    assert sorted(set(scales[:80] + scales[81:])) == [0x71]


def test_a_grouped_payload_is_the_planes_at_256_byte_offsets():
    # Q5 [2, 96]: K pads to 128, two groups of 64 per row. Row 0 is 1..96 / 8, so group 0's amax is 8 and
    # its scale 8 / 15 (binary16 0.5332 = 0x3844); the padding group of row 1 (all zero) has scale 0.
    w = torch.zeros(2, 96, dtype=torch.bfloat16)
    w[0] = torch.arange(1, 97, dtype=torch.float32) / 8
    payload = encode.grouped(w, inventory.Q5, "cpu")
    k_pad, groups, base_row, high_row, scale_row, high_off, scale_off, size = inventory.row_split_geometry(
        inventory.Q5, (2, 96))
    assert (k_pad, groups, base_row, high_row, high_off, scale_off, size) == (128, 2, 64, 16, 256, 512, 520)
    assert len(payload) == 520
    assert payload[scale_off: scale_off + 2] == struct.pack("<e", 8 / 15)
    assert payload[scale_off + 6: scale_off + 8] == b"\x00\x00"
    # Element 63 of row 0 is 8.0: code round(8 / 0.5332) = 15, low nibble 0xF in the high half of byte 31,
    # and no high bit (15 < 16).
    assert payload[31] >> 4 == 0xF


@pytest.mark.skipif(not os.path.isdir(NINFER), reason=f"ninfer's checkout is not on this machine ({NINFER})")
def test_the_bytes_are_ninfers_on_the_same_values():
    sys.path.insert(0, NINFER)
    try:
        from tools.artifact.layouts import encode_nvfp4
        from tools.convert.common.quantize import quantize_and_encode
        from tools.convert.qwen3_8_27b import nvfp4_encode
    finally:
        sys.path.remove(NINFER)
    rng = torch.Generator().manual_seed(468)
    for shape in ((128, 64), (256, 192), (384, 128)):
        w = (torch.randn(shape, generator=rng) * 0.03).to(torch.bfloat16)
        w[0, 0] = 0.0
        words = nvfp4_encode.quantize_nvfp4(w, device="cpu")
        theirs = encode_nvfp4(words.packed_codes, words.natural_scales, struct.pack("<f", words.weight_divisor), shape)
        payload, divisor, _ = encode.nvfp4(w, "cpu")
        assert payload == theirs, shape
        assert divisor == words.weight_divisor
    for fmt in (inventory.W8, inventory.Q4, inventory.Q5, inventory.Q6):
        for shape in ((3, 32), (5, 96), (64, 128), (7, 300)):
            w = (torch.randn(shape, generator=rng) * 0.05).to(torch.bfloat16)
            w[1, : shape[1] // 2] = 0.0
            assert encode.grouped(w, fmt, "cpu") == quantize_and_encode(w, fmt, device="cpu"), (fmt, shape)
