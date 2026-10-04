"""FP8 row-scale payloads: the container's FP8_E4M3FN_ROW_BF16S / row-scale-v1 (layout.md §6.1)."""
import numpy as np
import torch

import fp8


def test_payload_geometry_is_the_readers():
    # crates/artifact row_scale_geometry_fp8: [256, 512] -> codes 131072, scales at 131072, 131584 bytes
    w = torch.randn(256, 512, dtype=torch.bfloat16)
    p = fp8.encode(w)
    assert len(p) == 131584
    w2 = torch.randn(48, 2560, dtype=torch.bfloat16)  # 122880 code bytes -> scale plane at 122880
    assert len(fp8.encode(w2)) == 122880 + 96
    w3 = torch.randn(3, 100, dtype=torch.bfloat16)    # 300 code bytes -> padded to 512
    assert len(fp8.encode(w3)) == 512 + 6


def test_decode_is_code_times_bf16_scale_and_close_to_the_weight():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(64, 256, generator=g) * torch.logspace(-3, 1, 64)[:, None]).to(torch.bfloat16)
    p = fp8.encode(w)
    back = fp8.decode(p, (64, 256))
    codes = torch.frombuffer(bytearray(p[:64 * 256]), dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    scales = torch.frombuffer(bytearray(p[64 * 256:64 * 256 + 128]), dtype=torch.bfloat16).float()
    assert torch.equal(back.float(), (codes.view(64, 256) * scales[:, None]).to(torch.bfloat16).float())
    rel = ((back.float() - w.float()).abs() / w.float().abs().amax(1, keepdim=True)).max()
    assert rel < 2 ** -4 + 2 ** -8


def test_no_code_saturates_and_the_scale_is_rounded_up():
    w = torch.tensor([[448.0 * 3.0, -1.0, 0.5], [1e-3, 2e-3, -3e-3]], dtype=torch.bfloat16)
    p = fp8.encode(w)
    codes = np.frombuffer(p[:6], dtype=np.uint8)
    assert not np.any((codes & 0x7F) == 0x7F)        # 0x7F / 0xFF are E4M3FN's NaN
    scales = torch.frombuffer(bytearray(p[256:260]), dtype=torch.bfloat16).float()
    assert torch.all(scales * 448 >= w.float().abs().amax(1))


def test_an_all_zero_row_has_scale_one_and_zero_codes():
    w = torch.zeros(2, 32, dtype=torch.bfloat16)
    p = fp8.encode(w)
    assert p[:64] == bytes(64)
    assert torch.frombuffer(bytearray(p[256:260]), dtype=torch.bfloat16).tolist() == [1.0, 1.0]
