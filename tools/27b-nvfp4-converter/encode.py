"""The encoders: a tensor's values to the bytes the container stores (docs/formats/27b-nvfp4-work-tree.md §3).

Ports of ninfer's, which made the served artifact and are its byte-level reference:

- `nvfp4` is `tools/convert/qwen3_8_27b/nvfp4_encode.py`'s `quantize_nvfp4` (profile
  `NVFP4_MAXABS_DIVISOR_RNE_V1`: the weight divisor `binary32(2688 / max|W|)`, one E4M3FN scale per
  16 elements of `W * d_w`, E2M1 codes rounded to nearest even) laid out by
  `tools/artifact/layouts.py`'s `encode_nvfp4` (the packed codes, zeros to 256, the swizzled scales, the
  divisor);
- `grouped` is `tools/convert/common/quantize.py`'s `quantize_and_encode` (binary16 group scales computed
  on the host in binary64, codes `round(x / scale)` clamped) laid out by `encode_row_split`;
- `direct` is `encode_direct`: the elements' little-endian words.

The arithmetic is torch's on the device given; the scale words of a grouped format are computed on the
host, as ninfer does, because a device's division is not correctly rounded at every binary16 boundary.
"""
import struct

import numpy as np
import torch

import inventory

ENCODER_PROFILE = "NVFP4_MAXABS_DIVISOR_RNE_V1"
FULL_RANGE = 2688.0  # 6 * 448: the largest product of an E2M1 and an E4M3FN magnitude
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_TIE_UP = (False, True, False, True, False, True, False)
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP16_MIN_SUBNORMAL = 2.0 ** -24
QRANGE = {inventory.W8: (-127, 127), inventory.Q4: (-8, 7), inventory.Q5: (-16, 15), inventory.Q6: (-32, 31)}


def binary32(value):
    """A Python float rounded to binary32 (to nearest even), as a float."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def divisor_of(amax):
    """`binary32(2688 / amax)`, 1 when amax is 0: the weight and the activation divisors' formula."""
    return 1.0 if amax == 0.0 else float(torch.tensor(FULL_RANGE / amax).item())


def e2m1_codes(values):
    """Finite FP32 values to saturating E2M1 codes (0..15, sign bit 3), ties to the even neighbour."""
    sign = torch.where(torch.signbit(values), torch.tensor(8, dtype=torch.uint8, device=values.device),
                       torch.tensor(0, dtype=torch.uint8, device=values.device))
    magnitude = values.abs()
    code = torch.zeros_like(magnitude, dtype=torch.long)
    for i, boundary in enumerate(_E2M1_BOUNDARIES):
        code = torch.where((magnitude > boundary) | ((magnitude == boundary) & _E2M1_TIE_UP[i]), code + 1, code)
    return code.to(torch.uint8) | sign


def nvfp4_words(weight, device):
    """(packed codes uint8 [N, K/2], natural scales uint8 [N, K/16], weight divisor) of a `[N, K]` parent."""
    if weight.dim() != 2 or weight.shape[1] % 16:
        raise ValueError("an NVFP4 parent is rank 2 with K divisible by 16")
    source = weight.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(source).all():
        raise ValueError("an NVFP4 parent holds NaN or infinity")
    rows, columns = source.shape
    amax = source.abs().amax().item()
    divisor = binary32(divisor_of(amax))
    if not divisor > 0.0:
        raise ValueError("the NVFP4 weight divisor is not finite and positive")
    blocks = (source * torch.tensor(divisor, dtype=torch.float32, device=device)).reshape(rows, columns // 16, 16)
    scales = (blocks.abs().amax(dim=2) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    scale_values = scales.view(torch.float8_e4m3fn).to(torch.float32)
    nonzero = scale_values > 0
    ratio = torch.where(nonzero.unsqueeze(-1),
                        blocks / torch.where(nonzero, scale_values, torch.ones_like(scale_values)).unsqueeze(-1),
                        torch.zeros_like(blocks))
    codes = e2m1_codes(ratio).reshape(rows, columns)
    pairs = codes.reshape(rows, -1, 2)
    packed = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)
    return packed.cpu(), scales.reshape(rows, columns // 16).contiguous().cpu(), divisor


def nvfp4_dequantize(packed, scales, divisor):
    """The represented values `E2M1 * E4M3FN / d_w` in FP32 (for the reconstruction error), on the words'
    device."""
    table = torch.tensor([(-1.0 if c & 8 else 1.0) * _E2M1_MAGNITUDES[c & 7] for c in range(16)], device=packed.device)
    rows, groups = scales.shape
    codes = torch.stack((packed.to(torch.int64) & 0xF, packed.to(torch.int64) >> 4), dim=2).reshape(rows, groups * 16)
    values = table[codes].reshape(rows, groups, 16)
    return (values * scales.view(torch.float8_e4m3fn).to(torch.float32).unsqueeze(-1) / divisor).reshape(rows, -1)


def nvfp4_layout(packed, scales, divisor, shape):
    """The `blockscale-k16-m128x4-v1` payload of exact NVFP4 words."""
    n, k = shape
    scale_offset, divisor_offset, size = inventory.block_scale_geometry(shape)
    if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
    swizzled = scales.reshape(n // 128, 4, 32, k // 64, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(-1)
    payload = bytearray(size)
    payload[: n * k // 2] = packed.numpy().tobytes()
    payload[scale_offset: divisor_offset] = swizzled.numpy().tobytes()
    payload[divisor_offset:] = struct.pack("<f", divisor)
    return bytes(payload)


def nvfp4(weight, device):
    """(payload, weight divisor, relative Frobenius error) of one NVFP4 parent."""
    packed, scales, divisor = nvfp4_words(weight, device)
    payload = nvfp4_layout(packed, scales, divisor, tuple(weight.shape))
    reconstructed = nvfp4_dequantize(packed.to(device), scales.to(device), divisor)
    return payload, divisor, relative_error(weight.to(device), reconstructed)


def relative_error(original, reconstructed):
    """`|W - W_hat| / |W|` (Frobenius), in binary64 on the tensors' device."""
    a = original.detach().double()
    d = (a - reconstructed.detach().double()).norm().item()
    n = a.norm().item()
    return 0.0 if n == 0.0 and d == 0.0 else float("inf") if n == 0.0 else d / n


def _scale_words(max_abs, qmax):
    """Canonical binary16 group scales and their binary32 reciprocals, on the host."""
    host = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(host).all():
        raise ValueError("a grouped source holds NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = (host.astype(np.float64) / float(qmax)).astype(np.float32).astype(np.float16)
    underflow = (scale == 0) & (host > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.float16(_FP16_MIN_SUBNORMAL)
    if np.any((host > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("a grouped scale is not finite and positive")
    reciprocal = np.zeros(host.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return torch.from_numpy(scale), torch.from_numpy(reciprocal)


def grouped(weight, fmt, device):
    """The `row-split-k128-v1` payload of a W8/Q4/Q5/Q6 matrix."""
    group, bits = inventory.GROUP[fmt]
    qmin, qmax = QRANGE[fmt]
    n, k = weight.shape
    k_pad, groups, base_row, high_row, _, high_offset, scale_offset, size = inventory.row_split_geometry(fmt, (n, k))
    logical = weight.detach().to(device=device, dtype=torch.float32)
    if k_pad != k:
        padded = torch.zeros((n, k_pad), dtype=torch.float32, device=device)
        padded[:, :k].copy_(logical)
        logical = padded
    blocks = logical.reshape(n, groups, group)
    scales, reciprocal = _scale_words(blocks.abs().amax(dim=2), qmax)
    codes = torch.clamp(torch.round(blocks * reciprocal.to(device).unsqueeze(-1)), qmin, qmax).to(torch.int8).cpu()
    flat = codes.reshape(-1, group)
    if bits == 8:
        base = flat.view(torch.uint8)
        high = None
    else:
        low = flat.to(torch.int16) & 0x0F
        base = (low[:, 0::2] | (low[:, 1::2] << 4)).to(torch.uint8)
        high = None
        if bits > 4:
            hb = bits - 4
            upper = ((flat.to(torch.int32) & ((1 << bits) - 1)) >> 4) & ((1 << hb) - 1)
            bit_planes = ((upper.unsqueeze(-1) >> torch.arange(hb, dtype=torch.int32)) & 1).reshape(flat.shape[0], -1, 8)
            high = (bit_planes * (1 << torch.arange(8, dtype=torch.int32))).sum(-1).to(torch.uint8)
    payload = bytearray(size)
    payload[: n * base_row] = base.numpy().tobytes()
    if high is not None:
        payload[high_offset: high_offset + n * high_row] = high.numpy().tobytes()
    payload[scale_offset:] = scales.contiguous().view(torch.uint8).numpy().tobytes()
    return bytes(payload)


DTYPE = {inventory.BF16: torch.bfloat16, inventory.FP32: torch.float32, inventory.I32: torch.int32}


def direct(tensor, fmt):
    """The `contiguous-le-v1` words of a BF16, FP32 or I32 tensor, which must already be of that dtype."""
    if tensor.dtype != DTYPE[fmt]:
        raise TypeError(f"{fmt} takes {DTYPE[fmt]}, not {tensor.dtype}")
    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def encode(obj, tensor, device):
    """(payload, NVFP4 record or None) of a tensor object from its materialized tensor."""
    if tuple(tensor.shape) != obj.shape:
        raise ValueError(f"{obj.name}: materialized {list(tensor.shape)}, the inventory's is {list(obj.shape)}")
    if obj.format == inventory.NVFP4:
        payload, divisor, error = nvfp4(tensor, device)
        return payload, {"weight_scale_divisor": divisor, "relative_frobenius_error": error}
    if obj.format in QRANGE:
        return grouped(tensor, obj.format, device), None
    return direct(tensor, obj.format), None
