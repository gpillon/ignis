#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Nvfp4W4a4Workspace {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
};

inline std::size_t nvfp4_w4a4_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 W4A4 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("nvfp4 W4A4 workspace size overflow");
    }
    return count * bytes_per_token;
}

template <class Arena>
Nvfp4W4a4Workspace allocate_nvfp4_w4a4_workspace(Arena& arena, std::int32_t tokens,
                                                 std::int32_t input_rows) {
    if (input_rows <= 0 || (input_rows % 64) != 0) {
        throw std::invalid_argument("nvfp4 W4A4 workspace: invalid K");
    }
    const std::size_t code_bytes =
        nvfp4_w4a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 2);
    const std::size_t scale_bytes =
        nvfp4_w4a4_checked_bytes(tokens, static_cast<std::size_t>(input_rows) / 16);
    const DeviceSpan codes  = arena.alloc_bytes(code_bytes, 256);
    const DeviceSpan scales = arena.alloc_bytes(scale_bytes, 256);
    return {static_cast<std::uint8_t*>(codes.data), static_cast<std::uint8_t*>(scales.data)};
}

inline std::size_t nvfp4_w4a4_workspace_capacity_bytes(std::int32_t tokens,
                                                       std::int32_t input_rows) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_w4a4_workspace(layout, tokens, input_rows);
    return layout.peak_bytes(1);
}

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream);

// ignis (GitHub #652, spec #646): where a W4A4 product's activations sit, which its quantizer
// writes and the product reads. TokenMajor is upstream's: codes [T][K/2 bytes], scales
// [T][K/16 bytes]. KTileMajor is the TMA products' own: codes [K/128][T][64 bytes], scales
// [K/256][T][16 bytes], so the 128-wide slice of K a 256-token tile reads is one piece of 16 KiB
// and its scales one of 4 KiB, whole 128-byte lines, where token-major gives 256 pieces of 64
// bytes and 256 of 16. The same values in other places: a product writes the same bytes from
// either. The weights keep their layout, which the decode and verify routes read too.
enum class Nvfp4W4a4ActivationLayout { TokenMajor, KTileMajor };

// ignis (GitHub #652): quantizes `x` for the W4A4 products of one call on `stream`, into the same
// workspace as launch_nvfp4_w4a4_quantize, and returns the layout it wrote, the one those products
// read. On the TMA route, a whole number of 256-token tiles from 1,024 tokens, that is the layout
// the load running on `stream` names: KTileMajor unless IGNIS_W4A4_ACTIVATION_LAYOUT=token
// (ignis's kernel/src/w4a4_activation_layout.h). Anywhere else it is TokenMajor, written by
// launch_nvfp4_w4a4_quantize.
Nvfp4W4a4ActivationLayout launch_nvfp4_w4a4_quantize_for_products(const Tensor& x,
                                                                  const Weight& weight,
                                                                  Nvfp4W4a4Workspace workspace,
                                                                  cudaStream_t stream);

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
