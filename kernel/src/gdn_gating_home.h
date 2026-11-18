// ignis kernel leaf: where the 27B's GDN gating runs in a decode or verify round -- OURS (spec #672,
// GitHub #681), leaf-internal.
//
// The gating (`ninfer::ops::gdn_gating_proj`: the a|b product of the layer's normed input `h`, then
// the gates g and beta from it) reads only `h` and the layer's weights, and only the recurrence reads
// its gates. Earlier releases run it on the round's stream between the conv and the recurrence, so
// every GDN layer's serial chain waits on it. Its homes:
//   - chain: that serial chain, the round of earlier releases.
//   - branch: on a stream of its own, forked from the round's after the input RMSNorm and joined back
//     just before the recurrence, so it runs beside the input projection and the conv, in the decode
//     and the verify round. The product is the vendored op's, unchanged. Its workspace is a span of
//     the layer's scratch held to the layer's end (`Beside`): the input projection allocates its own
//     from the same arena while the branch runs.
// The gates are the same bytes in both homes (ignis_kernel_gdn_gating_home_test; the round identity
// test's arms, crates/core/tests/round_identity_gpu.rs).
//
// Decided at each capture of the decode graphs, from the card and the measurement switch
// IGNIS_GDN_GATING, for every round after it, replayed or eager; before a load's first capture a
// round runs on the chain. Compute capability 12, where the round identity test ran, takes the home
// the switch names, `kUnset` when it names none; every other card keeps the chain whatever it says.
#pragma once

#include "decode_graph_copies.h"
#include "process_environment.h"

#include "flash_next/side_branch.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::gdn_gating {

// The values are `ignis_program_stats`'s `gdn_gating` (kernel/include/ignis_step.h).
enum class Home : std::uint32_t { Chain = 0, Branch = 1 };

// The home of a round captured with IGNIS_GDN_GATING unset, on a card that takes the switch: the
// branch, the faster of the two in the spec's round timing, alternated with the chain (GitHub #681).
inline constexpr Home kUnset = Home::Branch;

// The switch in the process environment (kernel/src/process_environment.h), read at each capture.
// Throws std::invalid_argument, naming the variable, on any value but `chain` or `branch`.
inline Home switch_at_capture() {
  return ignis::named_switch("IGNIS_GDN_GATING", kUnset, {{"chain", Home::Chain}, {"branch", Home::Branch}});
}

// The one rule: the home the switch names on a card the round identity test ran on (compute
// capability major 12, the copy-free round's cards), the chain on every other.
constexpr Home home_on(Home named, int compute_capability_major) {
  return ignis::decode_graph_copies::runs_on(compute_capability_major) ? named : Home::Chain;
}

// One GDN layer's gating, `ninfer::ops::gdn_gating_proj`'s operands: the normed input `h`, the
// contiguous a|b parent `ab`, `a_log` and `dt_bias`, and the gates `g` and `beta` it writes.
struct Gating {
  const ninfer::Tensor &h;
  const ninfer::Weight &ab;
  const ninfer::Tensor &a_log;
  const ninfer::Tensor &dt_bias;
  ninfer::Tensor &g;
  ninfer::Tensor &beta;
};

// The branch home's fork of one layer's gating: the vendored `gdn_gating_proj` of `gating`, queued
// on `branch.stream` after everything queued on `stream` so far, its workspace a span of `scratch`
// that stays allocated until the caller's scope of `scratch` ends, so nothing the caller allocates
// from it meanwhile shares its bytes. `join` makes `stream` wait for the branch; a Beside left
// without it joins in its destructor, so a round that fails after the fork never leaves the branch
// open (a capture would then fail to end). Capturable: the fork and the join are edges of the
// captured graph. Throws what the op throws, joined, and std::runtime_error on a CUDA error.
class Beside {
public:
  Beside(const ignis::flash_next::SideBranch &branch, cudaStream_t stream, const Gating &gating,
         ninfer::DeviceArena &scratch);
  ~Beside();

  Beside(const Beside &) = delete;
  Beside &operator=(const Beside &) = delete;

  void join();

private:
  void join_quietly() noexcept;

  const ignis::flash_next::SideBranch &branch_;
  cudaStream_t stream_;
  bool open_ = false;
};

} // namespace ignis::gdn_gating
