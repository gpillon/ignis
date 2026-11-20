// ignis kernel leaf: launches of ours of the vendored NVFP4 A16 small-T kernel for the DFlash2
// drafter (GitHub #694, spec #684) -- an ADAPTED op (ADR 0051 tier 2), leaf-internal: the launch
// restates the vendored launcher's (kernel/src/dflash2_small_t.cu names its upstream, as
// kernel/NOTICE does). The kernel is the vendored template (ops/linear/nvfp4/nvfp4_small_t.cuh),
// instantiated here and not changed, on the vendored production schedule for the call's width with
// one field of its own: 4 warps a CTA, the schedule's own choice past 16 columns. What is ours is
// how many warps a CTA holds and which rows of the weight it computes. A row's sum is one warp's,
// in an order that depends on neither, so each launch writes the bytes `ninfer::ops::linear` writes
// for the same rows.
//
// Two levers, each behind a measurement switch read at each call, so a graph keeps what its
// capture read, and once at load (kernel/src/model.cu), so a value it does not know refuses the
// load by name. Each was measured on the RTX 5090 alone (kernel/src/card_lever.h): unset, it runs
// on a card of compute capability 12 and every other card keeps today's path; `1` runs it on any
// card; `0` is today's path everywhere.
//
//   IGNIS_DFLASH2_APPEND_KV_ONLY  on: the append after a verify round
//       (IgnisA16Call::General) projects only the key and value rows of the drafter's q | k | v
//       weight [6144, 5120], its last 2,048 rows, the ones it keeps: the 4,096 query rows are not
//       computed. At 5 to 32 columns, where today's product is one launch of the small-T kernel;
//       2,048 rows on 4 warps a CTA are 256 CTAs. Off: the whole product, today's append. A
//       prefill chunk's taps and a carried anchor keep the whole product either way.
//
//   IGNIS_DFLASH2_SMALL_T_FILL  on: the forward's conv projections [1280, 5120] and its
//       q | k | v product [6144, 5120] at 5 to 16 columns run on 4 warps a CTA where the vendored
//       schedule holds 8 (5 to 12 columns) or 16 (13 to 16): at 16 columns, a one-lane round at the
//       extended shape or a two-lane one at the base shape, 160 and 768 CTAs where it launches 40
//       and 192 on the RTX 5090's 170 multiprocessors; at 8, a one-lane round at the base shape,
//       160 and 768 where it launches 80 and 384. Off: the vendored launch.
#pragma once

#include "ignis_nvfp4_a16_mma.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::dflash2_small_t {

// Whether IGNIS_DFLASH2_APPEND_KV_ONLY at this call, on this card, turns its lever on
// (`ignis::card_lever::takes_here`). Throws std::invalid_argument, naming the variable, on a value
// but `1` or `0`.
bool append_kv_only();

// The same for IGNIS_DFLASH2_SMALL_T_FILL.
bool small_t_fill();

// Whether the append's product of `tokens` columns through `qkv`, a call of kind `call`, projects
// the key and value rows alone (`key_value_linear`): `qkv` is the drafter's q | k | v weight, the
// call is one launch of the small-T kernel today, and IGNIS_DFLASH2_APPEND_KV_ONLY turns the lever
// on (`append_kv_only`).
bool projects_key_value_only(const ninfer::Weight &qkv, std::int32_t tokens, IgnisA16Call call);

// kv[2048, T] = rows 4096..6143 of `qkv` times x[5120, T], for a call `projects_key_value_only`
// admits: the bytes `ninfer::ops::linear` writes to those rows of its [6144, T] output.
void key_value_linear(const ninfer::Tensor &x, const ninfer::Weight &qkv, ninfer::Tensor &kv,
                      cudaStream_t stream);

// Whether `forward_linear` launches `weight` at `tokens` columns on 4 warps a CTA: a conv
// projection or the q | k | v weight of the drafter at 5 to 16 columns, and
// IGNIS_DFLASH2_SMALL_T_FILL turns the lever on (`small_t_fill`).
bool fills(const ninfer::Weight &weight, std::int32_t tokens);

// out[N, T] = weight[N, K] * x[K, T], the bytes of `ninfer::ops::linear(x, weight, out, stream)`:
// on 4 warps a CTA where `fills` says so, through `ninfer::ops::linear` otherwise.
void forward_linear(const ninfer::Tensor &x, const ninfer::Weight &weight, ninfer::Tensor &out,
                    cudaStream_t stream);

// How many launches of ours (`key_value_linear`, `forward_linear` on 4 warps) this process has
// enqueued: what a test reads to know which route a call took, the bytes being the same.
std::uint64_t launches();

} // namespace ignis::dflash2_small_t
