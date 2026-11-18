// ignis kernel leaf: the GDN layer's causal conv, writing q, k and v split -- an ADAPTED op
// (ADR 0051 tier 2), leaf-internal (GitHub #507, spec #503 "Prefill work outside the kernels";
// GitHub #680, spec #672, for the decode round's form).
//
// The GDN layer's conv input is one feature-major plane `x` [C, T] whose channels are
// [query | key | value] (C = 10240 = 2048 | 2048 | 6144 on the 27B), and its recurrence reads
// the three as contiguous planes of their own. The vendored causal conv writes one [C, T]
// plane, so the prefill split it with three strided device-to-device copies per layer and
// chunk. This conv writes each channel straight into its own plane: query [Cq, T], key [Ck, T],
// value [Cv, T], and moves the rolling taps the same way the vendored op does.
//
// The arithmetic is the vendored prefill form's, unchanged: every output equals, bit for bit,
// `ninfer::ops::causal_conv1d_silu` on the same input followed by the old copies, at every T
// (the vendored wrapper's decode, small-T and sequence forms compute the same expression, and
// kernel/tests/test_gdn_conv_split.cu holds all of them to it).
//
// `split_conv_slots` is the decode round's form (GitHub #680): one column per lane, each lane's
// taps in the pool slot the round staged for it, as the vendored snapshot form reads and writes
// them; its planes are bit for bit `ninfer::ops::causal_conv1d_silu_snapshot` followed by the
// three strided copies that split its plane in the decode graph's copying form.
//
// GitHub #681 (spec #672): `split_conv_slots` with a `Gating` also computes the layer's gating, its
// fused home (kernel/src/gdn_gating_home.h), in CTAs of their own launched ahead of the conv's in
// the same grid, so a round's conv and gating are one kernel. Its gates are bit for bit
// `ninfer::ops::gdn_gating_proj`'s at the 27B's shape, which runs one form at one column and another
// at 2..8 (kernel/src/gdn_conv_split.cu names both).
#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ignis::gdn_conv {

// The gating `split_conv_slots` computes beside the conv: `ninfer::ops::gdn_gating_proj`'s
// contiguous-parent form at the 27B's shape. `h` BF16 [5120, B], the layer's normed input, one
// column per lane; `ab` BF16 [96, 5120] (a in rows 0..47, b in rows 48..95), 16-byte aligned;
// `a_log` and `dt_bias` FP32 [48]; `g` and `beta` FP32 [48, B], written. Every tensor contiguous.
struct Gating {
  const ninfer::Tensor &h;
  const ninfer::Weight &ab;
  const ninfer::Tensor &a_log;
  const ninfer::Tensor &dt_bias;
  ninfer::Tensor &g;
  ninfer::Tensor &beta;
};

// `x` [C, T] BF16, `weight` [C, 4] BF16 (the stored `gdn/convolution`), `conv_state` [C, 3]
// BF16, read as the taps before the chunk and overwritten with the taps after it; `query`,
// `key`, `value` BF16 [Cq, T], [Ck, T], [Cv, T] with Cq + Ck + Cv == C. Every width even and
// every plane 4-byte aligned (the pair form), every tensor contiguous. Throws
// std::invalid_argument on anything else; enqueues on `stream` and does not synchronize.
void split_conv(const ninfer::Tensor &x, const ninfer::Tensor &weight, ninfer::Tensor &conv_state,
                ninfer::Tensor &query, ninfer::Tensor &key, ninfer::Tensor &value, cudaStream_t stream);

// The decode round's form: `x` [C, B] BF16, one column per lane (B lanes, 1..8 on the 27B);
// `weight` [C, 4] BF16; `conv_states` [C, 3, slots] BF16, the pool's whole conv plane; `lane_slots`
// I32 [B] in device memory, lane b's slot, read at run time (so a captured graph serves every
// round), whose taps are read before any store and overwritten with the taps after its column
// (the vendored snapshot form with every lane's initial and base slot the same, the decode
// graph's). The slots must be distinct. `query`, `key`, `value` BF16 [Cq, B], [Ck, B], [Cv, B]
// with Cq + Ck + Cv == C, lane b's channels in column b; each column contiguous, the columns of a
// plane `nb[1]` bytes apart, at least the column's own (so three views of one [C, B] plane are the
// decode graph's copying form's conv plane). Every other tensor contiguous. With `gating`, its
// gates too (GitHub #681), for the same B lanes. Throws std::invalid_argument on anything else;
// enqueues on `stream` and does not synchronize.
void split_conv_slots(const ninfer::Tensor &x, const ninfer::Tensor &weight, ninfer::Tensor &conv_states,
                      const ninfer::Tensor &lane_slots, ninfer::Tensor &query, ninfer::Tensor &key,
                      ninfer::Tensor &value, cudaStream_t stream, const Gating *gating = nullptr);

} // namespace ignis::gdn_conv
