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
#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ignis::gdn_conv {

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
// with Cq + Ck + Cv == C, lane b's channels in column b. Every tensor contiguous. Throws
// std::invalid_argument on anything else; enqueues on `stream` and does not synchronize.
void split_conv_slots(const ninfer::Tensor &x, const ninfer::Tensor &weight, ninfer::Tensor &conv_states,
                      const ninfer::Tensor &lane_slots, ninfer::Tensor &query, ninfer::Tensor &key,
                      ninfer::Tensor &value, cudaStream_t stream);

} // namespace ignis::gdn_conv
