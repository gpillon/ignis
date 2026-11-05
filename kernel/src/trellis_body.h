// ignis kernel leaf - the 27B's trellis body in its layer programs -- OURS
// (spec 27b-16gb/05a, GitHub #339; ADR 0060). The 16 GB artifact (spec
// 27b-16gb/01) carries every body linear and the output head as dense trellis
// matrices, each run by the op of kernel/include/ignis_trellis_linear.h (spec
// 27b-16gb/02); this is what the layer bodies (kernel/src/gdn_layer.cu,
// gqa_layer.cu), the decode and verify rounds (decode_graph.cu), the prefill
// heads (step.cu) and the drafter's head (dflash2_drafter.cu) call in place of
// the vendored NVFP4 ops on a load whose body is trellis.
//
// A load binds one body, never a mix (kernel/src/model.cu): the
// `ninfer::Weight` linears of a trellis load are left empty, and its
// `ignis_trellis_weight`s are what every site below dispatches to.
//
// The activations are the program's: feature-major BF16 `[in, T]`, which is
// the op's token-major `x[T][in]`; an output `[ldy, T]` is its `y` at `ldy`,
// so a call writes its columns into a wider buffer (the GDN conv input) or
// adds into the residual in place. Every call runs on the model's one stream
// against the model's one workspace (`ignis_model::trellis_workspace`, a plan
// line reserved at load), which is what keeps it graph-capturable: the
// workspace's address is fixed for the load and the op leaves its counters
// zero.
//
// Not part of the flat C ABI: leaf-internal, like layer_internal.h.
#pragma once

#include "ignis_model.h"
#include "ignis_trellis_linear.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

struct ignis_model;
struct TrellisTail;

namespace ninfer {
struct Weight;
}

namespace ignis::trellis_body {

// Whether `qtype` (enum ignis_qtype) is one of the trellis codes.
inline bool is_trellis_qtype(int32_t qtype) {
  return qtype >= IGNIS_QTYPE_TRELLIS_MUL1_K2 && qtype <= IGNIS_QTYPE_TRELLIS_MUL1_K6;
}

// A trellis code's 2K (ignis_trellis_weight::k2): K 2, 2.5, 3, 4, 5, 6.
inline uint32_t k2_of(int32_t qtype) {
  switch (qtype) {
  case IGNIS_QTYPE_TRELLIS_MUL1_K2: return 4;
  case IGNIS_QTYPE_TRELLIS_MUL1_K2P5: return 5;
  case IGNIS_QTYPE_TRELLIS_MUL1_K3: return 6;
  case IGNIS_QTYPE_TRELLIS_MUL1_K4: return 8;
  case IGNIS_QTYPE_TRELLIS_MUL1_K5: return 10;
  case IGNIS_QTYPE_TRELLIS_MUL1_K6: return 12;
  default: return 0;
  }
}

// The bytes of a record's tiles: `in * out / 16` words of 2K bits each, the
// `suh` and `svh` planes following them (crates/artifact's `trellis_geometry`).
inline uint64_t tile_bytes(const ignis_trellis_weight &w) {
  return static_cast<uint64_t>(w.in) * w.out / 16 * w.k2;
}

// GitHub #509: whether a body call of `tokens` tokens takes the dense prefill
// (kernel/include/ignis_dense_prefill.h): a load under
// IGNIS_PREFILL_LINEAR_DENSE, and a call wider than the fused op's decode
// route. Every other call, and the head always, is the fused op's. The calls
// below dispatch on it; the ones that read one input with several matrices
// quantize it once for all of them.
bool dense_prefill(const ignis_model &model, int32_t tokens);

// GitHub #511: whether such a call also runs A8 fused into the producers of
// its inputs: the dense prefill, under the leaf's test-only switch
// (ignis_dense_prefill_set_fused_producers, on by default). The RMSnorm
// before a block's first matrices then quantizes their input as it normalizes
// it (the norm_* calls below, which the layer programs run in place of the
// RMSnorm and the projection), and the MLP's SwiGLU epilogue keeps down's
// block maxima (output_and_mlp). The activations are the standalone A8's bit
// for bit; only the passes change.
bool fused_producers(const ignis_model &model, int32_t tokens);

// y = x W through `epilogue` (IGNIS_TRELLIS_STORE or IGNIS_TRELLIS_ADD): `x`
// a BF16 `[in, T]` activation, `y` the first column of a BF16 `[ldy, T]`
// buffer. Throws std::runtime_error with the op's message on a refusal, as the
// vendored ops do, so a layer body's own catch reports it.
void linear(const ignis_model &model, const ignis_trellis_weight &w, const ninfer::Tensor &x, void *y,
            uint32_t ldy, uint32_t epilogue, cudaStream_t stream);

// The GDN layer's input projection: `in_proj_qkv` into `qkv` (`[ldqkv, T]`,
// the conv input) and `in_proj_z` into `z` (`[ldz, T]`), both stores.
void gdn_projection(const ignis_model &model, const ignis_trellis_weight &in_proj_qkv,
                    const ignis_trellis_weight &in_proj_z, const ninfer::Tensor &x, void *qkv, uint32_t ldqkv,
                    void *z, uint32_t ldz, cudaStream_t stream);

// The GQA layer's input projection: the query and the output gate from
// `q_proj` (as `query_gate` below), the key and the value from `k_proj` and
// `v_proj`. On the dense prefill the GEMM's epilogue writes the query and the
// gate straight to their buffers, so the split's copy does not run.
void attention_projection(const ignis_model &model, const ignis_trellis_weight &q_proj,
                          const ignis_trellis_weight &k_proj, const ignis_trellis_weight &v_proj,
                          const ninfer::Tensor &x, int32_t heads, int32_t head_dim, ninfer::DeviceArena &arena,
                          ninfer::Tensor &query, ninfer::Tensor &gate, ninfer::Tensor &key, ninfer::Tensor &value,
                          cudaStream_t stream);

// GitHub #511, under fused_producers: the layer's input RMSnorm (`norm`, BF16
// `[hidden]`, unit offset, the model's eps) of the residual `x` and the input
// projection above, the projection's A8 fused into the norm. The GDN form
// writes the normalized row to `h` (its gating projection reads it); the GQA
// form writes none, as nothing past the projection reads it.
void norm_gdn_projection(const ignis_model &model, const ninfer::Tensor &x, const ninfer::Tensor &norm,
                         ninfer::Tensor &h, const ignis_trellis_weight &in_proj_qkv,
                         const ignis_trellis_weight &in_proj_z, void *qkv, uint32_t ldqkv, void *z, uint32_t ldz,
                         cudaStream_t stream);
void norm_attention_projection(const ignis_model &model, const ninfer::Tensor &x, const ninfer::Tensor &norm,
                               const ignis_trellis_weight &q_proj, const ignis_trellis_weight &k_proj,
                               const ignis_trellis_weight &v_proj, int32_t heads, int32_t head_dim,
                               ninfer::Tensor &query, ninfer::Tensor &gate, ninfer::Tensor &key,
                               ninfer::Tensor &value, cudaStream_t stream);

// h = silu(x G) * (x U): the MLP's gate and up matrices, `h` BF16 `[out, T]`.
void swiglu(const ignis_model &model, const ignis_trellis_weight &gate, const ignis_trellis_weight &up,
            const ninfer::Tensor &x, ninfer::Tensor &h, cudaStream_t stream);

// The GQA layer's query and output gate from `q_proj`, whose rows are
// transformers' (spec 01's As built): `heads` heads of [q `head_dim` | gate
// `head_dim`], interleaved per head. The projection lands in a
// `[2 heads head_dim, T]` buffer taken from `arena`, and `split_query_gate`
// moves each head's halves into `query` and `gate` (`[heads head_dim, T]`
// each, head-major) -- the layout the served artifact's q and gate blocks
// give the attention.
void query_gate(const ignis_model &model, const ignis_trellis_weight &q_proj, const ninfer::Tensor &x,
                int32_t heads, int32_t head_dim, ninfer::DeviceArena &arena, ninfer::Tensor &query,
                ninfer::Tensor &gate, cudaStream_t stream);

// The de-interleave alone: `interleaved` BF16 [tokens][heads][2][head_dim] ->
// `query` and `gate` BF16 [tokens][heads][head_dim], two 2-D copies (a row is
// one head's half; graph-capturable, no kernel of its own).
cudaError_t split_query_gate(const void *interleaved, int32_t heads, int32_t head_dim, int32_t tokens, void *query,
                             void *gate, cudaStream_t stream);

// The MLP intermediate of a layer: its gate matrix's rows on a trellis load,
// the NVFP4 `gate_up` plane's rows / 2 otherwise.
std::int32_t mlp_width(const ignis_model &model, const TrellisTail &tail, const ninfer::Weight &gate_up);

// What every GDN and GQA body runs after its token mixer on a trellis load,
// in this order: the output projection of `mixed` added into `residual`, the
// post-attention norm (`post_norm`, BF16 `[hidden]`) into `post`, the SwiGLU
// into `fused`, and the down projection added into `residual`. Under
// fused_producers (GitHub #511) the norm, the SwiGLU and down run as one dense
// MLP with every A8 fused into its producer, and `post` is not written.
void output_and_mlp(const ignis_model &model, const TrellisTail &tail, const ninfer::Tensor &mixed,
                    const ninfer::Tensor &post_norm, ninfer::Tensor &residual, ninfer::Tensor &post,
                    ninfer::Tensor &fused, cudaStream_t stream);

// The output head over `normalized` (BF16 `[hidden, N]`) into `logits` (BF16
// `[vocab, N]`): the trellis head on a trellis load, `ninfer::ops::linear`
// over the bound `output_head` otherwise -- the one call every head site
// makes, so a site cannot dispatch the other body's head.
void output_head(const ignis_model &model, const ninfer::Tensor &normalized, ninfer::Tensor &logits,
                 cudaStream_t stream);

}  // namespace ignis::trellis_body
