// ignis kernel leaf - the 27B's trellis body in its layer programs -- OURS
// (spec 27b-16gb/05a, GitHub #339; GitHub #509 adds the dense prefill). See
// kernel/src/trellis_body.h.

#include "trellis_body.h"

#include "ignis_dense_prefill.h"
#include "model_internal.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::trellis_body {
namespace {

void require_activation(const ninfer::Tensor &x, uint32_t in, const char *op) {
  if (x.dtype != ninfer::DType::BF16 || x.ne[0] != static_cast<std::int32_t>(in) || x.ne[1] <= 0 ||
      x.ne[2] != 1 || x.ne[3] != 1 || !x.is_contiguous()) {
    throw std::invalid_argument(std::string(op) + ": the activation is not a BF16 [" + std::to_string(in) +
                                ", T] matrix");
  }
}

void check(int32_t rc, const char *op) {
  if (rc != 0) {
    throw std::runtime_error(std::string(op) + ": " + ignis_trellis_last_error());
  }
}

// The fused op's linear: every call the dense prefill does not take, the head's among them.
void fused_linear(const ignis_model &model, const ignis_trellis_weight &w, const ninfer::Tensor &x, void *y,
                  uint32_t ldy, uint32_t epilogue, cudaStream_t stream) {
  require_activation(x, w.in, "trellis linear");
  check(ignis_trellis_linear(&w, x.data, static_cast<uint32_t>(x.ne[1]), y, ldy, epilogue,
                             &model.trellis_workspace, stream),
        "trellis linear");
}

ignis_dense_prefill_output dense_output(void *y, uint32_t ldy, uint32_t epilogue) {
  return ignis_dense_prefill_output{y, ldy, epilogue, 0, 0, nullptr};
}

// GitHub #509: the dense prefill of `count` matrices that read `x`, against the
// load's workspace (the INT8 activations) and its one dense buffer.
void dense_linear(const ignis_model &model, const ignis_trellis_weight *const *w, uint32_t count,
                  const ninfer::Tensor &x, const ignis_dense_prefill_output *out, cudaStream_t stream) {
  for (uint32_t i = 0; i < count; ++i) {
    require_activation(x, w[i]->in, "dense prefill");
  }
  if (model.trellis_dense_buffer == nullptr) {
    throw std::runtime_error("dense prefill: the load reserved no dense buffer");
  }
  check(ignis_dense_prefill_linear(w, count, x.data, static_cast<uint32_t>(x.ne[1]), out, &model.trellis_workspace,
                                   model.trellis_dense_buffer->p, model.trellis_dense_buffer->bytes, stream),
        "dense prefill");
}

// GitHub #511: the norm the dense prefill fuses A8 into, a BF16 [hidden] weight.
void require_norm(const ignis_model &model, const ninfer::Tensor &norm, const ninfer::Tensor &x) {
  if (norm.dtype != ninfer::DType::BF16 || norm.ne[0] != x.ne[0] || norm.data == nullptr) {
    throw std::invalid_argument("dense prefill: the RMSnorm's weight is not BF16 [" + std::to_string(x.ne[0]) + "]");
  }
  if (model.trellis_dense_buffer == nullptr) {
    throw std::runtime_error("dense prefill: the load reserved no dense buffer");
  }
}

// GitHub #511: dense_linear with the A8 of `count` matrices fused into the
// RMSnorm of `x` by `norm`, the normalized row written to `h` when non-null.
void norm_linear(const ignis_model &model, const ignis_trellis_weight *const *w, uint32_t count,
                 const ninfer::Tensor &x, const ninfer::Tensor &norm, void *h, const ignis_dense_prefill_output *out,
                 cudaStream_t stream) {
  for (uint32_t i = 0; i < count; ++i) {
    require_activation(x, w[i]->in, "dense prefill");
  }
  require_norm(model, norm, x);
  check(ignis_dense_prefill_norm_linear(w, count, x.data, norm.data, model.rms_norm_eps,
                                        static_cast<uint32_t>(x.ne[1]), h, out, &model.trellis_workspace,
                                        model.trellis_dense_buffer->p, model.trellis_dense_buffer->bytes, stream),
        "dense prefill");
}

// The GQA layer's three outputs on the dense prefill: the query and the gate
// split from `q_proj` by the epilogue (column c of head h lands in the query
// at h head_dim + c, or past head_dim in the gate), the key and the value
// straight into theirs.
struct AttentionOutputs {
  ignis_dense_prefill_output o[3];
};

AttentionOutputs attention_outputs(const ignis_trellis_weight &q_proj, int32_t tokens, int32_t heads,
                                   int32_t head_dim, ninfer::Tensor &query, ninfer::Tensor &gate, ninfer::Tensor &key,
                                   ninfer::Tensor &value) {
  const std::int32_t width = heads * head_dim;
  if (q_proj.out != static_cast<uint32_t>(2 * width) || query.ne[0] != width || gate.ne[0] != width ||
      query.ne[1] != tokens || gate.ne[1] != tokens) {
    throw std::invalid_argument("trellis query/gate: q_proj is not " + std::to_string(heads) + " heads of 2 x " +
                                std::to_string(head_dim) + " rows for this call's query and gate");
  }
  return AttentionOutputs{{ignis_dense_prefill_output{query.data, static_cast<uint32_t>(width), IGNIS_TRELLIS_STORE,
                                                      static_cast<uint32_t>(head_dim), 0, gate.data},
                           dense_output(key.data, static_cast<uint32_t>(key.ne[0]), IGNIS_TRELLIS_STORE),
                           dense_output(value.data, static_cast<uint32_t>(value.ne[0]), IGNIS_TRELLIS_STORE)}};
}

}  // namespace

bool dense_prefill(const ignis_model &model, int32_t tokens) {
  return model.prefill_linear == IGNIS_PREFILL_LINEAR_DENSE && tokens > IGNIS_TRELLIS_DECODE_MAX_TOKENS;
}

bool fused_producers(const ignis_model &model, int32_t tokens) {
  return dense_prefill(model, tokens) && ignis_dense_prefill_fused_producers() != 0;
}

void linear(const ignis_model &model, const ignis_trellis_weight &w, const ninfer::Tensor &x, void *y,
            uint32_t ldy, uint32_t epilogue, cudaStream_t stream) {
  if (dense_prefill(model, x.ne[1])) {
    const ignis_trellis_weight *ws[1] = {&w};
    const ignis_dense_prefill_output out[1] = {dense_output(y, ldy, epilogue)};
    dense_linear(model, ws, 1, x, out, stream);
    return;
  }
  fused_linear(model, w, x, y, ldy, epilogue, stream);
}

void gdn_projection(const ignis_model &model, const ignis_trellis_weight &in_proj_qkv,
                    const ignis_trellis_weight &in_proj_z, const ninfer::Tensor &x, void *qkv, uint32_t ldqkv,
                    void *z, uint32_t ldz, cudaStream_t stream) {
  if (dense_prefill(model, x.ne[1])) {
    const ignis_trellis_weight *ws[2] = {&in_proj_qkv, &in_proj_z};
    const ignis_dense_prefill_output out[2] = {dense_output(qkv, ldqkv, IGNIS_TRELLIS_STORE),
                                               dense_output(z, ldz, IGNIS_TRELLIS_STORE)};
    dense_linear(model, ws, 2, x, out, stream);
    return;
  }
  fused_linear(model, in_proj_qkv, x, qkv, ldqkv, IGNIS_TRELLIS_STORE, stream);
  fused_linear(model, in_proj_z, x, z, ldz, IGNIS_TRELLIS_STORE, stream);
}

void swiglu(const ignis_model &model, const ignis_trellis_weight &gate, const ignis_trellis_weight &up,
            const ninfer::Tensor &x, ninfer::Tensor &h, cudaStream_t stream) {
  require_activation(x, gate.in, "trellis swiglu");
  if (h.dtype != ninfer::DType::BF16 || h.ne[0] != static_cast<std::int32_t>(gate.out) || h.ne[1] != x.ne[1] ||
      !h.is_contiguous()) {
    throw std::invalid_argument("trellis swiglu: the output is not a BF16 [out, T] matrix");
  }
  if (dense_prefill(model, x.ne[1])) {
    // The fused op's SwiGLU too stores the gate in BF16 between its passes.
    if (up.in != gate.in || up.out != gate.out) {
      throw std::invalid_argument("trellis swiglu: gate and up shapes differ");
    }
    const ignis_trellis_weight *ws[2] = {&gate, &up};
    const ignis_dense_prefill_output out[2] = {dense_output(h.data, gate.out, IGNIS_TRELLIS_STORE),
                                               dense_output(h.data, gate.out, IGNIS_DENSE_PREFILL_SWIGLU_UP)};
    dense_linear(model, ws, 2, x, out, stream);
    return;
  }
  check(ignis_trellis_linear_swiglu(&gate, &up, x.data, static_cast<uint32_t>(x.ne[1]), h.data, gate.out,
                                    &model.trellis_workspace, stream),
        "trellis swiglu");
}

cudaError_t split_query_gate(const void *interleaved, int32_t heads, int32_t head_dim, int32_t tokens, void *query,
                             void *gate, cudaStream_t stream) {
  // One row per (token, head): `head_dim` BF16 values, read every 2 head_dim
  // from the interleaved rows -- the q half at offset 0, the gate half after
  // it -- and written densely.
  const std::size_t half = static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
  const std::size_t rows = static_cast<std::size_t>(heads) * static_cast<std::size_t>(tokens);
  const auto *src = static_cast<const std::uint8_t *>(interleaved);
  cudaError_t err = cudaMemcpy2DAsync(query, half, src, 2 * half, half, rows, cudaMemcpyDeviceToDevice, stream);
  if (err == cudaSuccess) {
    err = cudaMemcpy2DAsync(gate, half, src + half, 2 * half, half, rows, cudaMemcpyDeviceToDevice, stream);
  }
  return err;
}

void query_gate(const ignis_model &model, const ignis_trellis_weight &q_proj, const ninfer::Tensor &x,
                int32_t heads, int32_t head_dim, ninfer::DeviceArena &arena, ninfer::Tensor &query,
                ninfer::Tensor &gate, cudaStream_t stream) {
  const std::int32_t width = heads * head_dim;
  const std::int32_t tokens = x.ne[1];
  if (q_proj.out != static_cast<uint32_t>(2 * width) || query.ne[0] != width || gate.ne[0] != width ||
      query.ne[1] != tokens || gate.ne[1] != tokens) {
    throw std::invalid_argument("trellis query/gate: q_proj is not " + std::to_string(heads) + " heads of 2 x " +
                                std::to_string(head_dim) + " rows for this call's query and gate");
  }
  ninfer::Tensor interleaved = arena.alloc(ninfer::DType::BF16, {2 * width, tokens, 1, 1});
  linear(model, q_proj, x, interleaved.data, static_cast<uint32_t>(2 * width), IGNIS_TRELLIS_STORE, stream);
  const cudaError_t err = split_query_gate(interleaved.data, heads, head_dim, tokens, query.data, gate.data, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("trellis query/gate: cudaMemcpy2DAsync failed: ") + cudaGetErrorString(err));
  }
}

void attention_projection(const ignis_model &model, const ignis_trellis_weight &q_proj,
                          const ignis_trellis_weight &k_proj, const ignis_trellis_weight &v_proj,
                          const ninfer::Tensor &x, int32_t heads, int32_t head_dim, ninfer::DeviceArena &arena,
                          ninfer::Tensor &query, ninfer::Tensor &gate, ninfer::Tensor &key, ninfer::Tensor &value,
                          cudaStream_t stream) {
  if (!dense_prefill(model, x.ne[1])) {
    query_gate(model, q_proj, x, heads, head_dim, arena, query, gate, stream);
    fused_linear(model, k_proj, x, key.data, static_cast<uint32_t>(key.ne[0]), IGNIS_TRELLIS_STORE, stream);
    fused_linear(model, v_proj, x, value.data, static_cast<uint32_t>(value.ne[0]), IGNIS_TRELLIS_STORE, stream);
    return;
  }
  const AttentionOutputs out = attention_outputs(q_proj, x.ne[1], heads, head_dim, query, gate, key, value);
  const ignis_trellis_weight *ws[3] = {&q_proj, &k_proj, &v_proj};
  dense_linear(model, ws, 3, x, out.o, stream);
}

void norm_gdn_projection(const ignis_model &model, const ninfer::Tensor &x, const ninfer::Tensor &norm,
                         ninfer::Tensor &h, const ignis_trellis_weight &in_proj_qkv,
                         const ignis_trellis_weight &in_proj_z, void *qkv, uint32_t ldqkv, void *z, uint32_t ldz,
                         cudaStream_t stream) {
  if (h.dtype != ninfer::DType::BF16 || h.ne[0] != x.ne[0] || h.ne[1] != x.ne[1] || !h.is_contiguous()) {
    throw std::invalid_argument("dense prefill: the normalized row is not a BF16 [hidden, T] matrix");
  }
  const ignis_trellis_weight *ws[2] = {&in_proj_qkv, &in_proj_z};
  const ignis_dense_prefill_output out[2] = {dense_output(qkv, ldqkv, IGNIS_TRELLIS_STORE),
                                             dense_output(z, ldz, IGNIS_TRELLIS_STORE)};
  norm_linear(model, ws, 2, x, norm, h.data, out, stream);
}

void norm_attention_projection(const ignis_model &model, const ninfer::Tensor &x, const ninfer::Tensor &norm,
                               const ignis_trellis_weight &q_proj, const ignis_trellis_weight &k_proj,
                               const ignis_trellis_weight &v_proj, int32_t heads, int32_t head_dim,
                               ninfer::Tensor &query, ninfer::Tensor &gate, ninfer::Tensor &key,
                               ninfer::Tensor &value, cudaStream_t stream) {
  const AttentionOutputs out = attention_outputs(q_proj, x.ne[1], heads, head_dim, query, gate, key, value);
  const ignis_trellis_weight *ws[3] = {&q_proj, &k_proj, &v_proj};
  norm_linear(model, ws, 3, x, norm, nullptr, out.o, stream);
}

std::int32_t mlp_width(const ignis_model &model, const TrellisTail &tail, const ninfer::Weight &gate_up) {
  return model.is_trellis_body ? static_cast<std::int32_t>(tail.gate_proj.out) : gate_up.n / 2;
}

void output_and_mlp(const ignis_model &model, const TrellisTail &tail, const ninfer::Tensor &mixed,
                    const ninfer::Tensor &post_norm, ninfer::Tensor &residual, ninfer::Tensor &post,
                    ninfer::Tensor &fused, cudaStream_t stream) {
  const auto hidden = static_cast<uint32_t>(model.hidden);
  linear(model, tail.output, mixed, residual.data, hidden, IGNIS_TRELLIS_ADD, stream);
  if (fused_producers(model, residual.ne[1])) {
    // GitHub #511: the norm with gate and up's A8, gate's store and up's
    // SwiGLU pass into `fused` (`post` unwritten); down's A8 stays the
    // standalone pass of the dense linear below.
    if (tail.up_proj.in != tail.gate_proj.in || tail.up_proj.out != tail.gate_proj.out) {
      throw std::invalid_argument("trellis swiglu: gate and up shapes differ");
    }
    if (fused.dtype != ninfer::DType::BF16 || fused.ne[0] != static_cast<std::int32_t>(tail.gate_proj.out) ||
        fused.ne[1] != residual.ne[1] || !fused.is_contiguous()) {
      throw std::invalid_argument("dense prefill: the SwiGLU's output is not a BF16 [ffn, T] matrix");
    }
    const ignis_trellis_weight *ws[2] = {&tail.gate_proj, &tail.up_proj};
    const ignis_dense_prefill_output out[2] = {dense_output(fused.data, tail.gate_proj.out, IGNIS_TRELLIS_STORE),
                                               dense_output(fused.data, tail.gate_proj.out, IGNIS_DENSE_PREFILL_SWIGLU_UP)};
    norm_linear(model, ws, 2, residual, post_norm, nullptr, out, stream);
  } else {
    ninfer::ops::rmsnorm(residual, post_norm, model.rms_norm_eps, /*unit_offset=*/true, post, stream);
    swiglu(model, tail.gate_proj, tail.up_proj, post, fused, stream);
  }
  linear(model, tail.down, fused, residual.data, hidden, IGNIS_TRELLIS_ADD, stream);
}

void output_head(const ignis_model &model, const ninfer::Tensor &normalized, ninfer::Tensor &logits,
                 cudaStream_t stream) {
  if (!model.is_trellis_body) {
    ninfer::ops::linear(normalized, model.output_head, logits, stream);
    return;
  }
  if (logits.dtype != ninfer::DType::BF16 || logits.ne[0] != static_cast<std::int32_t>(model.trellis_head.out) ||
      logits.ne[1] != normalized.ne[1] || !logits.is_contiguous()) {
    throw std::invalid_argument("trellis output head: the logits are not a BF16 [vocab, N] matrix");
  }
  fused_linear(model, model.trellis_head, normalized, logits.data, model.trellis_head.out, IGNIS_TRELLIS_STORE,
               stream);
}

}  // namespace ignis::trellis_body
