/* ignis kernel leaf: model-load flat C ABI (ADR 0009, GitHub #53).
 *
 * Rust materializes the artifact's text-scope tensors onto its device arena
 * (crates/artifact) and hands the leaf one flat descriptor per bound
 * tensor -- name, qtype, storage layout, device-plane pointers,
 * logical/padded shapes, scale geometry, and the weight/input divisors,
 * mirroring the reference weight descriptor (kernel/vendor/src/core/
 * weight.h `ninfer::Weight`) -- plus one topology descriptor (layer kinds,
 * widths, heads, rotary, vocab, eps). No host activation pointer crosses
 * this boundary; the descriptors only carry device pointers into the
 * artifact's arena (ADR 0009).
 *
 * `ignis_model_load` builds the leaf's per-layer weight structures by
 * matching each bound tensor's name against the topology-derived per-layer
 * schema, and rejects (returns nonzero, sets the last-error message) a
 * missing, extra, or mis-shaped bound tensor -- a load is all-or-nothing,
 * never partial.
 *
 * Rust bindings: crates/core/src/model_load.rs (keep 1:1).
 */
#ifndef IGNIS_MODEL_H
#define IGNIS_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mirrors ninfer::QType (kernel/vendor/src/core/tensor.h) numeric values. */
enum ignis_qtype {
  IGNIS_QTYPE_Q4G64_F16S = 0,
  IGNIS_QTYPE_Q5G64_F16S = 1,
  IGNIS_QTYPE_Q6G64_F16S = 2,
  IGNIS_QTYPE_W8G32_F16S = 3,
  IGNIS_QTYPE_BF16_CTRL = 4,
  IGNIS_QTYPE_FP32_CTRL = 5,
  IGNIS_QTYPE_I32_CTRL = 6,
  IGNIS_QTYPE_NVFP4 = 7,
  IGNIS_QTYPE_FP8_E4M3FN_ROW_BF16S = 8,
};

/* Mirrors ninfer::QuantLayout (kernel/vendor/src/core/tensor.h). */
enum ignis_quant_layout {
  IGNIS_LAYOUT_ROW_SPLIT = 0,
  IGNIS_LAYOUT_CONTIGUOUS = 1,
  IGNIS_LAYOUT_BLOCKSCALE_K16_M128X4 = 2,
  IGNIS_LAYOUT_ROW_SCALE = 3,
};

/* The kind of attention a decoder layer uses (mirrors
 * crates/core/src/compute.rs `LayerKind`). */
enum ignis_layer_kind {
  IGNIS_LAYER_GDN = 0,
  IGNIS_LAYER_GQA = 1,
};

/* One bound tensor crossing the ABI: the artifact directory name (e.g.
 * "text/layers/3/attention/output") plus the reference weight descriptor's
 * device-plane geometry. Device pointers point into the artifact crate's
 * device arena (ADR 0002/0009); a plane that does not apply to this
 * tensor's layout is NULL. */
struct ignis_bound_tensor {
  const char *name; /* NUL-terminated; not owned, valid for the load call */
  int32_t qtype;     /* enum ignis_qtype */
  int32_t layout;    /* enum ignis_quant_layout */
  const void *qdata;   /* the low/code plane (every layout) */
  const void *qhigh;   /* row-split high plane, or NULL */
  const void *scales;  /* scale plane (row-split/blockscale/row-scale), or NULL */
  uint64_t bytes;      /* the layout's exact encoded payload length */
  int32_t shape[4];
  int32_t padded_shape[4];
  uint32_t ndim;
  /* The blockscale layout's trailing FP32 weight divisor, read host-side
   * from the container (not a device pointer: the reference applies it to
   * the group scales as `coeff = e4m3_scale * 1/divisor`, a per-tensor
   * scalar). 0 for a non-blockscale tensor. */
  float weight_scale_divisor;
  /* The paired `*_input_scale_divisor` object's value (the W4A4 activation
   * quant path, G2 -- unread/0 until then). */
  float input_scale_divisor;
};

/* The Qwen 3.8-27B text topology: layer kinds, head/rotary geometry, GDN
 * state widths, vocab, and the RMSNorm epsilon (ADR 0009 -- one source for
 * the leaf's per-layer op sequence and sequence-state geometry, not
 * guesses). */
struct ignis_topology {
  uint32_t num_layers;
  const int32_t *layer_kinds; /* enum ignis_layer_kind[num_layers] */
  uint64_t hidden;
  uint64_t vocab;
  uint64_t num_q_heads;
  uint64_t num_kv_heads;
  uint64_t head_dim;
  uint64_t rotary_dim;
  double rope_theta;
  uint64_t gdn_state_rows;
  uint64_t gdn_state_cols;
  uint64_t gdn_num_layers;
  uint64_t gdn_q_width;
  uint64_t gdn_z_width;
  uint64_t gdn_ab_width;
  uint64_t ffn_intermediate;
  float rms_norm_eps;
};

/* Opaque loaded-model handle. Never dereferenced across the boundary. */
struct ignis_model;

struct ignis_model_stats {
  uint64_t vram_bytes;        /* sum of every bound tensor's payload bytes */
  uint64_t bound_tensor_count;
};

/* Build the leaf's per-layer weight structures from `tensors` (`count`
 * entries) against `topology`. Returns 0 and a handle in `*out_model` on
 * success. Returns -1 (no model produced; see ignis_model_last_error) on a
 * null argument, a duplicate name, a missing or extra bound tensor, or a
 * tensor whose shape does not match the one `topology` implies -- a load
 * is all-or-nothing. */
int32_t ignis_model_load(const struct ignis_bound_tensor *tensors, uint64_t count,
                          const struct ignis_topology *topology, struct ignis_model **out_model);

/* Statistics of a loaded model. Returns 0 on success, -1 on a null
 * argument. */
int32_t ignis_model_stats(const struct ignis_model *model, struct ignis_model_stats *out_stats);

/* Release a model handle. NULL is a no-op. */
void ignis_model_free(struct ignis_model *model);

/* The message from the most recent failing ignis_model_load call on this
 * thread (thread-local; overwritten by the next call; empty string if none
 * failed yet). Never NULL. */
const char *ignis_model_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_MODEL_H */
