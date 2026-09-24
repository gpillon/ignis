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

/* The speculative backend a load selects (P5-02, GitHub #150, spec 05).
 * Speculation is engine residency, fixed for the life of the load. */
enum ignis_speculative_backend {
  IGNIS_SPECULATIVE_NONE = 0,
  /* The 5-layer sliding-window DFlash2 drafter: binds the 66 `dflash2/*`
   * objects; a sequence pool built with it carries the drafter's window per
   * slot (ignis_seq_pool_spec::speculative_backend). */
  IGNIS_SPECULATIVE_DFLASH2 = 1,
  /* The verify substrate alone (P5-04, GitHub #153): the verify round, its
   * ReplaySSM records and its graphs at the load's window, with no drafter
   * bound and no window pool. The drafts come through
   * `ignis_decode_options::drafts` -- the internal seam a test's fake
   * drafter fills, and what proves accept, rollback and fold before the real
   * drafter is wired in (P5-05). Not an operator-facing backend. */
  IGNIS_SPECULATIVE_VERIFY_ONLY = 2,
};

/* The widest DFlash2 draft window a load accepts. */
#define IGNIS_DFLASH2_MAX_DRAFT_TOKENS 7

/* Load options (ADR 0016: `size` first, `sizeof` the struct the caller
 * compiled against; a NULL pointer means the production defaults -- no
 * speculation). */
struct ignis_model_load_options {
  uint32_t size;
  int32_t speculative_backend; /* enum ignis_speculative_backend */
  /* The draft window: 1..IGNIS_DFLASH2_MAX_DRAFT_TOKENS under DFLASH2 and
   * VERIFY_ONLY, 0 with no backend. Fixed for the life of the load: it is
   * the window the verify graphs are captured at, and the only
   * `speculative_window` `ignis_program_decode` accepts besides 0. */
  uint32_t draft_tokens;
  /* The vision envelope in merged vision tokens per request (GitHub #177);
   * 0 = no vision. Nonzero binds every `vision/*` tensor and reserves, at
   * load, the encoder workspace for V = min(max_context_tokens,
   * vision_max_tokens) and the embedding pool below -- before the caller
   * builds its sequence pool. The workspace is not an arena of
   * its own (GitHub #212): the load's scratch arena is sized for the larger
   * of a prefill chunk's scratch and the encoder's workspace, and media
   * encode runs out of it between prefill steps. At most
   * IGNIS_VISION_MAX_TOKENS_LIMIT. Independent of the backend above
   * (GitHub #195): a load may ask for both, and then the verify round
   * rotates its columns at `position + rope_delta` the way a decode round
   * already does. */
  uint32_t vision_max_tokens;
  /* GitHub #243: bytes reserved for the media embedding pool, carved into
   * fixed-width column pages (IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS columns of
   * `[hidden]` BF16 each). An item takes the pages its own columns need, so a
   * pool holds as many embeddings as fit rather than a fixed count: at 5120
   * hidden a page is 1,280 KiB, a 320x240 thumbnail is one page and a
   * 4096x4096 screenshot is 128. 0 with `vision_max_tokens` 0; otherwise at
   * least the envelope's own output (one item must always fit once everything
   * else is released), and `ignis_media_encode` refuses with
   * IGNIS_MEDIA_ENCODE_POOL_FULL while live embeddings hold the rest. This is
   * where the reference keeps a single output transient: the pool is a
   * departure (ADR 0035), because an embedding outliving its encode is what
   * lets a fan-out over one image encode it once. */
  uint64_t vision_embedding_pool_bytes;
  /* GitHub #227: YaRN RoPE scaling, the text frequency table this load
   * rotates at. 0 (or 1) is no scaling -- the linear table the engine has
   * always used, whose `attention_factor` of 1 keeps `ops::rope`'s exact
   * legacy FP32 angle route -- and a factor in (1, 64] builds the YaRN table
   * over the checkpoint's 262,144-position trained envelope, which is what
   * lets a context past that envelope mean anything. The three below are the
   * ramp's and are read only with a factor: `temperature` scales the q-side
   * attention factor (`temperature * ln(factor) + 1`), `beta_fast` /
   * `beta_slow` place the blend band. Zero-filling them with no factor is
   * accepted; with a factor they must be positive and finite, and
   * beta_fast > beta_slow. Text only: the DFlash2 drafter keeps its own
   * unscaled table at window-local positions, and vision its 2-D one. */
  float rope_scaling_factor;
  float rope_scaling_temperature;
  float rope_scaling_beta_fast;
  float rope_scaling_beta_slow;
};

/* The widest vision envelope a load accepts, in merged tokens: 4x that many
 * raw patches must fit the encoder's int32 extents with room to spare. */
#define IGNIS_VISION_MAX_TOKENS_LIMIT (1u << 20)

/* Every device reservation a load makes beside the weights, in bytes, one
 * field per VRAM plan line (GitHub #210, ADR 0030). Filled by
 * `ignis_model_plan_reservations` before a load and by `ignis_model_stats`
 * from what a load actually holds, so a caller can plan with the first and
 * check the second. */
struct ignis_model_reservations {
  /* The one scratch arena prefill chunks and media encode share (GitHub
   * #212): the program scratch for one `prefill_chunk_tokens` chunk (with
   * the drafter's context append under DFLASH2), or with vision the encoder
   * workspace when that is larger. The two are never live at once. */
  uint64_t workspace_bytes;
  /* The media embedding pool (GitHub #243): the load's
   * `vision_embedding_pool_bytes`, rounded up to a whole number of column
   * pages. 0 without vision. Holds as many items as their own columns fit,
   * not a fixed count -- see `ignis_model_load_options`. */
  uint64_t media_embedding_bytes;
  /* Device sampling's staging buffers and candidate-selection workspace. */
  uint64_t sampling_bytes;
  /* The decode round's scratch arena and staging (token ids, slots, and
   * with vision the rope positions). */
  uint64_t decode_graph_bytes;
  /* The verify round's staging, replay records and accept workspace; 0
   * without a draft window. */
  uint64_t verify_round_bytes;
  /* The DFlash2 drafter's feature taps, append counts and round scratch; 0
   * without the drafter. */
  uint64_t drafter_round_bytes;
};

struct ignis_model_stats {
  uint64_t vram_bytes;        /* sum of every bound tensor's payload bytes */
  uint64_t bound_tensor_count;
  /* What vision reserves beside the weights (GitHub #177, #212): the
   * per-item output transient, plus whatever the encoder workspace grows
   * the shared scratch arena past a prefill chunk's own scratch. 0 without
   * vision. */
  uint64_t vision_reserved_bytes;
  /* What this load holds beside the weights, read off its own buffers
   * (GitHub #210). */
  struct ignis_model_reservations reserved;
};

/* What `ignis_model_load` would reserve beside the weights for the same
 * arguments, without allocating anything (GitHub #210): the argument checks
 * and tensor binding run as they would for the load, so an invalid call
 * fails here with the load's own message. Only the descriptors of `tensors`
 * are read -- name, qtype, layout, shape, bytes -- never their data, so a
 * caller can plan before the weights are on the device and pass null
 * pointers for `qdata`, `qhigh` and `scales`.
 *
 * Returns 0 and fills `*out` on success, -1 otherwise (see
 * ignis_model_last_error). */
int32_t ignis_model_plan_reservations(const struct ignis_bound_tensor *tensors, uint64_t count,
                                      const struct ignis_topology *topology,
                                      uint32_t prefill_chunk_tokens, uint32_t max_context_tokens,
                                      int32_t kv_format,
                                      const struct ignis_model_load_options *options,
                                      struct ignis_model_reservations *out);

/* Build the leaf's per-layer weight structures from `tensors` (`count`
 * entries) against `topology`. `prefill_chunk_tokens` is the widest prefill
 * chunk the caller will ever hand `ignis_program_prefill` (P2-01, GitHub
 * #83): the load reserves the program's scratch arena once, sized for a
 * chunk of that width across every decoder layer's transient allocations
 * plus each vendored op's own workspace-capacity query over the token
 * interval [1, prefill_chunk_tokens], under the widest compute policy each
 * weight's own qtype admits (AllowA4 for NVFP4, A16Only for the real
 * artifact's few BF16 exception arms) -- so enabling AllowA4 for a weight
 * that already admits it changes no reservation. Must be a nonzero multiple
 * of 128. `max_context_tokens` is
 * the largest single-sequence KV reservation the caller's sequence-state
 * pool will ever build (mirrors `ignis_seq_pool_spec::max_context_tokens`,
 * ignis_seq.h): the reservation sizes the GQA attention workspace for the
 * worst-case visible-key count. Must be positive and at least
 * `prefill_chunk_tokens` (a chunk wider than the sequence pool's own
 * context bound could never be prefilled anyway).
 *
 * `kv_format` (one of `enum ignis_kv_format`, ignis_seq.h) is the KV
 * storage format every sequence pool used with this handle will be built in
 * (P4-05, GitHub #123). It is a load argument rather than something read off
 * the pool because the GQA attention workspace is part of the scratch
 * reservation above and its size depends on the format: the hq-e8-2b prompt
 * route materializes the envelope's visible history into two rotated-frame
 * BF16 scratch planes, which BF16's own prompt route has no counterpart for.
 * A pool whose format differs from this argument is refused by the layer
 * entry points rather than run against an arena sized for the other format.
 *
 * `options` (NULL = no speculation) selects a speculative backend and its
 * draft window (P5-02, GitHub #150). Under IGNIS_SPECULATIVE_DFLASH2 the
 * `dflash2/*` tensors must be among `tensors` -- without the option they are
 * extra bound tensors like any other -- and the prefill scratch grows by the
 * drafter's context append over a chunk (the feature taps and their
 * projection). The drafter's window is per-sequence state, so it lives in
 * the sequence pool (`ignis_seq_pool_spec::speculative_backend`, P5-03,
 * GitHub #152), which must be built with the same backend; both are reported
 * by `ignis_program_stats`.
 *
 * Returns 0 and a handle in `*out_model` on success. Returns -1 (no model
 * produced; see ignis_model_last_error) on a null argument, a duplicate
 * name, a missing or extra bound tensor, a tensor whose shape does not
 * match the one `topology` implies, an invalid `prefill_chunk_tokens` /
 * `max_context_tokens` / `kv_format` / `options`, or a chunk width whose
 * scratch reservation does not fit the device's free memory -- a load is
 * all-or-nothing. */
int32_t ignis_model_load(const struct ignis_bound_tensor *tensors, uint64_t count,
                          const struct ignis_topology *topology, uint32_t prefill_chunk_tokens,
                          uint32_t max_context_tokens, int32_t kv_format,
                          const struct ignis_model_load_options *options,
                          struct ignis_model **out_model);

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
