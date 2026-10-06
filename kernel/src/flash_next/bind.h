// ignis kernel leaf -- the Flash-Next model's weights and their binder (spec
// flash-next/04, GitHub #302; OURS, ADR 0043). S1-private.
//
// The binder matches the descriptors the Rust loader hands across the load
// ABI (crates/artifact/src/flash_next.rs's inventory, placed on the device)
// against the topology-derived schema of docs/specs/flash-next/layout.md
// section 6: every name required, every shape exact, linears either FP8
// row-scale or BF16 (a part the conversion flags is re-converted to BF16),
// everything else BF16, and any descriptor the schema does not name refused.
// The experts are residency's and the n-gram table, its hash buffers and its
// hot rows the host's: none of them crosses as a bound tensor.

#pragma once

#include "flash_next_internal.h"

#include <memory>
#include <string>
#include <vector>

namespace ignis::flash_next {

// One decoder layer's weights.
struct LayerWeights {
  bool attention = false;  // QSA (true) or GDN
  HcWeights attn_hc;
  HcWeights mlp_hc;
  GdnWeights gdn;          // GDN layers
  QsaWeights qsa;          // QSA layers
  MoeWeights moe;
};

// Every non-expert weight of a Flash-Next load.
struct Weights {
  Linear embed;            // embed_tokens [vocab, hidden]
  Linear head;             // lm_head [vocab, hidden]
  HcWeights final_mixer;   // hyper_connection_mixer (no block_inject)
  std::vector<LayerWeights> layers;
  NgramWeights ngram;      // layer g.ngram_layer's PLE
};

// Binds `count` descriptors against `topology`'s schema. Null (and *error
// naming the first fault) on a missing, extra, mis-shaped or mis-formatted
// descriptor, or a topology whose geometry the program does not run.
std::unique_ptr<Weights> bind_flash_next(const ignis_bound_tensor *tensors, uint64_t count,
                                         const ignis_topology &topology, std::string *error);

}  // namespace ignis::flash_next
