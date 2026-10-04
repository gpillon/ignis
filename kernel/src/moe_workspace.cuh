// ignis kernel leaf: the MoE ops' workspace layout -- OURS (kernel/include/ignis_moe.h).
// Leaf-internal: offsets of every region inside the one workspace buffer the load reserves.
#ifndef IGNIS_MOE_WORKSPACE_CUH
#define IGNIS_MOE_WORKSPACE_CUH

#include "moe_common.cuh"

#include <cstddef>
#include <cstdint>

namespace ignis_moe {

constexpr int kDecodeMaxTokens = IGNIS_MOE_DECODE_MAX_TOKENS;
constexpr int kDecodeMaxUnique = kDecodeMaxTokens * kTopK;  // distinct experts in one decode call
constexpr int kGateUpBlocks = kInter / 128;                  // 5 output blocks of gate (and of up)
constexpr int kDownBlocks = kHidden / 128;                   // 20 output blocks of down
constexpr int kDecodeSplits = 4;                             // k-splits of a gate/up unit (640 each)

inline std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

// Decode route: counters (zero between calls), the gate/up k-split partials and the SwiGLU
// output h of every distinct expert.
struct DecodeCounters {
  uint32_t ticket;
  uint32_t done;
  uint32_t pad[30];
  uint32_t gate_up_arrivals[kDecodeMaxUnique * kGateUpBlocks];
  uint32_t h_ready[kDecodeMaxUnique];
};

struct WorkspaceLayout {
  std::size_t decode_counters = 0;
  std::size_t decode_partials = 0;  // f32 [unique][5][splits][8 tokens][256]
  std::size_t decode_h = 0;         // f32 [unique][8 tokens][640]
  std::size_t prefill = 0;          // start of the prefill route's regions
  std::size_t total = 0;
};

inline WorkspaceLayout workspace_layout(uint32_t /*max_tokens*/) {
  WorkspaceLayout l;
  std::size_t off = 0;
  l.decode_counters = off;
  off = align256(off + sizeof(DecodeCounters));
  l.decode_partials = off;
  off = align256(off + sizeof(float) * kDecodeMaxUnique * kGateUpBlocks * kDecodeSplits * kDecodeMaxTokens * 256);
  l.decode_h = off;
  off = align256(off + sizeof(float) * kDecodeMaxUnique * kDecodeMaxTokens * kInter);
  l.prefill = off;
  l.total = off;
  return l;
}

}  // namespace ignis_moe

#endif  // IGNIS_MOE_WORKSPACE_CUH
