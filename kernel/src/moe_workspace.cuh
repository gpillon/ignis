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

// Decode route: counters and the gate/up fixed-point sums (zero between calls), and the SwiGLU
// output h of every distinct expert.
struct DecodeCounters {
  uint32_t ticket;
  uint32_t done;
  uint32_t pad[30];
  uint32_t gate_up_arrivals[kDecodeMaxUnique * kGateUpBlocks];
  uint32_t h_ready[kDecodeMaxUnique];
};

// Prefill route: tokens are grouped by expert in 64-token chunks of the call.
constexpr int kGroupChunk = 64;  // tokens per grouping chunk
constexpr int kTileRows = 64;    // assignments per GEMM work item

inline uint32_t group_chunks(uint32_t tokens) { return (tokens + kGroupChunk - 1) / kGroupChunk; }

// Work items (expert, first row) of a call: at most one partial tile per expert that has rows.
inline uint32_t max_items(uint32_t tokens) {
  const uint32_t assignments = tokens * kTopK;
  const uint32_t partial = assignments < static_cast<uint32_t>(kExperts) ? assignments : kExperts;
  return assignments / kTileRows + partial;
}

struct WorkspaceLayout {
  std::size_t decode_counters = 0;
  std::size_t decode_gate_up = 0;   // int64 [10 D][D][1280], fixed point (D = decode tokens)
  std::size_t decode_h = 0;         // f32 [10 D][D][640]
  std::size_t chunk_hist = 0;       // u32 [chunks][512]: assignments per expert per chunk, then bases
  std::size_t expert_offset = 0;    // u32 [513]: first sorted row of each expert, then the total
  std::size_t item_count = 0;       // u32 [1]
  std::size_t items = 0;            // int2 [max_items]: (expert, first row within the expert)
  std::size_t sorted = 0;           // i32 [10 T]: assignment (t * 10 + rank) of each sorted row
  std::size_t prefill_h = 0;        // f32 [10 T][640]: each sorted row's SwiGLU output
  std::size_t total = 0;
};

// The layout for a workspace whose decode regions hold `decode_tokens` tokens (the load's lane
// count) and whose prefill regions hold `max_tokens` (the load's maximum prefill chunk).
inline WorkspaceLayout workspace_layout(uint32_t decode_tokens, uint32_t max_tokens) {
  WorkspaceLayout l;
  const std::size_t d = decode_tokens;
  std::size_t off = 0;
  auto take = [&](std::size_t bytes) {
    const std::size_t at = off;
    off = align256(off + bytes);
    return at;
  };
  l.decode_counters = take(sizeof(DecodeCounters));
  l.decode_gate_up = take(sizeof(int64_t) * kTopK * d * d * kGateUpOut);
  l.decode_h = take(sizeof(float) * kTopK * d * d * kInter);
  l.chunk_hist = take(sizeof(uint32_t) * group_chunks(max_tokens) * kExperts);
  l.expert_offset = take(sizeof(uint32_t) * (kExperts + 1));
  l.item_count = take(sizeof(uint32_t));
  l.items = take(sizeof(int2) * max_items(max_tokens));
  l.sorted = take(sizeof(int32_t) * static_cast<std::size_t>(max_tokens) * kTopK);
  l.prefill_h = take(sizeof(float) * static_cast<std::size_t>(max_tokens) * kTopK * kInter);
  l.total = off;
  return l;
}

// 0 when the description is usable, else -1 with the reason recorded for `op`.
inline int32_t check_workspace(const char *op, const ignis_moe_workspace *ws) {
  if (ws == nullptr || ws->base == nullptr) return fail(std::string(op) + ": workspace is NULL");
  if (ws->decode_tokens == 0 || ws->decode_tokens > static_cast<uint32_t>(kDecodeMaxTokens)) {
    return fail(std::string(op) + ": workspace decode_tokens must be 1.." + std::to_string(kDecodeMaxTokens));
  }
  if (ws->prefill_tokens == 0) return fail(std::string(op) + ": workspace prefill_tokens must be at least 1");
  return 0;
}

}  // namespace ignis_moe

#endif  // IGNIS_MOE_WORKSPACE_CUH
