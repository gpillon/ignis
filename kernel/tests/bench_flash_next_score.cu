// The QSA indexer's decode score at Flash-Next's geometry (GitHub #306, step 3): a tool, NOT a
// CTest. What a decode call's score costs one lane at a few contexts, three ways:
//   bound    score_kernel<1, 1> on the graph's grid, sized for 262,144 tokens (1,024 CTAs);
//   fitted   the same kernel on a grid sized to the context's blocks (the empty CTAs' share);
//   staged   score_decode_kernel (fusion.h's Score): each key staged in 16-byte loads behind one
//            page lookup, the grid capped at 1,024 CTAs a row and strided.
// Each as 48 calls captured in one graph, replayed; the median replay per call.
//   ignis_kernel_flash_next_score_bench [replays]

#include "flash_next/fusion.h"
#include "flash_next/indexer.h"

#include "moe_fixture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace moe_test;
namespace fn = ignis::flash_next;
namespace ix = ignis::flash_next::indexer;

int main(int argc, char **argv) {
  const int replays = argc > 1 ? std::atoi(argv[1]) : 50;
  constexpr int kCalls = 48;
  constexpr int kHd = 128, kHeads = 4;
  constexpr int32_t kBound = 262144;
  fn::Geometry g;
  g.hidden = 2560;
  g.head_dim = 256;
  g.rotary_dim = 64;
  g.indexer_heads = kHeads;
  g.indexer_head_dim = kHd;
  g.indexer_kv_heads = 1;
  g.compress_ratio = 4;
  g.indexer_budget = 2048;
  g.rms_norm_eps = 1e-6F;
  const int logical = kBound / 64;
  std::vector<int32_t> table(logical);
  for (int i = 0; i < logical; ++i) table[i] = static_cast<int32_t>((static_cast<int64_t>(i) * 7919) % logical);
  std::vector<uint16_t> keys(static_cast<std::size_t>(logical) * ix::kBlocksPerPage * kHd);
  for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = f32_to_bf16(hash_uniform(0x5C0, i, 1.0F));
  std::vector<uint16_t> q(kHeads * kHd);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = f32_to_bf16(hash_uniform(0x5C1, i, 1.0F));
  DeviceBytes d_table(table.size() * 4), d_keys(keys.size() * 2), d_q(q.size() * 2), d_slot(4), d_pos(4);
  DeviceBytes d_scores(static_cast<std::size_t>(kBound / 4) * 4);
  upload(d_table, table);
  upload(d_keys, keys);
  upload(d_q, q);
  upload(d_slot, std::vector<int32_t>{0});
  ix::Paged paged;
  paged.block_tables = d_table.as<int32_t>();
  paged.logical_pages = logical;
  paged.block_keys = d_keys.as<__nv_bfloat16>();
  fn::Batch b;
  b.lanes = 1;
  b.tokens = 1;
  b.slots = d_slot.as<int32_t>();
  b.positions = d_pos.as<int32_t>();
  b.max_visible = kBound;
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t t0, t1;
  MOE_CUDA(cudaEventCreate(&t0));
  MOE_CUDA(cudaEventCreate(&t1));
  std::printf("indexer decode score, one lane, %d calls a graph, median of %d replays (us a call)\n", kCalls, replays);
  std::printf("%10s %8s %8s %8s %8s\n", "position", "blocks", "bound", "fitted", "staged");
  for (const int32_t position : {2050, 8194, 32770, 131074, 262142}) {
    upload(d_pos, std::vector<int32_t>{position});
    const int blocks = (position + 1) / 4;
    double us[3] = {};
    for (int variant = 0; variant < 3; ++variant) {
      fn::set_fused(fn::Fusion::Score, variant == 2);
      const int max_blocks = variant == 1 ? blocks : kBound / 4;
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      for (int c = 0; c < kCalls; ++c) {
        if (ix::score(g, paged, b, 0, 1, d_q.as<__nv_bfloat16>(), max_blocks, d_scores.as<float>(), kBound / 4,
                      stream) != nullptr) {
          std::fprintf(stderr, "score failed\n");
          return 1;
        }
      }
      MOE_CUDA(cudaStreamEndCapture(stream, &graph));
      MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
      std::vector<float> times;
      for (int r = 0; r < replays + 3; ++r) {
        MOE_CUDA(cudaEventRecord(t0, stream));
        MOE_CUDA(cudaGraphLaunch(exec, stream));
        MOE_CUDA(cudaEventRecord(t1, stream));
        MOE_CUDA(cudaEventSynchronize(t1));
        float ms = 0.0F;
        MOE_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        if (r >= 3) times.push_back(ms);
      }
      std::sort(times.begin(), times.end());
      us[variant] = times[times.size() / 2] * 1000.0 / kCalls;
      MOE_CUDA(cudaGraphExecDestroy(exec));
      MOE_CUDA(cudaGraphDestroy(graph));
    }
    std::printf("%10d %8d %8.2f %8.2f %8.2f\n", position, blocks, us[0], us[1], us[2]);
  }
  return 0;
}
