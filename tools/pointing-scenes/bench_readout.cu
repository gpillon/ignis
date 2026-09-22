// Cost of reading the pointing heads (vision study, R4).
//
// A: today     -- one head, the engine's attention_readout_kernel<true> verbatim
//                 (hq plane, query rotated in-kernel), + D2H of key_count floats.
// B: naive R4  -- the same kernel launched once per head: 96 launches over 9 layers,
//                 + D2H of 96 * key_count floats.
// C: fused R4  -- one launch per layer: each warp takes one key row of one KV head
//                 and scores it against every armed query head sharing that KV head;
//                 a per-head argmax (packed float|index, atomicMax) stays on device;
//                 D2H of 96 x 8 bytes + L39.h10's full row (TAG needs its map).
// Each layer gets its own plane buffer (as in the engine, a fresh plane per layer).
// "hot": plane just written (in L2 when it fits, as right after attention);
// "cold": L2 flushed before every layer.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>

constexpr int kHeadDim = 256, kKvHeads = 4, kQHeads = 24, kWarps = 8, kThreads = 256;

__device__ __forceinline__ float sign_of(int t) { return (t * 2654435761u >> 31) ? -1.f : 1.f; }

// ---- A/B: the engine kernel (attention_readout.cu), hq branch ----
__global__ void readout_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *keys, int64_t key_begin,
                               int64_t key_count, float scale, float *scores) {
  __shared__ float q[kHeadDim];
  const int t = threadIdx.x;
  float value = __bfloat162float(query[t]);
  q[t] = value * sign_of(t);
  __syncthreads();
  for (int len = 1; len < kHeadDim; len <<= 1) {
    const float a = q[t], b = q[t ^ len];
    __syncthreads();
    q[t] = (t & len) ? (b - a) : (a + b);
    __syncthreads();
  }
  value = q[t] * (1.f / 16.f);
  __syncthreads();
  q[t] = value;
  __syncthreads();
  const int warp = t / 32, lane = t % 32;
  for (int64_t k = (int64_t)blockIdx.x * kWarps + warp; k < key_count; k += (int64_t)gridDim.x * kWarps) {
    const __nv_bfloat16 *row = keys + (key_begin + k) * kHeadDim;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kHeadDim / 32; ++j) acc += q[lane + 32 * j] * __bfloat162float(row[lane + 32 * j]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) scores[k] = acc * scale;
  }
}

// ---- C: fused per layer ----
// heads[i] = query head index (0..23) armed in this layer, n_heads <= 16.
// block: rotates the armed queries into smem, then each warp scores key rows of
// KV head `kv` (blockIdx.y) against the armed heads mapped to that KV head.
__device__ __forceinline__ unsigned long long pack(float s, uint32_t idx) {
  uint32_t u = __float_as_uint(s);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // order-preserving
  return ((unsigned long long)u << 32) | idx;
}
__global__ void fused_kernel(const __nv_bfloat16 *query, const __nv_bfloat16 *plane, int64_t span, int64_t key_begin,
                             int64_t key_count, const int *heads, int n_heads, float scale,
                             unsigned long long *best, float *full_row, int full_head) {
  __shared__ float q[16][kHeadDim];
  __shared__ int mine[16];
  __shared__ int n_mine;
  const int kv = blockIdx.y, t = threadIdx.x;
  if (t == 0) {
    n_mine = 0;
    for (int i = 0; i < n_heads; ++i)
      if (heads[i] / (kQHeads / kKvHeads) == kv) mine[n_mine++] = i;
  }
  __syncthreads();
  for (int m = 0; m < n_mine; ++m) {  // rotate each armed query (as the engine does)
    float *qq = q[m];
    qq[t] = __bfloat162float(query[heads[mine[m]] * kHeadDim + t]) * sign_of(t);
    __syncthreads();
    for (int len = 1; len < kHeadDim; len <<= 1) {
      const float a = qq[t], b = qq[t ^ len];
      __syncthreads();
      qq[t] = (t & len) ? (b - a) : (a + b);
      __syncthreads();
    }
    qq[t] *= (1.f / 16.f);
    __syncthreads();
  }
  if (n_mine == 0) return;
  __shared__ unsigned long long wbest[kWarps][16];
  if (t < kWarps * 16) wbest[t / 16][t % 16] = 0;
  __syncthreads();
  const __nv_bfloat16 *keys = plane + (int64_t)kv * span * kHeadDim;
  const int warp = t / 32, lane = t % 32;
  for (int64_t k = (int64_t)blockIdx.x * kWarps + warp; k < key_count; k += (int64_t)gridDim.x * kWarps) {
    const __nv_bfloat16 *row = keys + (key_begin + k) * kHeadDim;
    float kr[kHeadDim / 32];
#pragma unroll
    for (int j = 0; j < kHeadDim / 32; ++j) kr[j] = __bfloat162float(row[lane + 32 * j]);
    for (int m = 0; m < n_mine; ++m) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < kHeadDim / 32; ++j) acc += q[m][lane + 32 * j] * kr[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
      if (lane == 0) {
        const float s = acc * scale;
        const unsigned long long p = pack(s, (uint32_t)k);
        if (p > wbest[warp][m]) wbest[warp][m] = p;
        if (heads[mine[m]] == full_head && full_row) full_row[k] = s;
      }
    }
  }
  __syncthreads();
  if (t < n_mine) {
    unsigned long long b = 0;
    for (int w = 0; w < kWarps; ++w) b = wbest[w][t] > b ? wbest[w][t] : b;
    atomicMax(&best[mine[t]], b);
  }
}

__global__ void fill(__nv_bfloat16 *p, int64_t n, uint32_t seed) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    uint32_t x = (uint32_t)i * 2654435761u ^ seed;
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = __float2bfloat16(((x & 0xffff) / 65536.f - 0.5f));
  }
}
__global__ void flush(int *p, int64_t n) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) p[i] += 1;
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
  // the frozen set: heads per layer (L31..L63); L39.h10 is the full row
  const int layers = 9;
  const int per_layer[layers] = {5, 9, 12, 14, 16, 16, 15, 8, 1};
  const int64_t sizes[] = {1024, 4096, 16384};  // image tokens: 1024 px, 2048 px, 4096 px
  const int reps = 30;
  int *dflush; const int64_t flushN = 128ll << 20 >> 2;  // 128 MB > 96 MB L2
  CK(cudaMalloc(&dflush, flushN * 4));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  printf("image_tokens,mode,variant,gpu_ms_per_prefill,d2h_bytes\n");
  for (int64_t n_img : sizes) {
    const int64_t span = n_img + 256, key_begin = 64;
    std::vector<__nv_bfloat16 *> planes(layers);
    for (int l = 0; l < layers; ++l) {
      CK(cudaMalloc(&planes[l], (size_t)kKvHeads * span * kHeadDim * 2));
      fill<<<1024, 256, 0, st>>>(planes[l], (int64_t)kKvHeads * span * kHeadDim, 17 + l);
    }
    __nv_bfloat16 *dq; CK(cudaMalloc(&dq, kQHeads * kHeadDim * 2)); fill<<<24, 256, 0, st>>>(dq, kQHeads * kHeadDim, 5);
    float *dscores; CK(cudaMalloc(&dscores, 96 * n_img * 4));
    float *hscores; CK(cudaMallocHost(&hscores, 96 * n_img * 4));
    int *dheads; CK(cudaMalloc(&dheads, layers * 16 * 4));
    std::vector<int> hh(layers * 16);
    for (int l = 0; l < layers; ++l) for (int i = 0; i < per_layer[l]; ++i) hh[l * 16 + i] = (i * 5 + l) % 24;
    CK(cudaMemcpy(dheads, hh.data(), hh.size() * 4, cudaMemcpyHostToDevice));
    unsigned long long *dbest, *hbest; CK(cudaMalloc(&dbest, layers * 16 * 8)); CK(cudaMallocHost(&hbest, layers * 16 * 8));
    const float scale = 1.f / 16.f;
    const unsigned blocks = (unsigned)std::min<int64_t>((n_img + kWarps - 1) / kWarps, 1024);
    for (int cold = 0; cold < 2; ++cold) {
      for (int variant = 0; variant < 3; ++variant) {
        float total = 0;
        for (int r = 0; r < reps + 3; ++r) {
          float ms_rep = 0;
          size_t d2h = 0;
          for (int l = 0; l < layers; ++l) {
            if (variant == 0 && l != 2) continue;  // today: L39 only
            if (cold) flush<<<1024, 256, 0, st>>>(dflush, flushN);
            else fill<<<256, 256, 0, st>>>(planes[l], (int64_t)kKvHeads * span * kHeadDim, 17 + l);  // "just written"
            if (variant == 2) cudaMemsetAsync(dbest + l * 16, 0, 16 * 8, st);
            cudaEventRecord(e0, st);
            if (variant == 0) {
              readout_kernel<<<blocks, kThreads, 0, st>>>(dq + 10 * kHeadDim, planes[l] + (int64_t)1 * span * kHeadDim, key_begin, n_img, scale, dscores);
            } else if (variant == 1) {
              for (int i = 0; i < per_layer[l]; ++i) {
                const int h = hh[l * 16 + i];
                readout_kernel<<<blocks, kThreads, 0, st>>>(dq + h * kHeadDim, planes[l] + (int64_t)(h / 6) * span * kHeadDim, key_begin, n_img, scale, dscores + (int64_t)(l * 16 + i) % 96 * n_img);
              }
            } else {
              dim3 grid(blocks, kKvHeads);
              fused_kernel<<<grid, kThreads, 0, st>>>(dq, planes[l], span, key_begin, n_img, dheads + l * 16, per_layer[l], scale,
                                                      dbest + l * 16, l == 2 ? dscores : nullptr, l == 2 ? hh[l * 16 + 0] : -1);
            }
            cudaEventRecord(e1, st);
            CK(cudaEventSynchronize(e1));
            float ms; cudaEventElapsedTime(&ms, e0, e1); ms_rep += ms;
          }
          // the one copy at the end of the chunk
          cudaEventRecord(e0, st);
          if (variant == 0) { cudaMemcpyAsync(hscores, dscores, n_img * 4, cudaMemcpyDeviceToHost, st); d2h = n_img * 4; }
          else if (variant == 1) { cudaMemcpyAsync(hscores, dscores, (size_t)96 * n_img * 4, cudaMemcpyDeviceToHost, st); d2h = (size_t)96 * n_img * 4; }
          else { cudaMemcpyAsync(hbest, dbest, layers * 16 * 8, cudaMemcpyDeviceToHost, st);
                 cudaMemcpyAsync(hscores, dscores, n_img * 4, cudaMemcpyDeviceToHost, st); d2h = layers * 16 * 8 + n_img * 4; }
          cudaEventRecord(e1, st);
          CK(cudaEventSynchronize(e1));
          { float ms; cudaEventElapsedTime(&ms, e0, e1); ms_rep += ms; }
          if (r >= 3) total += ms_rep;
          if (r == reps + 2)
            printf("%lld,%s,%s,%.4f,%zu\n", (long long)n_img, cold ? "cold" : "hot",
                   variant == 0 ? "A_today_1head" : variant == 1 ? "B_naive_96launch" : "C_fused_argmax", total / reps, d2h);
        }
      }
    }
    for (auto p : planes) cudaFree(p);
    cudaFree(dq); cudaFree(dscores); cudaFreeHost(hscores); cudaFree(dheads); cudaFree(dbest); cudaFreeHost(hbest);
  }
  return 0;
}
