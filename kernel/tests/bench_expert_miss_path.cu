// The expert residency miss path, measured -- OURS, and NOT a CTest test (a
// timing is a finding, not a pass/fail; docs/agents/testing.md).
//
// Spec flash-next/03 (GitHub #301) picks its miss path from this table: an
// expert projection that is not resident in the VRAM expert cache is copied
// from the pinned host expert pool either
//
//   ce   by the copy engine (cudaMemcpyAsync / cudaMemcpyBatchAsync), which
//        the host must issue once it knows the miss: host-orchestrated
//        residency, one selection readback per layer;
//   sm   by a kernel whose threads load from mapped pinned memory and store
//        into the slot, which a device-side LRU can launch without the host:
//        device-resident residency, graph-capturable.
//
// The spec's rule: device-resident only if `sm` reaches >= 80% of `ce`'s
// bandwidth on the transfers residency actually makes, 1-3 MB each.
//
// Every cell copies N projections of S bytes from random 4 KiB-aligned
// places in a large pinned pool (the host expert pool's stand-in) into N
// slots. Each rep uses a different list, so no rep re-reads what L2 may
// have kept from the last one. Times are device time between events unless
// a column says "host": those wrap the call in a host clock and include
// WDDM's submission cost, which is what a per-layer round trip pays.
//
//   ignis_expert_miss_path_bench [--host-gib 4] [--reps 15] [--quick]
//
// Self-contained (CUDA runtime only), so it also builds with plain nvcc:
//   nvcc -O3 -arch=sm_120a -o bench bench_expert_miss_path.cu

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

#define CUDA_OK(expr)                                                                              \
    do {                                                                                           \
        const cudaError_t status_ = (expr);                                                        \
        if (status_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr,                      \
                         cudaGetErrorString(status_));                                             \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr size_t KIB = 1024;
constexpr size_t MIB = 1024 * KIB;
constexpr size_t GIB = 1024 * MIB;
constexpr size_t ALIGN = 4 * KIB;

struct Job {
    const uint4* src;
    uint4* dst;
};

// One copy of `njobs` equal projections, flattened into 16-byte vectors and
// walked grid-stride so a warp's lanes read consecutive vectors of one
// projection (one 512-byte request per warp). UNROLL loads are issued before
// any store: over PCIe a load is ~1-2 us away, and only many loads in flight
// per SM keep the link busy.
template <int UNROLL>
__global__ void copy_jobs(const Job* __restrict__ jobs, int njobs, unsigned vecs_per_job) {
    const unsigned long long total = static_cast<unsigned long long>(vecs_per_job) * njobs;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long base = static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
                                   threadIdx.x;
         base < total; base += stride * UNROLL) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned long long k = base + u * stride;
            if (k < total) {
                const Job job = jobs[k / vecs_per_job];
                v[u] = job.src[k % vecs_per_job];
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const unsigned long long k = base + u * stride;
            if (k < total) {
                const Job job = jobs[k / vecs_per_job];
                job.dst[k % vecs_per_job] = v[u];
            }
        }
    }
}

__global__ void touch(int* flag) { *flag += 1; }

struct SmConfig {
    int blocks;
    int threads;
    int unroll;
};

void launch_sm(const SmConfig& c, const Job* jobs, int n, size_t bytes, cudaStream_t s) {
    const unsigned vecs = static_cast<unsigned>(bytes / sizeof(uint4));
    switch (c.unroll) {
    case 1: copy_jobs<1><<<c.blocks, c.threads, 0, s>>>(jobs, n, vecs); break;
    case 4: copy_jobs<4><<<c.blocks, c.threads, 0, s>>>(jobs, n, vecs); break;
    case 8: copy_jobs<8><<<c.blocks, c.threads, 0, s>>>(jobs, n, vecs); break;
    default: std::fprintf(stderr, "unroll %d not built\n", c.unroll); std::exit(1);
    }
    CUDA_OK(cudaGetLastError());
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Pools {
    char* host = nullptr;          // pinned, mapped: the host expert pool's stand-in
    char* host_dev = nullptr;      // the same memory as the device sees it
    size_t host_bytes = 0;
    char* host_wc = nullptr;       // pinned, mapped, write-combined
    char* host_wc_dev = nullptr;
    size_t host_wc_bytes = 0;
    char* slots = nullptr;         // device: the slot pool's stand-in
    size_t slot_bytes = 0;
};

// `lists` job lists of n copies of `bytes` each: random sources in the host
// pool (two may overlap, which costs a copy nothing), distinct slots on the
// device.
struct Lists {
    int n = 0;
    size_t bytes = 0;
    std::vector<std::vector<size_t>> src;  // offsets into the host pool
    std::vector<std::vector<size_t>> dst;  // offsets into the slot pool
    Job* device_jobs = nullptr;            // [lists][n], host pool (plain)
    Job* device_jobs_wc = nullptr;         // [lists][n], write-combined pool
};

Lists make_lists(const Pools& p, int n, size_t bytes, int lists, std::mt19937_64& rng,
                 bool want_wc) {
    Lists l;
    l.n = n;
    l.bytes = bytes;
    const size_t span = (bytes + ALIGN - 1) / ALIGN * ALIGN;
    const size_t host_places = (p.host_bytes - span) / ALIGN;
    const size_t wc_places = want_wc ? (p.host_wc_bytes - span) / ALIGN : 0;
    const size_t slot_places = p.slot_bytes / span;
    if (static_cast<size_t>(n) > slot_places) {
        std::fprintf(stderr, "slot pool too small for %d x %zu\n", n, bytes);
        std::exit(1);
    }
    std::vector<Job> jobs(static_cast<size_t>(lists) * n);
    std::vector<Job> jobs_wc(static_cast<size_t>(lists) * n);
    std::vector<size_t> slot_order(slot_places);
    for (size_t i = 0; i < slot_places; ++i) { slot_order[i] = i; }
    for (int li = 0; li < lists; ++li) {
        std::vector<size_t> src, dst;
        std::shuffle(slot_order.begin(), slot_order.end(), rng);
        for (int j = 0; j < n; ++j) {
            const size_t s = (rng() % host_places) * ALIGN;
            const size_t d = slot_order[j] * span;
            src.push_back(s);
            dst.push_back(d);
            jobs[static_cast<size_t>(li) * n + j] = {
                reinterpret_cast<const uint4*>(p.host_dev + s),
                reinterpret_cast<uint4*>(p.slots + d)};
            if (want_wc) {
                const size_t sw = (rng() % wc_places) * ALIGN;
                jobs_wc[static_cast<size_t>(li) * n + j] = {
                    reinterpret_cast<const uint4*>(p.host_wc_dev + sw),
                    reinterpret_cast<uint4*>(p.slots + d)};
            }
        }
        l.src.push_back(src);
        l.dst.push_back(dst);
    }
    CUDA_OK(cudaMalloc(&l.device_jobs, jobs.size() * sizeof(Job)));
    CUDA_OK(cudaMemcpy(l.device_jobs, jobs.data(), jobs.size() * sizeof(Job),
                       cudaMemcpyHostToDevice));
    if (want_wc) {
        CUDA_OK(cudaMalloc(&l.device_jobs_wc, jobs_wc.size() * sizeof(Job)));
        CUDA_OK(cudaMemcpy(l.device_jobs_wc, jobs_wc.data(), jobs_wc.size() * sizeof(Job),
                           cudaMemcpyHostToDevice));
    }
    return l;
}

void free_lists(Lists& l) {
    CUDA_OK(cudaFree(l.device_jobs));
    if (l.device_jobs_wc) { CUDA_OK(cudaFree(l.device_jobs_wc)); }
}

struct Bench {
    Pools p;
    int reps = 25;
    cudaStream_t s[4] = {};
    cudaEvent_t start = nullptr, stop = nullptr, fork = nullptr, join[4] = {};

    // Device ms for `body`, median over reps (rep r uses list r).
    template <class Body>
    double device_ms(Body&& body) {
        for (int w = 0; w < 2; ++w) { body(w % reps); }
        CUDA_OK(cudaStreamSynchronize(s[0]));
        std::vector<double> t;
        for (int r = 0; r < reps; ++r) {
            CUDA_OK(cudaEventRecord(start, s[0]));
            body(r);
            CUDA_OK(cudaEventRecord(stop, s[0]));
            CUDA_OK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
            t.push_back(ms);
        }
        return median(t);
    }

    // Host ms for `body` followed by a stream sync, median over reps.
    template <class Body>
    double host_ms(Body&& body) {
        for (int w = 0; w < 2; ++w) { body(w % reps); CUDA_OK(cudaStreamSynchronize(s[0])); }
        std::vector<double> t;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            body(r);
            CUDA_OK(cudaStreamSynchronize(s[0]));
            const auto t1 = std::chrono::steady_clock::now();
            t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return median(t);
    }

    // The n copies of list r over `streams` streams, forked from and joined
    // back into s[0], so s[0]'s events bracket all of them.
    void ce_streams(const Lists& l, int r, int streams) {
        if (streams > 1) {
            CUDA_OK(cudaEventRecord(fork, s[0]));
            for (int k = 1; k < streams; ++k) { CUDA_OK(cudaStreamWaitEvent(s[k], fork, 0)); }
        }
        for (int j = 0; j < l.n; ++j) {
            CUDA_OK(cudaMemcpyAsync(p.slots + l.dst[r][j], p.host + l.src[r][j], l.bytes,
                                    cudaMemcpyHostToDevice, s[j % streams]));
        }
        if (streams > 1) {
            for (int k = 1; k < streams; ++k) {
                CUDA_OK(cudaEventRecord(join[k], s[k]));
                CUDA_OK(cudaStreamWaitEvent(s[0], join[k], 0));
            }
        }
    }

    void ce_batch(const Lists& l, int r) {
        std::vector<void*> dsts(l.n);
        std::vector<const void*> srcs(l.n);
        std::vector<size_t> sizes(l.n, l.bytes);
        for (int j = 0; j < l.n; ++j) {
            dsts[j] = p.slots + l.dst[r][j];
            srcs[j] = p.host + l.src[r][j];
        }
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attr_idx = 0;
        CUDA_OK(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), l.n, &attr,
                                     &attr_idx, 1, s[0]));
    }

    void sm(const Lists& l, int r, const SmConfig& c, bool wc) {
        const Job* jobs = (wc ? l.device_jobs_wc : l.device_jobs) + static_cast<size_t>(r) * l.n;
        launch_sm(c, jobs, l.n, l.bytes, s[0]);
    }
};

double gbps(size_t bytes, double ms) { return static_cast<double>(bytes) / (ms * 1e-3) / 1e9; }

std::vector<SmConfig> sm_configs(int sm_count, bool quick) {
    std::vector<SmConfig> out;
    const std::vector<int> blocks = quick
        ? std::vector<int>{8, 16, sm_count, 2 * sm_count}
        : std::vector<int>{8, 16, 32, 64, sm_count, 2 * sm_count, 4 * sm_count};
    for (int b : blocks) {
        for (int t : {256, 512}) {
            for (int u : {1, 4, 8}) { out.push_back({b, t, u}); }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    size_t host_gib = 4;
    int reps = 15;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--host-gib" && i + 1 < argc) { host_gib = std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--reps" && i + 1 < argc) { reps = std::atoi(argv[++i]); }
        else if (a == "--quick") { quick = true; }
        else { std::fprintf(stderr, "usage: %s [--host-gib 4] [--reps 15] [--quick]\n", argv[0]); return 2; }
    }
    if (reps < 1 || host_gib < 1) {
        std::fprintf(stderr, "--reps and --host-gib must be at least 1\n");
        return 2;
    }

    int dev = 0;
    CUDA_OK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    int tcc = 0;
    CUDA_OK(cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, dev));
    int copy_engines = prop.asyncEngineCount;
    std::printf("# expert miss path: %s, %d SMs, L2 %d MiB, %s driver, %d async copy engines\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize / static_cast<int>(MIB),
                tcc ? "TCC" : "WDDM", copy_engines);
    std::printf("# host pool %zu GiB pinned+mapped, write-combined pool 1 GiB, reps %d (median)\n\n",
                host_gib, reps);

    Bench b;
    b.reps = reps;
    b.p.host_bytes = host_gib * GIB;
    const auto alloc_t0 = std::chrono::steady_clock::now();
    CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&b.p.host), b.p.host_bytes,
                          cudaHostAllocMapped | cudaHostAllocPortable));
    const auto alloc_t1 = std::chrono::steady_clock::now();
    CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.p.host_dev), b.p.host, 0));
    b.p.host_wc_bytes = 1 * GIB;
    CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&b.p.host_wc), b.p.host_wc_bytes,
                          cudaHostAllocMapped | cudaHostAllocPortable |
                              cudaHostAllocWriteCombined));
    CUDA_OK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.p.host_wc_dev), b.p.host_wc, 0));
    std::printf("# cudaHostAlloc of %zu GiB took %.0f ms\n\n", host_gib,
                std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count());
    // A recognisable pattern, so the correctness check below means something.
    for (size_t i = 0; i < b.p.host_bytes / sizeof(uint64_t); ++i) {
        reinterpret_cast<uint64_t*>(b.p.host)[i] = i * 0x9E3779B97F4A7C15ull;
    }
    for (size_t i = 0; i < b.p.host_wc_bytes / sizeof(uint64_t); ++i) {
        reinterpret_cast<uint64_t*>(b.p.host_wc)[i] = i * 0x9E3779B97F4A7C15ull;
    }
    b.p.slot_bytes = 64 * 3 * MIB + 64 * ALIGN;
    CUDA_OK(cudaMalloc(&b.p.slots, b.p.slot_bytes));
    for (auto& st : b.s) { CUDA_OK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking)); }
    CUDA_OK(cudaEventCreate(&b.start));
    CUDA_OK(cudaEventCreate(&b.stop));
    CUDA_OK(cudaEventCreateWithFlags(&b.fork, cudaEventDisableTiming));
    for (auto& e : b.join) { CUDA_OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
    std::mt19937_64 rng(301);

    // Spec 01's layout: gate/up 2560x1280 and down 640x2560 at K bits per
    // weight plus fp16 channel scales, 4 KiB-aligned. The eight class sizes
    // span 0.42-1.65 MB; 3 MiB is the spec's upper end.
    const std::vector<size_t> sizes = {512 * KIB, 1 * MIB, 1536 * KIB, 2 * MIB, 3 * MIB};
    // 1: one miss. 20: a decode layer's ten experts, both projections. 32: a
    // prefetch of W = 16 experts. 64: three lanes' worth of misses.
    const std::vector<int> batches = {1, 20, 32, 64};

    // 1. The copy engine.
    std::printf("## 1. copy engine, pinned host -> device (GB/s, device time)\n\n");
    std::printf("| bytes | n | 1 stream | 2 streams | 4 streams | cudaMemcpyBatchAsync | host: 1 stream + sync |\n");
    std::printf("|---:|---:|---:|---:|---:|---:|---:|\n");
    std::vector<std::vector<double>> ce_best(sizes.size(), std::vector<double>(batches.size()));
    for (size_t si = 0; si < sizes.size(); ++si) {
        for (size_t bi = 0; bi < batches.size(); ++bi) {
            Lists l = make_lists(b.p, batches[bi], sizes[si], reps, rng, false);
            const size_t moved = l.bytes * l.n;
            const double one = gbps(moved, b.device_ms([&](int r) { b.ce_streams(l, r, 1); }));
            const double two = gbps(moved, b.device_ms([&](int r) { b.ce_streams(l, r, 2); }));
            const double four = gbps(moved, b.device_ms([&](int r) { b.ce_streams(l, r, 4); }));
            const double batch = gbps(moved, b.device_ms([&](int r) { b.ce_batch(l, r); }));
            const double host = gbps(moved, b.host_ms([&](int r) { b.ce_streams(l, r, 1); }));
            ce_best[si][bi] = std::max({one, two, four, batch});
            std::printf("| %zu | %d | %.2f | %.2f | %.2f | %.2f | %.2f |\n", sizes[si], l.n, one,
                        two, four, batch, host);
            free_lists(l);
        }
    }

    // 2. SM-driven copies across grid shapes, on the decode layer's shape.
    const auto configs = sm_configs(prop.multiProcessorCount, quick);
    {
        std::printf("\n## 2. SM-driven copy from mapped pinned memory, n = 20 x 1 MiB, by grid (GB/s)\n\n");
        std::printf("| blocks | threads | unroll | GB/s |\n|---:|---:|---:|---:|\n");
        Lists l = make_lists(b.p, 20, 1 * MIB, reps, rng, false);
        for (const auto& c : configs) {
            const double g = gbps(l.bytes * l.n, b.device_ms([&](int r) { b.sm(l, r, c, false); }));
            std::printf("| %d | %d | %d | %.2f |\n", c.blocks, c.threads, c.unroll, g);
        }
        // Correctness: the last rep's list, read back slot by slot.
        CUDA_OK(cudaStreamSynchronize(b.s[0]));
        const int r = reps - 1;
        std::vector<char> back(l.bytes);
        for (int j = 0; j < l.n; ++j) {
            CUDA_OK(cudaMemcpy(back.data(), b.p.slots + l.dst[r][j], l.bytes, cudaMemcpyDeviceToHost));
            if (std::memcmp(back.data(), b.p.host + l.src[r][j], l.bytes) != 0) {
                std::fprintf(stderr, "SM copy mismatch at job %d\n", j);
                return 1;
            }
        }
        std::printf("\n(SM copy verified byte for byte on the last list.)\n");
        free_lists(l);
    }

    // 3. Each cell's best SM config, the best one confined to <= 16 blocks
    // (a prefetch copy runs beside the expert kernel and cannot have the
    // card), and the ratio to the copy engine's best: the 80% rule.
    std::printf("\n## 3. SM-driven best vs copy engine best (GB/s)\n\n");
    std::printf("| bytes | n | ce best | sm best (config) | sm best <= 16 blocks (config) | sm / ce | sm<=16 / ce |\n");
    std::printf("|---:|---:|---:|---|---|---:|---:|\n");
    for (size_t si = 0; si < sizes.size(); ++si) {
        for (size_t bi = 0; bi < batches.size(); ++bi) {
            Lists l = make_lists(b.p, batches[bi], sizes[si], reps, rng, false);
            double best = 0, best_small = 0;
            SmConfig bc{}, bcs{};
            for (const auto& c : configs) {
                const double g =
                    gbps(l.bytes * l.n, b.device_ms([&](int r) { b.sm(l, r, c, false); }));
                if (g > best) { best = g; bc = c; }
                if (c.blocks <= 16 && g > best_small) { best_small = g; bcs = c; }
            }
            std::printf("| %zu | %d | %.2f | %.2f (%dx%d u%d) | %.2f (%dx%d u%d) | %.0f%% | %.0f%% |\n",
                        sizes[si], l.n, ce_best[si][bi], best, bc.blocks, bc.threads, bc.unroll,
                        best_small, bcs.blocks, bcs.threads, bcs.unroll,
                        100.0 * best / ce_best[si][bi], 100.0 * best_small / ce_best[si][bi]);
            free_lists(l);
        }
    }

    // 4. Write-combined host memory, on the decode layer's shape.
    {
        std::printf("\n## 4. write-combined pinned pool, n = 20 x 1 MiB (GB/s)\n\n");
        Lists l = make_lists(b.p, 20, 1 * MIB, reps, rng, true);
        const size_t moved = l.bytes * l.n;
        double best_plain = 0, best_wc = 0;
        SmConfig bp{}, bw{};
        for (const auto& c : configs) {
            const double g = gbps(moved, b.device_ms([&](int r) { b.sm(l, r, c, false); }));
            const double w = gbps(moved, b.device_ms([&](int r) { b.sm(l, r, c, true); }));
            if (g > best_plain) { best_plain = g; bp = c; }
            if (w > best_wc) { best_wc = w; bw = c; }
        }
        std::printf("| pool | sm best (config) |\n|---|---|\n");
        std::printf("| mapped | %.2f (%dx%d u%d) |\n", best_plain, bp.blocks, bp.threads, bp.unroll);
        std::printf("| mapped + write-combined | %.2f (%dx%d u%d) |\n", best_wc, bw.blocks,
                    bw.threads, bw.unroll);
        free_lists(l);
    }

    // 5. What a host-orchestrated layer pays before its first copy: the
    // router's selection read back (10 ids x 3 lanes as int32) behind a
    // trivial kernel, and the stream synced, host-timed.
    {
        int* flag = nullptr;
        CUDA_OK(cudaMalloc(&flag, 256));
        int* readback = nullptr;
        CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&readback), 256, cudaHostAllocDefault));
        const double ms = b.host_ms([&](int) {
            touch<<<1, 1, 0, b.s[0]>>>(flag);
            CUDA_OK(cudaMemcpyAsync(readback, flag, 120, cudaMemcpyDeviceToHost, b.s[0]));
        });
        std::printf("\n## 5. host round trip per layer (kernel + 120 B readback + sync, host clock)\n\n");
        std::printf("| median us | x 48 layers, ms per token |\n|---:|---:|\n| %.1f | %.2f |\n",
                    ms * 1e3, ms * 48);
        CUDA_OK(cudaFree(flag));
        CUDA_OK(cudaFreeHost(readback));
    }

    CUDA_OK(cudaFree(b.p.slots));
    CUDA_OK(cudaFreeHost(b.p.host));
    CUDA_OK(cudaFreeHost(b.p.host_wc));
    return 0;
}
