// Is cudaGraphLaunch's host cost proportional to the graph's node count?
//
// The decode round's device timeline shows ~370 us of idle immediately before
// the replay, and CUPTI puts the host-side cudaGraphLaunch call at the same
// ~370 us for a graph of roughly 1,090 nodes. That is ~0.34 us per node, which
// would make node count a lever. It could equally be a fixed per-submission
// cost of this platform. This asks the driver directly.
//
//   nvcc -O2 -o graphlaunch_bench graphlaunch_bench.cu
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

__global__ void trivial(float *sink) {
  if (threadIdx.x == 1024) { *sink = 1.0F; }
}

// A graph of `nodes` trivial kernels chained one after another, which is the
// decode graph's shape: a single dependency chain, not a wide fan-out.
static double time_launch(int nodes, int reps, bool sync_each, double *out_wall_ms) {
  float *sink = nullptr;
  cudaMalloc(&sink, sizeof(float));
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  for (int i = 0; i < nodes; ++i) { trivial<<<1, 32, 0, stream>>>(sink); }
  cudaGraph_t graph = nullptr;
  cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t exec = nullptr;
  cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  cudaGraphUpload(exec, stream);
  cudaStreamSynchronize(stream);

  // Warm up: the first launch of an exec does work the rest do not.
  for (int i = 0; i < 3; ++i) { cudaGraphLaunch(exec, stream); }
  cudaStreamSynchronize(stream);

  double launch_us = 0.0;
  const auto wall_begin = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    const auto begin = std::chrono::steady_clock::now();
    cudaGraphLaunch(exec, stream);
    const auto end = std::chrono::steady_clock::now();
    launch_us += std::chrono::duration<double, std::micro>(end - begin).count();
    if (sync_each) { cudaStreamSynchronize(stream); }
  }
  cudaStreamSynchronize(stream);
  const auto wall_end = std::chrono::steady_clock::now();
  *out_wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();

  cudaGraphExecDestroy(exec);
  cudaStreamDestroy(stream);
  cudaFree(sink);
  return launch_us / reps;
}

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("%s, %d SMs\n\n", prop.name, prop.multiProcessorCount);

  std::printf("%8s %14s %14s %12s\n", "nodes", "launch us", "us per node", "wall/rep ms");
  for (const int nodes : {1, 16, 64, 256, 512, 1024, 1090, 2048}) {
    double wall = 0.0;
    const double us = time_launch(nodes, 200, /*sync_each=*/true, &wall);
    std::printf("%8d %14.2f %14.4f %12.3f\n", nodes, us, us / nodes, wall / 200.0);
  }

  std::printf("\nwithout a synchronize between launches (the queue stays full):\n");
  std::printf("%8s %14s %14s %12s\n", "nodes", "launch us", "us per node", "wall/rep ms");
  for (const int nodes : {64, 512, 1090}) {
    double wall = 0.0;
    const double us = time_launch(nodes, 200, /*sync_each=*/false, &wall);
    std::printf("%8d %14.2f %14.4f %12.3f\n", nodes, us, us / nodes, wall / 200.0);
  }
  return 0;
}
