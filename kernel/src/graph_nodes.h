// ignis kernel leaf - GitHub #718 (spec #711): what a captured graph is made of, as
// `ignis_program_stats` reports it -- OURS. Its root nodes (those with no dependency, where a
// replay starts work at once) and its nodes of one type; each 0 where the runtime cannot say.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ignis::graph_nodes {

inline std::uint32_t roots(cudaGraph_t graph) {
  std::size_t count = 0;
  if (cudaGraphGetRootNodes(graph, nullptr, &count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return static_cast<std::uint32_t>(count);
}

inline std::uint32_t of_type(cudaGraph_t graph, cudaGraphNodeType type) {
  std::size_t count = 0;
  if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  std::vector<cudaGraphNode_t> nodes(count);
  if (count > 0 && cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  std::uint32_t matching = 0;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType node_type = cudaGraphNodeTypeCount;
    if (cudaGraphNodeGetType(node, &node_type) != cudaSuccess) {
      (void)cudaGetLastError();
      return 0;
    }
    matching += node_type == type ? 1 : 0;
  }
  return matching;
}

} // namespace ignis::graph_nodes
