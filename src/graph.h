#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <cstdint>
#include <string>

struct Graph {
    int numNodes = 0;
    uint64_t numEdges = 0;

    // CPU representation
    std::vector<std::vector<int>> adj;

    // Original node IDs
    std::vector<int64_t> originalIds;
};

bool loadGraph(const std::string& filename, Graph& graph);

void printGraphInfo(const Graph& graph);

struct LPAResult {
    std::vector<int> labels;
    int iterations = 0;
    int numCommunities = 0;
    double executionTime = 0.0;
};

LPAResult runSequentialLPA(
    const Graph& graph,
    int maxIterations
);

LPAResult runCudaLPA(
    const Graph& graph,
    int maxIterations
);

#endif