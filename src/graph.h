
#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <cstdint>

struct Graph {
    int numNodes = 0;
    int64_t numEdges = 0;

    std::vector<int> offsets;
    std::vector<int> neighbors;
    std::vector<int> originalIds;
};

struct BFSResult {
    std::vector<int> distance;

    int source = -1;
    int reachedVertices = 0;
    int maxDistance = 0;

    double executionTime = 0.0;
};

Graph loadGraph(const char* filename);

BFSResult runSequentialBFS(
    const Graph& graph,
    int source
);

BFSResult runParallelBFS(
    const Graph& graph,
    int source
);

#endif
