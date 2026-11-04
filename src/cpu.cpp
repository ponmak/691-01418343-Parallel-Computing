#include "graph.h"

#include <vector>
#include <queue>
#include <chrono>

BFSResult runSequentialBFS(
    const Graph& graph,
    int source
) {
    BFSResult result;
    result.source = source;

    result.distance.assign(graph.numNodes, -1);

    std::queue<int> frontier;

    auto start = std::chrono::high_resolution_clock::now();

    // Level 0
    result.distance[source] = 0;
    frontier.push(source);

    while (!frontier.empty()) {

        int u = frontier.front();
        frontier.pop();

        result.reachedVertices++;

        int currentDistance = result.distance[u];

        if (currentDistance > result.maxDistance) {
            result.maxDistance = currentDistance;
        }

        // Explore neighbors
        for (int i = graph.offsets[u];
             i < graph.offsets[u + 1]; i++) {

            int v = graph.neighbors[i];

            if (result.distance[v] == -1) {

                result.distance[v] = currentDistance + 1;

                frontier.push(v);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    result.executionTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    return result;
}
