
#include "graph.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>

LPAResult runSequentialLPA( const Graph& graph, int maxIterations) {
    LPAResult result;

    int n = graph.numNodes;

    std::vector<int> oldLabels(n);
    std::vector<int> newLabels(n);

    // Initialize labels
    for (int i = 0; i < n; i++) {
        oldLabels[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Start LPA iterations
    for (int iter = 0; iter < maxIterations; iter++) {

        bool changed = false;

        for (int node = 0; node < n; node++) {

            std::unordered_map<int, int> frequency;

            // Count neighbor labels
            for (int neighbor : graph.adj[node]) {
                frequency[oldLabels[neighbor]]++;
            }

            // Isolated node keeps its current label
            if (frequency.empty()) {
                newLabels[node] = oldLabels[node];
                continue;
            }

            int bestLabel = -1;
            int bestCount = -1;

            // Find most frequent label
            // Tie-breaking: choose smallest label
            for (const auto& entry : frequency) {

                int label = entry.first;
                int count = entry.second;

                if (count > bestCount ||
                    (count == bestCount && label < bestLabel)) {

                    bestCount = count;
                    bestLabel = label;
                }
            }

            newLabels[node] = bestLabel;

            if (newLabels[node] != oldLabels[node]) {
                changed = true;
            }
        }

        oldLabels.swap(newLabels);

        result.iterations = iter + 1;

        if (!changed) {
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    result.executionTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    result.labels = oldLabels;

    // Count unique communities
    std::vector<int> uniqueLabels = oldLabels;

    std::sort(uniqueLabels.begin(), uniqueLabels.end());

    uniqueLabels.erase(
        std::unique(uniqueLabels.begin(), uniqueLabels.end()),
        uniqueLabels.end()
    );

    result.numCommunities = uniqueLabels.size();

    return result;
}
