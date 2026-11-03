
#include "graph.h"

#include <iostream>
#include <algorithm>

int main() {

    Graph graph;

    std::string filename =
        "dataset/com-orkut.ungraph.txt";

    std::cout << "Loading graph...\n";

    if (!loadGraph(filename, graph)) {
        std::cout << "Failed to load graph!\n";
        return 1;
    }

    printGraphInfo(graph);

    int maxIterations = 20;

    // Sequential CPU
    std::cout << "\nRunning Sequential LPA...\n";

    LPAResult cpuResult =
        runSequentialLPA(graph, maxIterations);

    std::cout << "\n===== CPU RESULTS =====\n";
    std::cout << "Iterations: "
              << cpuResult.iterations << "\n";
    std::cout << "Communities: "
              << cpuResult.numCommunities << "\n";
    std::cout << "Execution Time: "
              << cpuResult.executionTime << " ms\n";

    // Parallel GPU
    std::cout << "\nRunning CUDA LPA...\n";

    LPAResult gpuResult =
        runCudaLPA(graph, maxIterations);

    std::cout << "\n===== GPU RESULTS =====\n";
    std::cout << "Iterations: "
              << gpuResult.iterations << "\n";
    std::cout << "Communities: "
              << gpuResult.numCommunities << "\n";
    std::cout << "Execution Time: "
              << gpuResult.executionTime << " ms\n";

    // Compare labels
    bool sameLabels =
        (cpuResult.labels == gpuResult.labels);

    std::cout << "\n===== CORRECTNESS CHECK =====\n";

    if (sameLabels) {
        std::cout << "CPU and GPU labels MATCH!\n";
    } else {
        std::cout << "CPU and GPU labels DO NOT MATCH!\n";
    }

    return 0;
}
