
#include "graph.h"

#include <iostream>

int main() {

    Graph graph;

    std::string filename =
        "dataset/com-amazon.ungraph.txt";

    std::cout << "Loading graph...\n";

    if (!loadGraph(filename, graph)) {
        std::cout << "Failed to load graph!\n";
        return 1;
    }

    printGraphInfo(graph);

    std::cout << "\nRunning Sequential LPA...\n";

    int maxIterations = 20;

    LPAResult result =
        runSequentialLPA(graph, maxIterations);

    std::cout << "\n===== SEQUENTIAL LPA RESULTS =====\n";

    std::cout << "Iterations: "
              << result.iterations << "\n";

    std::cout << "Communities: "
              << result.numCommunities << "\n";

    std::cout << "Execution Time: "
              << result.executionTime << " ms\n";

    std::cout << "==================================\n";

    return 0;
}
