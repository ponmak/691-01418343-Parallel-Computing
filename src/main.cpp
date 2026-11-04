#include "graph.h"

#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <exception>

using namespace std;

int main() {

    try {

        const char* filename =
            "dataset/com-orkut.ungraph.txt";

        cout << "Loading Orkut Dataset...\n";

        Graph graph = loadGraph(filename);

        cout << "Dataset loaded successfully!\n";

        cout << "Vertices: "
             << graph.numNodes << "\n";

        cout << "Undirected Edges: "
             << graph.numEdges << "\n";

        cout << "Adjacency Entries: "
             << graph.neighbors.size() << "\n";

        // Random Source with fixed seed
        const unsigned int seed = 42;

        mt19937 generator(seed);

        uniform_int_distribution<int> distribution(
            0, graph.numNodes - 1
        );

        int source = distribution(generator);

        cout << "\n===== BFS CONFIGURATION =====\n";

        cout << "Random Seed: " << seed << "\n";

        cout << "Source Internal ID: "
             << source << "\n";

        cout << "Source Original ID: "
             << graph.originalIds[source] << "\n";

        cout << "\nRunning Sequential CPU DFS...\n";

        BFSResult cpuResult = runSequentialBFS(graph, source);

        BFSResult gpuResult = runParallelBFS(graph, source);

        cout << "\n===== CPU BFS =====\n";
        cout << "Reached vertices: "
                  << cpuResult.reachedVertices << "\n";
        cout << "Max distance: "
                  << cpuResult.maxDistance << "\n";
        cout << "Execution time: "
                  << cpuResult.executionTime << " ms\n";

        cout << "\n===== GPU BFS =====\n";
        cout << "Reached vertices: "
                  << gpuResult.reachedVertices << "\n";
        cout << "Max distance: "
                  << gpuResult.maxDistance << "\n";
        cout << "Execution time: "
                  << gpuResult.executionTime << " ms\n";

        // Verify all BFS distances.
        bool correct =
            cpuResult.distance == gpuResult.distance;

        cout << "\n===== Comparison =====\n";
        cout << "Reached vertices match: "
                  << (cpuResult.reachedVertices ==
                      gpuResult.reachedVertices ? "YES" : "NO")
                  << "\n";
                
        cout << "BFS distances match: "
                  << (correct ? "YES" : "NO") << "\n";
                
        if (gpuResult.executionTime > 0.0) {
            cout << "Speedup: "
                      << cpuResult.executionTime /
                         gpuResult.executionTime
                      << "x\n";
        }
    }
    
    catch (const exception& e) {

        cerr << "Error: " << e.what() << "\n";

        return 1;
    }

    return 0;
}
