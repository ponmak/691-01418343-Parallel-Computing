
#include "graph.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

bool loadGraph(const std::string& filename, Graph& graph) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Cannot open dataset!\n";
        return false;
    }

    std::unordered_map<int64_t, int> idMap;

    auto getNodeId = [&](int64_t id) {
        auto it = idMap.find(id);

        if (it != idMap.end())
            return it->second;

        int newId = graph.originalIds.size();

        idMap[id] = newId;
        graph.originalIds.push_back(id);

        return newId;
    };

    std::string line;
    int64_t u, v;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);

        if (!(iss >> u >> v))
            continue;

        if (u == v)
            continue;

        int src = getNodeId(u);
        int dst = getNodeId(v);

        if (graph.adj.size() <= static_cast<size_t>(src))
            graph.adj.resize(src + 1);

        if (graph.adj.size() <= static_cast<size_t>(dst))
            graph.adj.resize(dst + 1);

        graph.adj[src].push_back(dst);
        graph.adj[dst].push_back(src);

        graph.numEdges++;
    }

    graph.numNodes = graph.adj.size();

    // Sort neighbors
    for (auto& neighbors : graph.adj) {
        std::sort(neighbors.begin(), neighbors.end());
    }

    file.close();

    return true;
}

void printGraphInfo(const Graph& graph) {
    uint64_t totalNeighbors = 0;
    size_t maxDegree = 0;

    for (const auto& neighbors : graph.adj) {
        totalNeighbors += neighbors.size();

        maxDegree = std::max(
            maxDegree,
            neighbors.size()
        );
    }

    std::cout << "\n===== GRAPH INFORMATION =====\n";
    std::cout << "Nodes: " << graph.numNodes << "\n";
    std::cout << "Undirected edges: " << graph.numEdges << "\n";
    std::cout << "Adjacency entries: " << totalNeighbors << "\n";
    std::cout << "Maximum degree: " << maxDegree << "\n";
    std::cout << "=============================\n";
}
