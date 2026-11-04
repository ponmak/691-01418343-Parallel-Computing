#include "graph.h"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <algorithm>

using namespace std;

// อ่าน Edge List และเรียก callback เมื่ออ่าน Edge ได้
template <typename Function>
void readEdges(const char* filename, Function process) {

    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open dataset file!");
    }

    string line;
    int u, v;

    while (getline(file, line)) {

        if (line.empty() || line[0] == '#') {
            continue;
        }

        istringstream iss(line);

        if (!(iss >> u >> v)) {
            continue;
        }

        if (u == v) {
            continue;
        }

        process(u, v);
    }
}

Graph loadGraph(const char* filename) {

    Graph graph;

    unordered_map<int, int> idMap;

    // PASS 1: สร้าง Vertex ID mapping
    readEdges(filename, [&](int u, int v) {

        if (idMap.find(u) == idMap.end()) {
            int index = graph.numNodes++;
            idMap[u] = index;
            graph.originalIds.push_back(u);
        }

        if (idMap.find(v) == idMap.end()) {
            int index = graph.numNodes++;
            idMap[v] = index;
            graph.originalIds.push_back(v);
        }

        graph.numEdges++;
    });

    // PASS 2: นับ Degree
    vector<int> degree(graph.numNodes, 0);

    readEdges(filename, [&](int u, int v) {

        int a = idMap.at(u);
        int b = idMap.at(v);

        degree[a]++;
        degree[b]++;
    });

    // สร้าง CSR offsets
    graph.offsets.resize(graph.numNodes + 1);

    graph.offsets[0] = 0;

    for (int i = 0; i < graph.numNodes; i++) {

        graph.offsets[i + 1] =
            graph.offsets[i] + degree[i];
    }

    // จองพื้นที่ Neighbor
    graph.neighbors.resize(
        static_cast<size_t>(graph.offsets.back())
    );

    // ใช้ cursor สำหรับเติม Neighbor
    vector<int> cursor(
        graph.offsets.begin(),
        graph.offsets.end() - 1
    );

    // PASS 3: เติม Neighbor
    readEdges(filename, [&](int u, int v) {

        int a = idMap.at(u);
        int b = idMap.at(v);

        graph.neighbors[cursor[a]++] = b;
        graph.neighbors[cursor[b]++] = a;
    });

    return graph;
}
