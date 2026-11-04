
#include "graph.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at line " << __LINE__ << std::endl;\
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// Initialize BFS source
__global__ void initializeBFS(
    int source,
    int* distance,
    int* frontier,
    int* frontierSize
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        distance[source] = 0;
        frontier[0] = source;
        *frontierSize = 1;
    }
}


// Expand current frontier
__global__ void expandFrontier(
    const int* offsets,
    const int* neighbors,
    int* distance,
    const int* currentFrontier,
    int currentSize,
    int* nextFrontier,
    int* nextSize,
    int currentLevel
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= currentSize)
        return;

    int u = currentFrontier[tid];

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int i = start; i < end; i++) {

        int v = neighbors[i];

        // Claim vertex only once.
        if (atomicCAS(&distance[v], -1, currentLevel + 1) == -1) {

            int position = atomicAdd(nextSize, 1);

            nextFrontier[position] = v;
        }
    }
}

// Parallel BFS
BFSResult runParallelBFS(
    const Graph& graph,
    int source
) {
    BFSResult result;
    result.source = source;

    int V = graph.numNodes;

    if (V <= 0 || source < 0 || source >= V) {
        std::cerr << "Invalid graph or source." << std::endl;
        return result;
    }

    result.distance.assign(V, -1);

    int* d_offsets = nullptr;
    int* d_neighbors = nullptr;
    int* d_distance = nullptr;

    int* d_frontierA = nullptr;
    int* d_frontierB = nullptr;

    int* d_currentSize = nullptr;
    int* d_nextSize = nullptr;

    size_t vertexBytes = (size_t)V * sizeof(int);
    size_t offsetBytes = (size_t)(V + 1) * sizeof(int);
    size_t neighborBytes =
        graph.neighbors.size() * sizeof(int);

    // Allocate GPU memory.
    CUDA_CHECK(cudaMalloc(&d_offsets, offsetBytes));
    CUDA_CHECK(cudaMalloc(&d_neighbors, neighborBytes));
    CUDA_CHECK(cudaMalloc(&d_distance, vertexBytes));

    CUDA_CHECK(cudaMalloc(&d_frontierA, vertexBytes));
    CUDA_CHECK(cudaMalloc(&d_frontierB, vertexBytes));

    CUDA_CHECK(cudaMalloc(&d_currentSize, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextSize, sizeof(int)));

    // Copy CSR graph to GPU.
    CUDA_CHECK(cudaMemcpy(
        d_offsets,
        graph.offsets.data(),
        offsetBytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_neighbors,
        graph.neighbors.data(),
        neighborBytes,
        cudaMemcpyHostToDevice
    ));

    // Initialize arrays.
    CUDA_CHECK(cudaMemset(d_distance, 0xFF, vertexBytes));

    CUDA_CHECK(cudaMemset(d_currentSize, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nextSize, 0, sizeof(int)));

    cudaEvent_t startEvent, stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    const int blockSize = 256;

    CUDA_CHECK(cudaEventRecord(startEvent));

    // Initialize source.
    initializeBFS<<<1, 1>>>(
        source,
        d_distance,
        d_frontierA,
        d_currentSize
    );

    CUDA_CHECK(cudaGetLastError());

    int* currentFrontier = d_frontierA;
    int* nextFrontier = d_frontierB;

    int currentSize = 1;
    int level = 0;

    // Level-synchronous BFS.
    while (currentSize > 0) {

        CUDA_CHECK(cudaMemset(
            d_nextSize, 0, sizeof(int)
        ));

        int blocks =
            (currentSize + blockSize - 1) / blockSize;

        expandFrontier<<<blocks, blockSize>>>(
            d_offsets,
            d_neighbors,
            d_distance,
            currentFrontier,
            currentSize,
            nextFrontier,
            d_nextSize,
            level
        );

        CUDA_CHECK(cudaGetLastError());

        // Synchronize to obtain the next frontier size.
        CUDA_CHECK(cudaMemcpy(
            &currentSize,
            d_nextSize,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        // Swap frontiers.
        std::swap(currentFrontier, nextFrontier);

        level++;
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsedMs,
        startEvent,
        stopEvent
    ));

    result.executionTime = elapsedMs;

    // Copy distances back.
    CUDA_CHECK(cudaMemcpy(
        result.distance.data(),
        d_distance,
        vertexBytes,
        cudaMemcpyDeviceToHost
    ));

    // Calculate result statistics.
    for (int v = 0; v < V; v++) {

        if (result.distance[v] >= 0) {
            result.reachedVertices++;

            result.maxDistance = std::max(
                result.maxDistance,
                result.distance[v]
            );
        }
    }

    // Free GPU memory.
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_distance));

    CUDA_CHECK(cudaFree(d_frontierA));
    CUDA_CHECK(cudaFree(d_frontierB));

    CUDA_CHECK(cudaFree(d_currentSize));
    CUDA_CHECK(cudaFree(d_nextSize));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    return result;
}
