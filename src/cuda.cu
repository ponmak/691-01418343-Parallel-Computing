#include "graph.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void lpaKernel(
    const int* offsets,
    const int* neighbors,
    const int* oldLabels,
    int* newLabels,
    int* changed,
    int numNodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= numNodes)
        return;

    int start = offsets[node];
    int end = offsets[node + 1];

    // Isolated node keeps its label
    if (start == end) {
        newLabels[node] = oldLabels[node];
        return;
    }

    int bestLabel = INT_MAX;
    int bestCount = 0;

    // Examine each neighbor's label as a candidate
    for (int i = start; i < end; i++) {

        int candidate = oldLabels[neighbors[i]];
        int count = 0;

        // Count how often this label appears
        for (int j = start; j < end; j++) {
            if (oldLabels[neighbors[j]] == candidate) {
                count++;
            }
        }

        // Most frequent label, tie -> smallest label
        if (count > bestCount ||
            (count == bestCount && candidate < bestLabel)) {

            bestCount = count;
            bestLabel = candidate;
        }
    }

    newLabels[node] = bestLabel;

    if (bestLabel != oldLabels[node]) {
        atomicExch(changed, 1);
    }
}

LPAResult runCudaLPA(
    const Graph& graph,
    int maxIterations
) {
    LPAResult result;

    int n = graph.numNodes;

    if (n == 0 || maxIterations <= 0) {
        return result;
    }

    // Convert CPU adjacency list to GPU flat arrays
    std::vector<int> offsets(n + 1, 0);
    std::vector<int> neighbors;

    for (int i = 0; i < n; i++) {
        offsets[i] = static_cast<int>(neighbors.size());

        for (int neighbor : graph.adj[i]) {
            neighbors.push_back(neighbor);
        }
    }

    offsets[n] = static_cast<int>(neighbors.size());

    // Initialize labels
    std::vector<int> initialLabels(n);

    for (int i = 0; i < n; i++) {
        initialLabels[i] = i;
    }

    int* d_offsets = nullptr;
    int* d_neighbors = nullptr;
    int* d_oldLabels = nullptr;
    int* d_newLabels = nullptr;
    int* d_changed = nullptr;

    size_t offsetBytes = offsets.size() * sizeof(int);
    size_t neighborBytes = neighbors.size() * sizeof(int);
    size_t labelBytes = n * sizeof(int);

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_offsets, offsetBytes));
    CUDA_CHECK(cudaMalloc(&d_neighbors, neighborBytes));
    CUDA_CHECK(cudaMalloc(&d_oldLabels, labelBytes));
    CUDA_CHECK(cudaMalloc(&d_newLabels, labelBytes));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    // Copy graph and initial labels to GPU
    CUDA_CHECK(cudaMemcpy(
        d_offsets, offsets.data(),
        offsetBytes, cudaMemcpyHostToDevice
    ));

    if (!neighbors.empty()) {
        CUDA_CHECK(cudaMemcpy(
            d_neighbors, neighbors.data(),
            neighborBytes, cudaMemcpyHostToDevice
        ));
    }

    CUDA_CHECK(cudaMemcpy(
        d_oldLabels, initialLabels.data(),
        labelBytes, cudaMemcpyHostToDevice
    ));

    const int threadsPerBlock = 256;
    const int blocks =
        (n + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < maxIterations; iter++) {

        int hostChanged = 0;

        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

        lpaKernel<<<blocks, threadsPerBlock>>>(
            d_offsets,
            d_neighbors,
            d_oldLabels,
            d_newLabels,
            d_changed,
            n
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            &hostChanged,
            d_changed,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        // Swap label buffers
        int* temp = d_oldLabels;
        d_oldLabels = d_newLabels;
        d_newLabels = temp;

        result.iterations = iter + 1;

        if (hostChanged == 0) {
            break;
        }
    }

    // Copy final labels back to CPU
    result.labels.resize(n);

    CUDA_CHECK(cudaMemcpy(
        result.labels.data(),
        d_oldLabels,
        labelBytes,
        cudaMemcpyDeviceToHost
    ));

    auto end = std::chrono::high_resolution_clock::now();

    result.executionTime =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    // Count communities
    std::vector<int> uniqueLabels = result.labels;

    std::sort(uniqueLabels.begin(), uniqueLabels.end());

    uniqueLabels.erase(
        std::unique(uniqueLabels.begin(), uniqueLabels.end()),
        uniqueLabels.end()
    );

    result.numCommunities = uniqueLabels.size();

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_neighbors));
    CUDA_CHECK(cudaFree(d_oldLabels));
    CUDA_CHECK(cudaFree(d_newLabels));
    CUDA_CHECK(cudaFree(d_changed));

    return result;
}