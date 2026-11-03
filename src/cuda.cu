
#include "graph.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <chrono>
#include <cstdlib>

using namespace std;

// cuda err check
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            cerr << "CUDA Error: "                            \
                 << cudaGetErrorString(err)                   \
                 << " at " << __FILE__                        \
                 << ":" << __LINE__ << endl;                  \
            exit(EXIT_FAILURE);                              \
        }                                                     \
    } while (0)

// config
#define DEGREE_THRESHOLD 32
#define MAX_SHARED_DEGREE 2048

#define HASH_SIZE 4096
#define HIGH_THREADS 256

// if Degree > DEGREE_THRESHOLD, use high-degree kernel
__global__ void lpaLowDegreeKernel(
    const int* offsets,
    const int* neighbors,
    const int* oldLabels,
    int* newLabels,
    int* changed,
    int numNodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= numNodes) return;

    int start = offsets[node];
    int end = offsets[node + 1];

    int degree = end - start;

    // This kernel only handles low-degree nodes.
    if (degree > DEGREE_THRESHOLD) return;

    // Isolated node
    if (degree == 0) {
        newLabels[node] = oldLabels[node];
        return;
    }

    int bestLabel = INT_MAX;
    int bestCount = 0;

    // Count frequency of each neighbor label.
    for (int i = start; i < end; i++) {

        int candidate = oldLabels[neighbors[i]];

        int count = 0;

        for (int j = start; j < end; j++) {
            if (oldLabels[neighbors[j]] == candidate) {
                count++;
            }
        }

        // Most frequent label.
        // Tie: choose the smallest label.
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


// high-degree kernel: each block processes one high-degree node
__global__ void lpaHighDegreeKernel(
    const int* offsets,
    const int* neighbors,
    const int* oldLabels,
    int* newLabels,
    int* changed,
    const int* highNodes,
    int numHighNodes,
    int* overflowFlags
) {
    int idx = blockIdx.x;

    if (idx >= numHighNodes) return;

    int node = highNodes[idx];

    int start = offsets[node];
    int end = offsets[node + 1];

    int degree = end - start;

    // Shared hash table.
    __shared__ int hashKeys[HASH_SIZE];
    __shared__ int hashCounts[HASH_SIZE];

    // Shared variables for reduction.
    __shared__ int overflow;
    __shared__ int bestLabels[HIGH_THREADS];
    __shared__ int bestCounts[HIGH_THREADS];

    // Initialize overflow flag.
    if (threadIdx.x == 0) {
        overflow = 0;
        overflowFlags[idx] = 0;
    }

    // Initialize hash table.
    for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x) {
        hashKeys[i] = -1;
        hashCounts[i] = 0;
    }

    __syncthreads();

    // Extremely high-degree node:
    // Let CPU handle this node.
    if (degree > MAX_SHARED_DEGREE) {

        if (threadIdx.x == 0) {
            overflowFlags[idx] = 1;
        }

        return;
    }


    // fill hash table with neighbor labels and their counts
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {

        int label = oldLabels[neighbors[i]];

        unsigned int hash =
            ((unsigned int)label * 2654435761u)
            & (HASH_SIZE - 1);

        bool inserted = false;

        // Linear probing.
        for (int probe = 0; probe < HASH_SIZE; probe++) {

            int pos = (hash + probe) & (HASH_SIZE - 1);

            int previous = atomicCAS(
                &hashKeys[pos],
                -1,
                label
            );

            if (previous == -1 || previous == label) {

                atomicAdd(&hashCounts[pos], 1);

                inserted = true;
                break;
            }
        }

        if (!inserted) {
            atomicExch(&overflow, 1);
        }
    }

    __syncthreads();

    // Hash table overflow:
    // CPU fallback will process this node.
    if (overflow) {

        if (threadIdx.x == 0) {
            overflowFlags[idx] = 1;
        }

        return;
    }

    // Find the most frequent label in the hash table.
    int localLabel = INT_MAX;
    int localCount = 0;

    for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x) {

        int label = hashKeys[i];
        int count = hashCounts[i];

        if (label != -1 &&
            (count > localCount ||
             (count == localCount && label < localLabel))) {

            localCount = count;
            localLabel = label;
        }
    }

    bestLabels[threadIdx.x] = localLabel;
    bestCounts[threadIdx.x] = localCount;

    __syncthreads();

    // Parallel reduction to find the best label and count.
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (threadIdx.x < stride) {

            int otherCount = bestCounts[threadIdx.x + stride];
            int otherLabel = bestLabels[threadIdx.x + stride];

            int currentCount = bestCounts[threadIdx.x];
            int currentLabel = bestLabels[threadIdx.x];

            if (otherCount > currentCount ||
                (otherCount == currentCount &&
                 otherLabel < currentLabel)) {

                bestCounts[threadIdx.x] = otherCount;
                bestLabels[threadIdx.x] = otherLabel;
            }
        }

        __syncthreads();
    }

    // Thread 0 writes the final result.
    if (threadIdx.x == 0) {

        int bestLabel = bestLabels[0];

        newLabels[node] = bestLabel;

        if (bestLabel != oldLabels[node]) {
            atomicExch(changed, 1);
        }
    }
}


// CPU fallback for high-degree nodes that overflowed the shared hash table.
int calculateFallbackLabel(
    int node,
    const Graph& graph,
    const vector<int>& oldLabels
) {
    unordered_map<int, int> frequency;

    int bestLabel = INT_MAX;
    int bestCount = 0;

    for (int neighbor : graph.adj[node]) {

        int label = oldLabels[neighbor];

        int count = ++frequency[label];

        if (count > bestCount ||
            (count == bestCount && label < bestLabel)) {

            bestCount = count;
            bestLabel = label;
        }
    }

    if (bestCount == 0) {
        return oldLabels[node];
    }

    return bestLabel;
}


// Main function to run CUDA LPA.
LPAResult runCudaLPA(
    const Graph& graph,
    int maxIterations
) {
    LPAResult result;

    int numNodes = graph.numNodes;

    if (numNodes == 0) {
        return result;
    }

    auto totalStart = chrono::high_resolution_clock::now();

    // 1. convert graph to CSR format for GPU processing.
    vector<int> offsets(numNodes + 1, 0);
    vector<int> neighbors;

    neighbors.reserve(
        static_cast<size_t>(graph.numEdges) * 2
    );

    for (int i = 0; i < numNodes; i++) {

        offsets[i] = static_cast<int>(neighbors.size());

        for (int neighbor : graph.adj[i]) {
            neighbors.push_back(neighbor);
        }
    }

    offsets[numNodes] = static_cast<int>(neighbors.size());

    // 2. Identify high-degree nodes.
    vector<int> highNodes;

    for (int i = 0; i < numNodes; i++) {

        int degree = offsets[i + 1] - offsets[i];

        if (degree > DEGREE_THRESHOLD) {
            highNodes.push_back(i);
        }
    }

    int numHighNodes = static_cast<int>(highNodes.size());

    cout << "\n===== CUDA V2 CONFIGURATION =====\n";
    cout << "Threshold Degree: " << DEGREE_THRESHOLD << endl;
    cout << "Low-degree nodes: "
         << numNodes - numHighNodes << endl;
    cout << "High-degree nodes: "
         << numHighNodes << endl;
    cout << "Shared Hash Table: "
         << HASH_SIZE << " slots" << endl;
    cout << "CPU fallback degree: > "
         << MAX_SHARED_DEGREE << endl;

    // 3. Initialize labels.
    vector<int> initialLabels(numNodes);

    for (int i = 0; i < numNodes; i++) {
        initialLabels[i] = i;
    }

    // 4. Allocate GPU memory.
    int* d_offsets = nullptr;
    int* d_neighbors = nullptr;

    int* d_oldLabels = nullptr;
    int* d_newLabels = nullptr;

    int* d_changed = nullptr;

    int* d_highNodes = nullptr;
    int* d_overflowFlags = nullptr;

    size_t offsetBytes =
        offsets.size() * sizeof(int);

    size_t neighborBytes =
        neighbors.size() * sizeof(int);

    size_t labelBytes =
        static_cast<size_t>(numNodes) * sizeof(int);

    size_t highNodeBytes =
        static_cast<size_t>(numHighNodes) * sizeof(int);

    size_t overflowBytes =
        static_cast<size_t>(numHighNodes) * sizeof(int);

    CUDA_CHECK(cudaMalloc(
        &d_offsets,
        offsetBytes
    ));

    if (!neighbors.empty()) {
        CUDA_CHECK(cudaMalloc(
            &d_neighbors,
            neighborBytes
        ));
    }

    CUDA_CHECK(cudaMalloc(
        &d_oldLabels,
        labelBytes
    ));

    CUDA_CHECK(cudaMalloc(
        &d_newLabels,
        labelBytes
    ));

    CUDA_CHECK(cudaMalloc(
        &d_changed,
        sizeof(int)
    ));

    if (numHighNodes > 0) {

        CUDA_CHECK(cudaMalloc(
            &d_highNodes,
            highNodeBytes
        ));

        CUDA_CHECK(cudaMalloc(
            &d_overflowFlags,
            overflowBytes
        ));
    }

    // 5. Copy data to GPU.
    CUDA_CHECK(cudaMemcpy(
        d_offsets,
        offsets.data(),
        offsetBytes,
        cudaMemcpyHostToDevice
    ));

    if (!neighbors.empty()) {
        CUDA_CHECK(cudaMemcpy(
            d_neighbors,
            neighbors.data(),
            neighborBytes,
            cudaMemcpyHostToDevice
        ));
    }

    CUDA_CHECK(cudaMemcpy(
        d_oldLabels,
        initialLabels.data(),
        labelBytes,
        cudaMemcpyHostToDevice
    ));

    if (numHighNodes > 0) {

        CUDA_CHECK(cudaMemcpy(
            d_highNodes,
            highNodes.data(),
            highNodeBytes,
            cudaMemcpyHostToDevice
        ));
    }

    // 6. config Grid and Block sizes.
    const int threadsPerBlock = 256;

    int blocks =
        (numNodes + threadsPerBlock - 1)
        / threadsPerBlock;

    // 7. Main LPA iterations.
    int hostChanged = 0;

    for (int iter = 0; iter < maxIterations; iter++) {

        // Reset changed flag.
        CUDA_CHECK(cudaMemset(
            d_changed,
            0,
            sizeof(int)
        ));

        // Low-degree kernel.
        lpaLowDegreeKernel<<<blocks, threadsPerBlock>>>(
            d_offsets,
            d_neighbors,
            d_oldLabels,
            d_newLabels,
            d_changed,
            numNodes
        );

        CUDA_CHECK(cudaGetLastError());

        // High-degree kernel.
        if (numHighNodes > 0) {

            lpaHighDegreeKernel<<<numHighNodes, HIGH_THREADS>>>(
                d_offsets,
                d_neighbors,
                d_oldLabels,
                d_newLabels,
                d_changed,
                d_highNodes,
                numHighNodes,
                d_overflowFlags
            );

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // 8. CPU fallback for high-degree nodes that overflowed the shared hash table.
        vector<int> overflowFlags(numHighNodes, 0);

        bool hasFallback = false;

        if (numHighNodes > 0) {

            CUDA_CHECK(cudaMemcpy(
                overflowFlags.data(),
                d_overflowFlags,
                overflowBytes,
                cudaMemcpyDeviceToHost
            ));

            for (int flag : overflowFlags) {
                if (flag != 0) {
                    hasFallback = true;
                    break;
                }
            }
        }

        bool fallbackChanged = false;

        if (hasFallback) {

            // Copy old and partially computed new labels.
            vector<int> hostOldLabels(numNodes);
            vector<int> hostNewLabels(numNodes);

            CUDA_CHECK(cudaMemcpy(
                hostOldLabels.data(),
                d_oldLabels,
                labelBytes,
                cudaMemcpyDeviceToHost
            ));

            CUDA_CHECK(cudaMemcpy(
                hostNewLabels.data(),
                d_newLabels,
                labelBytes,
                cudaMemcpyDeviceToHost
            ));

            // Process only nodes marked for fallback.
            for (int i = 0; i < numHighNodes; i++) {

                if (overflowFlags[i] == 0) {
                    continue;
                }

                int node = highNodes[i];

                int bestLabel = calculateFallbackLabel(
                    node,
                    graph,
                    hostOldLabels
                );

                hostNewLabels[node] = bestLabel;

                if (bestLabel != hostOldLabels[node]) {
                    fallbackChanged = true;
                }
            }

            // Copy updated labels back to GPU.
            CUDA_CHECK(cudaMemcpy(
                d_newLabels,
                hostNewLabels.data(),
                labelBytes,
                cudaMemcpyHostToDevice
            ));
        }

        // 9. Check for convergence.
        CUDA_CHECK(cudaMemcpy(
            &hostChanged,
            d_changed,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        if (fallbackChanged) {
            hostChanged = 1;
        }

        result.iterations = iter + 1;

        // Swap old and new labels.
        swap(d_oldLabels, d_newLabels);

        if (hostChanged == 0) {
            break;
        }
    }

    // 10. Copy final labels back to host.
    result.labels.resize(numNodes);

    CUDA_CHECK(cudaMemcpy(
        result.labels.data(),
        d_oldLabels,
        labelBytes,
        cudaMemcpyDeviceToHost
    ));

    // 11. Count unique communities.
    unordered_set<int> uniqueLabels(
        result.labels.begin(),
        result.labels.end()
    );

    result.numCommunities =
        static_cast<int>(uniqueLabels.size());

    // 12. Free GPU memory.
    CUDA_CHECK(cudaFree(d_offsets));

    if (d_neighbors != nullptr) {
        CUDA_CHECK(cudaFree(d_neighbors));
    }

    CUDA_CHECK(cudaFree(d_oldLabels));
    CUDA_CHECK(cudaFree(d_newLabels));
    CUDA_CHECK(cudaFree(d_changed));

    if (d_highNodes != nullptr) {
        CUDA_CHECK(cudaFree(d_highNodes));
    }

    if (d_overflowFlags != nullptr) {
        CUDA_CHECK(cudaFree(d_overflowFlags));
    }

    // 13. Measure execution time.
    auto totalEnd = chrono::high_resolution_clock::now();

    result.executionTime =
        chrono::duration<double, milli>(
            totalEnd - totalStart
        ).count();

    return result;
}
