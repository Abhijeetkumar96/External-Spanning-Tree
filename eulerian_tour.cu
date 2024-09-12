#include <iostream>
#include <fstream>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "list_ranking.cuh"
#include "util.cuh"

template<typename T>
void display_device_array(T* d_arr, int num_elem) {
    T *host_array = new T[num_elem];
    cudaMemcpy(host_array, d_arr, num_elem * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < num_elem; ++i) {
      std::cout << "arr[" << i << "]= " << host_array[i] << "\n";
    }
    std::cout << std::endl;
}

// Function to display an edge list from device memory
void DisplayDeviceEdgeList(int* d_u, int* d_v, int num_edges) {
    // Allocate host memory for the edges
    int* h_u = new int[num_edges];
    int* h_v = new int[num_edges];

    // Check for memory allocation failure
    if (h_u == nullptr || h_v == nullptr) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return;
    }

    // Copy the edges from device to host
    CUCHECK(cudaMemcpy(h_u, d_u, sizeof(int) * num_edges, cudaMemcpyDeviceToHost));
    CUCHECK(cudaMemcpy(h_v, d_v, sizeof(int) * num_edges, cudaMemcpyDeviceToHost));

    // Print the edges
    std::cout << "Edge list:" << std::endl;
    for (int i = 0; i < num_edges; i++) {
        std::cout << i << " :(" << h_u[i] << ", " << h_v[i] << ")" << std::endl;
    }

    // Free host memory
    delete[] h_u;
    delete[] h_v;
}

__global__ 
void create_dup_edges(
    int *d_edges_to, 
    int *d_edges_from, 
    const uint64_t *d_edges_input, 
    const int root,
    int N) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thid < N) {

        if (thid == root)
          return;

        int edge_count = N - 1;
        uint64_t i = d_edges_input[thid];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        int afterRoot = thid > root;
        // printf("For thid: %d, thid - afterRoot: %d, thid - afterRoot + edge_count: %d\n", thid, thid - afterRoot, thid - afterRoot + edge_count);

        d_edges_from[thid - afterRoot + edge_count] = d_edges_to[thid - afterRoot] = v;
        d_edges_to[thid - afterRoot + edge_count] = d_edges_from[thid - afterRoot] = u;
    }
}

__global__
void init_nxt(int* d_next, int E) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < E) {
        d_next[thid] = -1;
    }
}

__global__
void update_first_last_nxt(int* d_edges_from, int* d_edges_to, int* d_first, int* d_last, int* d_next, uint64_t* d_index, int E) {
    
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if(thid < E) {
        int f = d_edges_from[d_index[thid]];
        int t = d_edges_to[d_index[thid]];

        if (thid == 0) {
            d_first[f] = d_index[thid];
            return;
        }

        if(thid == E - 1) {
            d_last[f] = d_index[thid];
        }

        int pf = d_edges_from[d_index[thid - 1]];
        int pt = d_edges_to[d_index[thid - 1]];

        // printf("For tid: %d, f: %d, t: %d, pf: %d, pt: %d\n", thid, f, t, pf, pt);

        // calculate the offset array
        if (f != pf) {
            d_first[f] = d_index[thid];
            // printf("d_last[%d] = d_index[%d] = %d\n", pf, thid - 1, d_index[thid - 1]);
            d_last[pf] = d_index[thid - 1];
        } else {
            d_next[d_index[thid - 1]] = d_index[thid];
        }
    }
}

__global__ 
void cal_succ(int* succ, const int* d_next, const int* d_first, const int* d_edges_from, int E) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < E) {
        int revEdge = (thid + E / 2) % E;

        if (d_next[revEdge] == -1) {
            succ[thid] = d_first[d_edges_from[revEdge]];
        } else {
            succ[thid] = d_next[revEdge];
        }
    }
}

__global__ 
void break_cycle_kernel(int *d_last, int *d_succ, int* d_roots, int roots_count, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < roots_count) {
        int root = d_roots[idx];
        // printf("Root: %d\n", root);
        if (d_last[root] != -1) {
            int last_edge = d_last[root];
            int rev_edge = (last_edge + E / 2) % E;
            // printf("\nFor root: %d, last_edge: %d, rev_edge: %d\n", root, last_edge, rev_edge);
            // Set the successor of the last edge to point to itself
            d_succ[rev_edge] = -1;
        }
    }
}

__global__
void init_parent(int* d_parent, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_parent[idx] = idx;
    }
}

__global__
void find_parent(int E, int *rank, int *d_edges_to, int *d_edges_from, int *parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < E) {
        int f = d_edges_from[tid];
        int t = d_edges_to[tid];
        int rev_edge = (tid + E / 2) % E;
        // printf("for tid: %d, f: %d, t: %d, rev_edge: %d\n", tid, f, t, rev_edge);
        if(rank[tid] > rank[rev_edge]) {
            parent[t] = f;
        }
        else {
            parent[f] = t;
        }
    }
}

__global__ 
void merge_key_value(const int *arrayU, const int *arrayV, uint64_t *arrayE, uint64_t *d_indices, long size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Cast to int64_t to ensure the shift operates on 64 bits
        uint64_t u = arrayU[idx];
        uint64_t v = arrayV[idx];

        arrayE[idx] = (u << 32) | (v & 0xFFFFFFFFLL);

        d_indices[idx] = idx;
    }
}


void LexSortIndices(int* d_keys, int* d_values, uint64_t* d_indices_sorted, int num_items) {

    uint64_t *d_merged, *d_merged_keys_sorted;
    cudaMalloc(&d_merged, sizeof(uint64_t) * num_items);
    cudaMalloc(&d_merged_keys_sorted, sizeof(uint64_t) * num_items);

    uint64_t* d_indices;
    cudaMalloc(&d_indices, sizeof(uint64_t)* num_items);   

    int blockSize = 1024;
    int numBlocks = (num_items + blockSize - 1) / blockSize; 

    // Initialize indices to 0, 1, 2, ..., num_items-1 also here
    merge_key_value<<<numBlocks, blockSize>>>(
        d_keys, 
        d_values, 
        d_merged, 
        d_indices, 
        num_items);
    CUCHECK(cudaDeviceSynchronize());

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temporary storage requirements
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_merged, d_merged_keys_sorted, d_indices, d_indices_sorted, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Sort indices based on keys
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_merged, d_merged_keys_sorted, d_indices, d_indices_sorted, num_items);

    cudaFree(d_merged);
    cudaFree(d_merged_keys_sorted);
    cudaFree(d_indices);
    cudaFree(d_temp_storage);
}

void cuda_euler_tour(
    int N, 
    int root, 
    uint64_t* d_edges_input) {
    
    int E = N * 2 - 2;
    int roots_count = 1;

    int *d_edges_to;
    int *d_edges_from;
    CUCHECK(cudaMalloc((void **)&d_edges_to, sizeof(int) * E));
    CUCHECK(cudaMalloc((void **)&d_edges_from, sizeof(int) * E));
    
    // index can be considered as edge_num
    uint64_t *d_index;
    CUCHECK(cudaMalloc((void **)&d_index, sizeof(uint64_t) * E));

    int *d_next;
    CUCHECK(cudaMalloc((void **)&d_next, sizeof(int) * E));

    int *d_roots;
    CUCHECK(cudaMalloc((void **)&d_roots, sizeof(int) * roots_count));
    cudaMemcpy(d_roots, &root, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (N - 1 + blockSize - 1) / blockSize; 

    // Launch the kernel
    create_dup_edges<<<numBlocks, blockSize>>>(
        d_edges_to, 
        d_edges_from, 
        d_edges_input, 
        root, 
        N);

    CUCHECK(cudaDeviceSynchronize());
    // std::cout << "Printing from Euler Tour after creating duplicates:\n";
    // DisplayDeviceEdgeList(d_edges_from, d_edges_to, E);

    numBlocks = (E + blockSize - 1) / blockSize;

    init_nxt<<<numBlocks, blockSize>>>(d_next, E);
    CUCHECK(cudaDeviceSynchronize()); 

    LexSortIndices(d_edges_from, d_edges_to, d_index, E);

    #ifdef DEBUG
        std::cout << "Index array:\n";
        display_device_array(d_index, E);

        std::vector<int> sorted_from(E), sorted_to(E);
        std::vector<uint64_t> sorted_index(E);
        
        CUCHECK(cudaMemcpy(sorted_index.data(), d_index, sizeof(uint64_t) * E, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(sorted_from.data(), d_edges_from, sizeof(int) * E, cudaMemcpyDeviceToHost));
        CUCHECK(cudaMemcpy(sorted_to.data(), d_edges_to, sizeof(int) * E, cudaMemcpyDeviceToHost));

        // Print the sorted edges
        std::cout << "Sorted Edges:" << std::endl;
        for (int i = 0; i < E; ++i) {
            int idx = sorted_index[i];
            std::cout << i << ": (" << sorted_from[idx] << ", " << sorted_to[idx] << ")" << std::endl;
        }
    #endif

    int *d_first;
    CUCHECK(cudaMalloc((void **)&d_first, sizeof(int) * N));
    CUCHECK(cudaMemset(d_first, -1, sizeof(int) * N));

    int *d_last;
    CUCHECK(cudaMalloc((void **)&d_last, sizeof(int) * N));
    CUCHECK(cudaMemset(d_last, -1, sizeof(int) * N));

    update_first_last_nxt<<<numBlocks, blockSize>>>(
        d_edges_from, 
        d_edges_to, 
        d_first, 
        d_last, 
        d_next, 
        d_index, 
        E);

    CUCHECK(cudaDeviceSynchronize());

    int *succ;
    CUCHECK(cudaMalloc((void **)&succ, sizeof(int) * E));

    int *devRank;
    CUCHECK(cudaMalloc((void **)&devRank, sizeof(int) * E));

    cal_succ<<<numBlocks, blockSize>>>(succ, d_next, d_first, d_edges_from, E);
    CUCHECK(cudaDeviceSynchronize());

    // std::cout << "successor array before break_cycle_kernel:\n";
    // display_device_array(succ, E);

    // break cycle_kernel
    numBlocks = (roots_count + blockSize - 1) / blockSize;
    break_cycle_kernel<<<numBlocks, blockSize>>>(d_last, succ, d_roots, roots_count, E);
    CUCHECK(cudaDeviceSynchronize());

    CudaSimpleListRank(devRank, E, succ);

    #ifdef DEBUG
        std::cout << "d_first array:\n";
        display_device_array(d_first, N);

        std::cout << "d_last array:\n";
        display_device_array(d_last, N);

        std::cout << "d_next array:\n";
        display_device_array(d_next, E);

        std::cout << "successor array:\n";
        display_device_array(succ, E);

        std::cout << "euler Path array:\n";
        display_device_array(devRank, E);
    #endif

    int *d_parent;
    CUCHECK(cudaMalloc((void **)&d_parent, sizeof(int) * N));

    numBlocks = (N + blockSize - 1) / blockSize;

    init_parent<<<numBlocks, blockSize>>>(d_parent, N);
    CUCHECK(cudaDeviceSynchronize());

    numBlocks = (E + blockSize - 1) / blockSize;
    find_parent<<<numBlocks, blockSize>>>(E, devRank, d_edges_to, d_edges_from, d_parent);
    CUCHECK(cudaDeviceSynchronize());

    #ifdef DEBUG
        std::cout << "Parent array:\n";
        display_device_array(d_parent, N);
    #endif
    
    CUCHECK(cudaFree(d_edges_to));
    CUCHECK(cudaFree(d_edges_from));
    CUCHECK(cudaFree(d_index));
    CUCHECK(cudaFree(d_next));
    CUCHECK(cudaFree(d_first));
    CUCHECK(cudaFree(succ));
}