#ifndef CUDA_UTILITY_CUH
#define CUDA_UTILITY_CUH

#include <iostream>
#include <cuda_runtime.h>

inline void check_for_error(cudaError_t error, const std::string& message, const std::string& file, int line) noexcept {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << message << "\n";
        std::cerr << "CUDA Error description: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// #define DEBUG

#define CUDA_CHECK(err, msg) check_for_error(err, msg, __FILE__, __LINE__)

template <typename T>
__global__ 
void print_device_array_kernel(T* array, long size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) { 
        for (int i = 0; i < size; ++i) {
            printf("Array[%d] = %lld\n", i, static_cast<long long>(array[i]));
        }
    }
}

template <typename T>
inline void print_device_array(const T* arr, long size) {
    print_device_array_kernel<<<1, 1>>>(arr, size);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_device_array_kernel");
    std::cout << std::endl;
}

// Function to print the edge list
inline void print_edge_list(uint64_t* edge_list, long numEdges) {
    for (long i = 0; i < numEdges; ++i) {
        uint64_t edge = edge_list[i];
        int u = edge >> 32;          // Extract the upper 32 bits
        int v = edge & 0xFFFFFFFF;   // Extract the lower 32 bits
        std::cout << u << " " << v << std::endl;
    }
}

// CUDA kernel to print the edge list from device memory
template <typename T>
__global__ 
void print_device_edge_list_kernel(T* d_edge_list, long numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for(long i = 0; i < numEdges; ++i) {
            uint64_t edge = d_edge_list[i];
            int u = edge >> 32;          // Extract the upper 32 bits
            int v = edge & 0xFFFFFFFF;   // Extract the lower 32 bits
            printf("%ld: %d %d\n", i, u, v);
        }
    }
}
template <typename T>
void print_device_edges(T* d_edge_list, long numEdges) {
    print_device_edge_list_kernel<<<1,1>>>(d_edge_list, numEdges);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
}

#endif // CUDA_UTILITY_CUH