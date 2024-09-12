#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

__global__ 
void print_array(int* arr, int num_elements) {
    for (int i = 0; i < num_elements; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {

    int num_elements = 130;
    int batch_size = 12;

    int num_batches = num_elements / batch_size;
    if (num_elements % batch_size != 0) {
        num_batches++;
    }

    int* h_arr = nullptr;
    // Correcting the function name to cudaMallocHost
    cudaMallocHost(&h_arr, num_elements * sizeof(int));
    for (int i = 0; i < num_elements; i++) {
        h_arr[i] = i;
    }

    int* d_arr;
    // Correcting the function name to cudaMalloc
    cudaMalloc(&d_arr, batch_size * sizeof(int));

    for (int i = 0; i < num_batches; i++) {
        int start = i * batch_size;
        int end = std::min((i + 1) * batch_size, num_elements);
        int num_elements_in_batch = end - start;    

        cudaMemcpy(d_arr, h_arr + start, num_elements_in_batch * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << "Batch " << i << ": " << start << " to " << end << std::endl;
        std::cout << "Number of elements in batch: " << num_elements_in_batch << std::endl;

        print_array<<<1, 1>>>(d_arr, num_elements_in_batch);
        cudaDeviceSynchronize();
    }

    // Freeing the allocated memory
    cudaFree(d_arr);
    cudaFreeHost(h_arr);

    return 0;
}
