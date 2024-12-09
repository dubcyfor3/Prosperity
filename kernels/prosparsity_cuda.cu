#include "prosparsity_cuda.hpp"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>

#define MAX_TILE_SIZE_M 1024
#define MAX_TILE_SIZE_K 128



__global__ void prosparsity_kernel(
    uint8_t *__restrict__ input_act,
    uint8_t *__restrict__ prosparsity_act,
    int *__restrict__ prefix_array,
    int M,
    int K,
    int tile_size_m,
    int tile_size_k
)
{
    int tile_m = blockIdx.x;
    int tile_k = blockIdx.y;

    int start_m = tile_m * tile_size_m;
    int start_k = tile_k * tile_size_k;

    int end_m = min(start_m + tile_size_m, M);
    int end_k = min(start_k + tile_size_k, K);

    // Note: blockDim.x <= 1024, but tile_size_m can be larger (e.g., 2048)
    // We'll process the rows in chunks of blockDim.x rows.
    int threads_for_rows = blockDim.x;

    extern __shared__ char shared_mem[];
    int* nnz_array = (int*)shared_mem;
    uint8_t* act_tile = (uint8_t*)(shared_mem + tile_size_m * sizeof(int));

    // Initialize nnz_array
    for (int row_offset = 0; row_offset < tile_size_m; row_offset += threads_for_rows) {
        int local_row = row_offset + threadIdx.x;
        if (local_row < tile_size_m) {
            nnz_array[local_row] = 0;
        }
    }
    __syncthreads();

    // Load data into shared memory
    // Each thread processes multiple rows in chunks.
    for (int row_offset = 0; row_offset < tile_size_m; row_offset += threads_for_rows) {
        int local_row = row_offset + threadIdx.x;
        int m = start_m + local_row;
        if (local_row < tile_size_m && m < end_m) {
            int nnz = 0;
            for (int k = start_k; k < end_k; k++) {
                uint8_t val = input_act[m * K + k];
                act_tile[local_row * tile_size_k + (k - start_k)] = val;
                nnz += val;
            }
            nnz_array[local_row] = nnz;
        }
        __syncthreads(); // Ensure all threads finish loading before next iteration
    }

    __syncthreads();

    // Now we have all rows in shared memory.
    // Compute subsets and prefixes for each row
    for (int row_offset = 0; row_offset < tile_size_m; row_offset += threads_for_rows) {
        int local_row = row_offset + threadIdx.x;
        int m = start_m + local_row;

        if (local_row < tile_size_m && m < end_m) {
            bool is_subset;
            int max_subset = 0;
            int prefix = -1;

            int current_nnz = nnz_array[local_row];
            uint8_t* current_row_ptr = &act_tile[local_row * tile_size_k];

            for (int i = 0; i < tile_size_m; i++) {
                if (i == local_row) continue;
                // Check if current row is a subset of row i
                // Original logic:
                // is_subset = !(nnz_array[threadIdx.x] == nnz_array[i - start_m] && threadIdx.x <= i - start_m);
                // Adjusted for our indexing:
                // The condition basically says we start is_subset as the negation:
                is_subset = !(current_nnz == nnz_array[i] && local_row <= i);

                uint8_t* candidate_row_ptr = &act_tile[i * tile_size_k];
                for (int k = 0; k < (end_k - start_k) && is_subset; k++) {
                    is_subset = (current_row_ptr[k] >= candidate_row_ptr[k]);
                }

                if (is_subset && nnz_array[i] > max_subset) {
                    max_subset = nnz_array[i];
                    prefix = i;
                }
            }

            if (current_nnz < 2) {
                prefix = -1;
            }

            prefix_array[tile_m * gridDim.y * tile_size_m + tile_k * tile_size_m + local_row] = prefix;

            // Compute prosparsity_act
            if (prefix != -1) {
                uint8_t* prefix_row_ptr = &act_tile[prefix * tile_size_k];
                for (int k = start_k; k < end_k; k++) {
                    prosparsity_act[m * K + k] = current_row_ptr[k - start_k] - prefix_row_ptr[k - start_k];
                }
            }
        }
        __syncthreads();
    }
}


std::tuple<torch::Tensor, torch::Tensor> find_product_sparsity(
    torch::Tensor _input_act,
    int tile_size_m,
    int tile_size_k
)
{
    int M = _input_act.size(0);
    int K = _input_act.size(1);
    int num_tiles_m = (M + tile_size_m - 1) / tile_size_m;
    int num_tiles_k = (K + tile_size_k - 1) / tile_size_k;

    _input_act = _input_act.to(torch::kByte).to(torch::kCUDA);

    auto input_act = reinterpret_cast<uint8_t*>(_input_act.data_ptr<uint8_t>());
    at::Tensor _prosparsity_act = _input_act.clone();
    at::Tensor _prefix_array = torch::full({num_tiles_m, num_tiles_k, tile_size_m}, -1, torch::kInt32).to(torch::kCUDA);

    auto prosparsity_act = reinterpret_cast<uint8_t*>(_prosparsity_act.data_ptr<uint8_t>());
    auto prefix_array = reinterpret_cast<int*>(_prefix_array.data_ptr<int>());

    dim3 num_blocks(num_tiles_m, num_tiles_k);

    // We limit threads per block to 1024 (or less)
    int threads_per_block = 1024; 
    if (tile_size_m < threads_per_block) {
        threads_per_block = tile_size_m; // if tile_size_m <= 1024
    }

    // Compute shared memory size dynamically based on tile_size_m and tile_size_k
    size_t shared_memory_size = tile_size_m * sizeof(int) + 
                                (size_t)tile_size_m * tile_size_k * sizeof(uint8_t);

    if (shared_memory_size > 49152) {
        printf("Error: too much shared memory required\n");
        _prosparsity_act = _prosparsity_act.to(torch::kCPU);
        _prefix_array = _prefix_array.to(torch::kCPU);
        return std::make_tuple(_prosparsity_act, _prefix_array);
    }

    prosparsity_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
        input_act,
        prosparsity_act,
        prefix_array,
        M,
        K,
        tile_size_m,
        tile_size_k
    );

    cudaDeviceSynchronize();

    _prosparsity_act = _prosparsity_act.to(torch::kCPU);
    _prefix_array = _prefix_array.to(torch::kCPU);

    return std::make_tuple(_prosparsity_act, _prefix_array);
}


void find_product_sparsity_cpp(
    uint8_t* input_act,
    uint8_t* prosparsity_act,
    int* prefix_array,
    int M,
    int K,
    int tile_size_m,
    int tile_size_k
)
{
    int num_tiles_m = (M + tile_size_m - 1) / tile_size_m;
    int num_tiles_k = (K + tile_size_k - 1) / tile_size_k;

    // Allocate memory for the input_act on the device
    uint8_t* d_input_act;
    cudaMalloc(&d_input_act, M * K * sizeof(uint8_t));

    // Copy the input_act from the host (CPU) to the device (GPU)
    cudaMemcpy(d_input_act, input_act, M * K * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Allocate memory for the output prosparsity_act on the device
    uint8_t* d_prosparsity_act;
    cudaMalloc(&d_prosparsity_act, M * K * sizeof(uint8_t));

    // copy d_input_act to d_prosparsity_act
    cudaMemcpy(d_prosparsity_act, d_input_act, M * K * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    // Allocate memory for the prefix_array on the device
    int* d_prefix_array;
    cudaMalloc(&d_prefix_array, num_tiles_m * num_tiles_k * tile_size_m * sizeof(int));

    // Set prefix array to -1 on the device
    cudaMemset(d_prefix_array, -1, num_tiles_m * num_tiles_k * tile_size_m * sizeof(int));

    // Configure the kernel execution parameters
    dim3 num_blocks(num_tiles_m, num_tiles_k);
    dim3 threads_per_block(tile_size_m);

    int shared_memory_size = tile_size_m * sizeof(int) + tile_size_m * tile_size_k * sizeof(uint8_t);

    if (shared_memory_size > 49152)
    {
        printf("Error: too much shared memory required\n");
        cudaFree(d_input_act);
        cudaFree(d_prosparsity_act);
        cudaFree(d_prefix_array);
        return;
    }

    // Launch the kernel
    prosparsity_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
        d_input_act,
        d_prosparsity_act,
        d_prefix_array,
        M,
        K,
        tile_size_m,
        tile_size_k
    );

    // Check for errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input_act);
        cudaFree(d_prosparsity_act);
        cudaFree(d_prefix_array);
        return;
    }

    // move the prosparsity_act array from the GPU to the CPU

    cudaMemcpy(prosparsity_act, d_prosparsity_act, M * K * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(prefix_array, d_prefix_array, num_tiles_m * num_tiles_k * tile_size_m * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the input and prefix array memory on the GPU as they are no longer needed
    cudaFree(d_input_act);
    cudaFree(d_prefix_array);
    cudaFree(d_prosparsity_act);

    return;
}
