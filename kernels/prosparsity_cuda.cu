#include "prosparsity_cuda.hpp"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/extension.h>

#define TILE_SIZE_M 256
#define TILE_SIZE_K 16


__global__ void prosparsity_kernel(
    uint8_t *__restrict__ input_act,
    uint8_t *__restrict__ prosparsity_act,
    int *__restrict__ prefix_array,
    int M,
    int K
)
{

    int tile_m = blockIdx.x;
    int tile_k = blockIdx.y;

    int start_m = tile_m * TILE_SIZE_M;
    int start_k = tile_k * TILE_SIZE_K;

    int end_m = min(start_m + TILE_SIZE_M, M);
    int end_k = min(start_k + TILE_SIZE_K, K);

    int m = start_m + threadIdx.x;

    extern __shared__ char shared_mem[];

    int* nnz_array = (int*)shared_mem;
    uint8_t* act_tile = (uint8_t*)(shared_mem + TILE_SIZE_M * sizeof(int));

    nnz_array[threadIdx.x] = 0;
    // copy the input act to shared memory
    if (m < end_m)
    {
        for (int k = start_k; k < end_k; k++)
        {
            act_tile[threadIdx.x * TILE_SIZE_K + k - start_k] = input_act[m * K + k];
            nnz_array[threadIdx.x] += input_act[m * K + k];
        }
    }

    __syncthreads();
    bool is_subset;
    int max_subset = 0;
    int prefix = -1;

    if (m < end_m)
    {
        for (int i = start_m; i < end_m; i++)
        {
            is_subset = !(nnz_array[threadIdx.x] == nnz_array[i - start_m] && threadIdx.x <= i - start_m);
            for (int k = start_k; k < end_k; k++)
            {
                // is_subset &= input_act[m * K + k] >= input_act[i * K + k];
                is_subset &= act_tile[threadIdx.x * TILE_SIZE_K + k - start_k] >= act_tile[(i - start_m) * TILE_SIZE_K + k - start_k];
            }
            if (is_subset && nnz_array[i - start_m] > max_subset)
            {
                max_subset = nnz_array[i - start_m];
                prefix = i - start_m;
            }
        }
        
        if (nnz_array[threadIdx.x] < 2)
        {
            prefix = -1;
        }

        prefix_array[tile_m * gridDim.y * TILE_SIZE_M + tile_k * TILE_SIZE_M + m - start_m] = prefix;
        if (prefix != -1)
        {
            for (int k = start_k; k < end_k; k++)
            {
                // prosparsity_act[m * K + k] = input_act[m * K + k] - input_act[(prefix + start_m) * K + k];
                prosparsity_act[m * K + k] = act_tile[threadIdx.x * TILE_SIZE_K + k - start_k] - act_tile[prefix * TILE_SIZE_K + k - start_k];
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> find_product_sparsity(
    torch::Tensor _input_act
)
{
    int M = _input_act.size(0);
    int K = _input_act.size(1);
    int num_tiles_m = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;
    int num_tiles_k = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;

    // convert input act from bool to uint8_t
    _input_act = _input_act.to(torch::kByte);

    // move to device
    _input_act = _input_act.to(torch::kCUDA);



    auto input_act = reinterpret_cast<uint8_t*>(_input_act.data_ptr<uint8_t>());

    // clone the input_act tensor to prosparsity_act
    at::Tensor _prosparsity_act = _input_act.clone();


    at::Tensor _prefix_array = torch::ones({num_tiles_m, num_tiles_k, TILE_SIZE_M}, torch::kInt32).to(torch::kCUDA);

    _prefix_array = _prefix_array * -1;


    auto prosparsity_act = reinterpret_cast<uint8_t*>(_prosparsity_act.data_ptr<uint8_t>());
    auto prefix_array = reinterpret_cast<int*>(_prefix_array.data_ptr<int>());

    dim3 num_blocks(num_tiles_m, num_tiles_k);
    dim3 threads_per_block(TILE_SIZE_M);
    int shared_memory_size = TILE_SIZE_M * sizeof(int) + TILE_SIZE_M * TILE_SIZE_K * sizeof(uint8_t);

    if (shared_memory_size > 49152)
    {
        printf("Error: too much shared memory required\n");
        // move to CPU
        _prosparsity_act = _prosparsity_act.to(torch::kCPU);
        _prefix_array = _prefix_array.to(torch::kCPU);
        return std::make_tuple(_prosparsity_act, _prefix_array);
    }



    prosparsity_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
    // prosparsity_kernel<<<num_blocks, threads_per_block>>>(
        input_act,
        prosparsity_act,
        prefix_array,
        M,
        K
    );

    // input_act = input_act.to(torch::kCPU);
    _prosparsity_act = _prosparsity_act.to(torch::kCPU);
    _prefix_array = _prefix_array.to(torch::kCPU);

    return std::make_tuple(_prosparsity_act, _prefix_array);
}

void find_product_sparsity_cpp(
    uint8_t* input_act,
    uint8_t* prosparsity_act,
    int* prefix_array,
    int M,
    int K
)
{
    int num_tiles_m = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;
    int num_tiles_k = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;

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
    cudaMalloc(&d_prefix_array, num_tiles_m * num_tiles_k * TILE_SIZE_M * sizeof(int));

    // Set prefix array to -1 on the device
    cudaMemset(d_prefix_array, -1, num_tiles_m * num_tiles_k * TILE_SIZE_M * sizeof(int));

    // Configure the kernel execution parameters
    dim3 num_blocks(num_tiles_m, num_tiles_k);
    dim3 threads_per_block(TILE_SIZE_M);
    int shared_memory_size = TILE_SIZE_M * sizeof(int) + TILE_SIZE_M * TILE_SIZE_K * sizeof(uint8_t);

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
        K
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

    cudaMemcpy(prefix_array, d_prefix_array, num_tiles_m * num_tiles_k * TILE_SIZE_M * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the input and prefix array memory on the GPU as they are no longer needed
    cudaFree(d_input_act);
    cudaFree(d_prefix_array);
    cudaFree(d_prosparsity_act);

    return;
}
