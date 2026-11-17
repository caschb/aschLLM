#include <cuda_runtime.h>
#include <iostream>
#include "aschllm/core/error_handling.h"

__global__ void saxpy(size_t total_elements, long a, float *x, float *y)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}

void run_saxpy(size_t threads_per_block, size_t total_elements, long a, float *x, float *y)
{
    float *x_dev;
    float *y_dev;

    CUDA_CHECK(cudaMalloc(&x_dev, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&y_dev, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(x_dev, x, total_elements, ::cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(y_dev, y, total_elements, ::cudaMemcpyHostToDevice));

    size_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    saxpy<<<num_blocks, threads_per_block>>>(total_elements, a, x_dev, y_dev);
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaMemcpy(y, y_dev, total_elements, ::cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_dev));

    CUDA_CHECK(cudaFree(y_dev));
}
