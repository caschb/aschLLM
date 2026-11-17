#include <cuda_runtime.h>
#include <iostream>
#include <span>
#include <vector>

#include "aschllm/core/error_handling.h"
#include "aschllm/core/stream.h"
#include "aschllm/core/device_buffer.h"

__global__ void saxpy(size_t total_elements, long a, float *x, float *y)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}

void run_saxpy(size_t threads_per_block, size_t total_elements, long a, float *x, float *y)
{

    AschLLM::DeviceBuffer<float> x_dev(total_elements);
    AschLLM::DeviceBuffer<float> y_dev(total_elements);

    AschLLM::Stream stream;
    x_dev.copy_host_to_dev(std::span<float>{x, total_elements});
    y_dev.copy_host_to_dev(std::span<float>{y, total_elements});

    auto num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    saxpy<<<num_blocks, threads_per_block, 0, stream.get_cuda_stream()>>>(total_elements, a, x_dev.get_buffer(), y_dev.get_buffer());
    CUDA_CHECK_LAUNCH();

    y_dev.copy_dev_to_host(std::span<float>{y, total_elements});
}
