#include <cuda_runtime.h>
#include <iostream>
#include <span>
#include <vector>
#include <algorithm>
#include "spdlog/spdlog.h"

#include "aschllm/core/error_handling.h"
#include "aschllm/core/stream.h"
#include "aschllm/core/event.h"
#include "aschllm/core/device_buffer.h"
#include "aschllm/kern.h"

__global__ void saxpy(size_t total_elements, long a, float *x, float *y)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        y[idx] = a * x[idx] + y[idx];
    }
}

__global__ void copy_kernel_basic(size_t total_elements, float *src, float *dst)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        dst[idx] = src[idx];
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

void run_copy_kernel_basic(size_t threads_per_block, std::vector<float> &x, std::vector<float> &y)
{
    AschLLM::DeviceBuffer<float> x_dev(x.size());
    AschLLM::DeviceBuffer<float> y_dev(y.size());
    x_dev.copy_host_to_dev(x);

    auto num_blocks = (x.size() + threads_per_block - 1) / threads_per_block;
    AschLLM::Stream stream;

    AschLLM::Event start;
    AschLLM::Event end;

    cudaEventRecord(start.get_cuda_event(), stream.get_cuda_stream());
    copy_kernel_basic<<<num_blocks, threads_per_block, 0, stream.get_cuda_stream()>>>(x_dev.get_length(), x_dev.get_buffer(), y_dev.get_buffer());
    cudaEventRecord(end.get_cuda_event(), stream.get_cuda_stream());

    stream.synchronize();

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start.get_cuda_event(), end.get_cuda_event());

    spdlog::info("Kernel execution time: {} ms", elapsedTime);

    y_dev.copy_dev_to_host(y);

    auto msm = std::mismatch(x.begin(), x.end(), y.begin());
    if (msm.first != x.end())
    {
        spdlog::error("Mismatch! x: {}, y: {}", *msm.first, *msm.second);
    }
}
