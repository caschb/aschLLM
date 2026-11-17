#include <cuda_runtime.h>
#include <iostream>

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

    cudaMalloc(&x_dev, total_elements * sizeof(float));
    auto err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    cudaMalloc(&y_dev, total_elements * sizeof(float));

    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    cudaMemcpy(x_dev, x, total_elements, ::cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    cudaMemcpy(y_dev, y, total_elements, ::cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    size_t num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    saxpy<<<num_blocks, threads_per_block>>>(total_elements, a, x_dev, y_dev);
    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    cudaMemcpy(y, y_dev, total_elements, ::cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    cudaFree(x_dev);
    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';

    cudaFree(y_dev);
    err = cudaGetLastError();
    std::cout << cudaGetErrorName(err) << '\t' << cudaGetErrorString(err) << '\n';
}
