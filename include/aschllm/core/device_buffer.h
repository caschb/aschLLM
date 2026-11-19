#pragma once
#include <cuda_runtime.h>
#include <utility>
#include <span>
#include "aschllm/core/error_handling.h"
#include "aschllm/core/pinned_host_buffer.h"
#include "aschllm/core/stream.h"

namespace AschLLM
{
    template <class T>
    class DeviceBuffer
    {
    private:
        T *buffer_dev = nullptr;
        size_t length = 0;

    public:
        explicit DeviceBuffer(size_t length_) : length(length_)
        {
            CUDA_CHECK(cudaMalloc(&buffer_dev, sizeof(T) * length));
        }

        DeviceBuffer(const DeviceBuffer &other) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &other) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept : buffer_dev(std::exchange(other.buffer_dev, nullptr)), length(std::exchange(other.length, 0)) {}
        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            std::swap(buffer_dev, other.buffer_dev);
            std::swap(length, other.length);
            return *this;
        }
        ~DeviceBuffer()
        {
            if (buffer_dev)
                CUDA_CHECK_NOTHROW(cudaFree(buffer_dev));
        }

        T *get_buffer()
        {
            return buffer_dev;
        }

        const T *get_buffer() const
        {
            return buffer_dev;
        }

        size_t get_length() const
        {
            return length;
        }

        size_t get_size() const
        {
            return length * sizeof(T);
        }

        void copy_host_to_dev_async(const PinnedHostBuffer<T> &buffer_host, const Stream &stream) const
        {
            SIZE_CHECK(get_length(), buffer_host.get_length());
            CUDA_CHECK(cudaMemcpyAsync(buffer_dev, buffer_host.as_span().data(), get_size(), ::cudaMemcpyHostToDevice, stream.get_cuda_stream()));
        }

        void copy_dev_to_host_async(PinnedHostBuffer<T> &buffer_host, const Stream &stream)
        {
            SIZE_CHECK(get_length(), buffer_host.get_length());
            CUDA_CHECK(cudaMemcpyAsync(buffer_host.as_span().data(), buffer_dev, get_size(), ::cudaMemcpyDeviceToHost, stream.get_cuda_stream()));
        }

        void copy_host_to_dev(std::span<const T> buffer_host)
        {
            SIZE_CHECK(get_length(), buffer_host.size());
            CUDA_CHECK(cudaMemcpy(buffer_dev, buffer_host.data(), get_size(), ::cudaMemcpyHostToDevice));
        }

        void copy_dev_to_host(std::span<T> buffer_host) const
        {
            SIZE_CHECK(get_length(), buffer_host.size());
            CUDA_CHECK(cudaMemcpy(buffer_host.data(), buffer_dev, get_size(), ::cudaMemcpyDeviceToHost));
        }
    };
}