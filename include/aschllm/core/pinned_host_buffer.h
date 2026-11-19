#pragma once
#include <cuda_runtime.h>
#include <utility>
#include <span>
#include "aschllm/core/error_handling.h"

namespace AschLLM
{
    template <class T>
    class PinnedHostBuffer
    {
    private:
        T *buffer_pinned_host = nullptr;
        size_t length = 0;

    public:
        explicit PinnedHostBuffer(size_t length_) : length(length_)
        {
            CUDA_CHECK(cudaMallocHost(&buffer_pinned_host, sizeof(T) * length));
        }

        PinnedHostBuffer(const PinnedHostBuffer &other) = delete;
        PinnedHostBuffer &operator=(const PinnedHostBuffer &other) = delete;

        PinnedHostBuffer(PinnedHostBuffer &&other) noexcept : buffer_pinned_host(std::exchange(other.buffer_pinned_host, nullptr)), length(std::exchange(other.length, 0)) {}
        PinnedHostBuffer &operator=(PinnedHostBuffer &&other) noexcept
        {
            std::swap(buffer_pinned_host, other.buffer_pinned_host);
            std::swap(length, other.length);
            return *this;
        }
        ~PinnedHostBuffer()
        {
            if (buffer_pinned_host)
                CUDA_CHECK_NOTHROW(cudaFreeHost(buffer_pinned_host));
        }

        std::span<T> as_span()
        {
            return std::span<T>{buffer_pinned_host, length};
        }

        std::span<const T> as_span() const
        {
            return std::span<const T>{buffer_pinned_host, length};
        }

        T *get_buffer()
        {
            return buffer_pinned_host;
        }

        const T *get_buffer() const
        {
            return buffer_pinned_host;
        }

        size_t get_length() const
        {
            return length;
        }

        size_t get_size() const
        {
            return length * sizeof(T);
        }
    };
}