#pragma once
#include <cuda_runtime.h>
#include <utility>

namespace AschLLM
{
    class Stream
    {
    private:
        cudaStream_t stream = nullptr;

    public:
        Stream();

        Stream(const Stream &) = delete;
        Stream &operator=(const Stream &) = delete;

        Stream(Stream &&other) noexcept : stream(std::exchange(other.stream, nullptr)) {}
        Stream &operator=(Stream &&other) noexcept
        {
            std::swap(stream, other.stream);
            return *this;
        }

        cudaStream_t get_cuda_stream() const;
        void synchronize();
        ~Stream();
    };
}