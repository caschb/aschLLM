#include "aschllm/core/stream.h"
#include "aschllm/core/error_handling.h"

#include <cuda_runtime.h>

namespace AschLLM
{
    Stream::Stream()
    {
        CUDA_CHECK(cudaStreamCreate(&this->stream));
    }

    cudaStream_t Stream::get_cuda_stream() const
    {
        return this->stream;
    }

    Stream::~Stream()
    {
        if (this->stream)
        {
            CUDA_CHECK_NOTHROW(cudaStreamDestroy(this->stream));
        }
    }
}