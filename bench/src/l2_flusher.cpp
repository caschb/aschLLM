#include <cstddef>

#include "l2_flusher.h"
#include "aschllm/core/error_handling.h"

namespace AschLLMBench
{
    void L2Flusher::flush_stream(const AschLLM::Stream &stream)
    {
        CUDA_CHECK(cudaMemsetAsync(l2buffer.get_buffer(), 10, l2buffer.get_size(), stream.get_cuda_stream()));
    }
} // namespace AschLLMBench
