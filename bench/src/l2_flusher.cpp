#include <cstddef>

#include "l2_flusher.h"
#include "aschllm/core/error_handling.h"

namespace AschLLMBench
{
    void L2Flusher::flush_stream(const AschLLM::Stream &stream)
    {
        l2buffer.memset_async(10, stream);
    }
} // namespace AschLLMBench
