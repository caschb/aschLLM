#include "timer.h"
#include "aschllm/core/error_handling.h"
#include <cuda_runtime.h>

namespace AschLLMBench
{
    void Timer::start_timer(const AschLLM::Stream &stream) const
    {
        CUDA_CHECK(cudaEventRecord(start.get_cuda_event(), stream.get_cuda_stream()));
    }

    void Timer::stop_timer(const AschLLM::Stream &stream) const
    {
        CUDA_CHECK(cudaEventRecord(stop.get_cuda_event(), stream.get_cuda_stream()));
    }

    float Timer::get_duration() const
    {
        float duration;
        CUDA_CHECK(cudaEventElapsedTime(&duration, start.get_cuda_event(), stop.get_cuda_event()));
        return duration;
    }

} // namespace AschLLMBench
