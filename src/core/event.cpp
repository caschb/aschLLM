#include "aschllm/core/event.h"
#include "aschllm/core/error_handling.h"

#include <cuda_runtime.h>

namespace AschLLM
{
    Event::Event()
    {
        CUDA_CHECK(cudaEventCreate(&this->event));
    }

    cudaEvent_t Event::get_cuda_event() const
    {
        return this->event;
    }

    Event::~Event()
    {
        if (this->event)
        {
            CUDA_CHECK_NOTHROW(cudaEventDestroy(this->event));
        }
    }
}