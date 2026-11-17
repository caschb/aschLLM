#include <cuda_runtime.h>

#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#define CUDA_CHECK(expr)                                       \
    do                                                         \
    {                                                          \
        cudaError_t e_ = (expr);                               \
        if (e_ != cudaSuccess)                                 \
            AschLLM::cuda_fail(e_, #expr, __FILE__, __LINE__); \
    } while (0)

#define CUDA_CHECK_LAUNCH()                       \
    do                                            \
    {                                             \
        CUDA_CHECK(cudaGetLastError());           \
        if constexpr (AschLLM::sync_after_launch) \
            CUDA_CHECK(cudaDeviceSynchronize());  \
    } while (0)

namespace AschLLM
{
#ifdef ASCHLLM_SYNC_AFTER_LAUNCH
    inline constexpr bool sync_after_launch = true;
#else
    inline constexpr bool sync_after_launch = false;
#endif

    void cuda_fail(cudaError_t error, const char *expr, const char *file, int line);
}

#endif