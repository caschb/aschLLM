#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstddef>

#define CUDA_CHECK(expr)                                       \
    do                                                         \
    {                                                          \
        cudaError_t e_ = (expr);                               \
        if (e_ != cudaSuccess)                                 \
        {                                                      \
            AschLLM::cuda_fail(e_, #expr, __FILE__, __LINE__); \
            std::abort();                                      \
        }                                                      \
    } while (0)

#define CUDA_CHECK_NOTHROW(expr)                               \
    do                                                         \
    {                                                          \
        cudaError_t e_ = (expr);                               \
        if (e_ != cudaSuccess)                                 \
        {                                                      \
            AschLLM::cuda_fail(e_, #expr, __FILE__, __LINE__); \
        }                                                      \
    } while (0)

#define CUDA_CHECK_LAUNCH()                       \
    do                                            \
    {                                             \
        CUDA_CHECK(cudaGetLastError());           \
        if constexpr (AschLLM::sync_after_launch) \
            CUDA_CHECK(cudaDeviceSynchronize());  \
    } while (0)

#define SIZE_CHECK(size1, size2)                                              \
    do                                                                        \
    {                                                                         \
        const auto s1_ = size1;                                               \
        const auto s2_ = size2;                                               \
        if (s1_ != s2_)                                                       \
        {                                                                     \
            AschLLM::size_fail(s1_, #size1, s2_, #size2, __FILE__, __LINE__); \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

namespace AschLLM
{
#ifdef ASCHLLM_SYNC_AFTER_LAUNCH
    inline constexpr bool sync_after_launch = true;
#else
    inline constexpr bool sync_after_launch = false;
#endif
    void cuda_fail(cudaError_t error, const char *expr, const char *file, int line);
    void size_fail(size_t first, const char *first_expr, size_t other, const char *other_expr, const char *file, int line);
}