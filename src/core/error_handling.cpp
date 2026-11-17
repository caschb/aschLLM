#include "aschllm/core/error_handling.h"
#include <iostream>

namespace AschLLM
{
    void cuda_fail(cudaError_t error, const char *expr, const char *file, int line)
    {
        auto error_name = cudaGetErrorName(error);
        auto error_message = cudaGetErrorString(error);

        std::cerr << file << ':' << line << ' ' << expr << ": " << error_name << ' ' << error_message << '\n';
    }
    void size_fail(size_t first, const char *first_expr, size_t other, const char *other_expr, const char *file, int line)
    {
        std::cerr << file << ':' << line << ' ' << "Sizes don't match: " << first_expr << " (" << first << ')' << '\t' << other_expr << " (" << other << ')' << '\n';
    }
}
