#pragma once
#include <vector>
#include <cstddef>

void run_saxpy(size_t threads_per_block, size_t total_elements, long a, float *x, float *y);
void run_copy_kernel_basic(size_t threads_per_block, std::vector<float> &x, std::vector<float> &y);
