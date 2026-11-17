#include <cstddef>
#ifndef KERN_H
#define KERN_H

void run_saxpy(size_t threads_per_block, size_t total_elements, long a, float *x, float *y);

#endif