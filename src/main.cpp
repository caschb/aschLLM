#include "aschllm/kern.h"
#include <iostream>

auto main(int argc, char **argv) -> int
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <total_elements> <a>\n";
        exit(EXIT_FAILURE);
    }
    auto threads_per_block = atol(argv[1]);
    auto total_elements = atol(argv[2]);
    auto a = atol(argv[3]);

    auto *x = new float[total_elements];
    auto *y = new float[total_elements];

    for (auto i = 0l; i < total_elements; ++i)
    {
        x[i] = 10. * i;
        y[i] = 20. * i;
    }
    std::cout << argv[0] << ' ' << threads_per_block << ' ' << total_elements << ' ' << a << '\n';
    std::cout << "x=" << x[1] << '\t' << "y=" << y[1] << '\n';
    std::cout << x[1] * a + y[1] << '\n';

    run_saxpy(threads_per_block, total_elements, a, x, y);

    std::cout << y[1] << '\n';

    delete[] x;
    delete[] y;

    return EXIT_SUCCESS;
}