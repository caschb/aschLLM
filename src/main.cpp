#include "aschllm/kern.h"
#include "aschllm/core/device_info.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include "spdlog/spdlog.h"

auto main(int argc, char **argv) -> int
{
    if (argc != 3)
    {
        spdlog::error("Usage {} <threads_per_block> <total_elements>", argv[0]);
        exit(EXIT_FAILURE);
    }
    auto threads_per_block = atol(argv[1]);
    auto total_elements = atol(argv[2]);

    std::vector<float> x(total_elements);
    std::vector<float> y(total_elements, 0.);

    AschLLM::DeviceInfo dev{0};

    spdlog::info("GPU#0 {}", dev.getDeviceName());
    spdlog::info("GPU#0 Clock Rate {} kHz", dev.getClockRate());
    spdlog::info("GPU#0 L2 Cache Size {} B", dev.getL2CacheSize());
    spdlog::info("GPU#0 Max Shared Memory Per Block {}", dev.getMaxSharedMemoryPerBlock());
    spdlog::info("GPU#0 Memory Bus Width {}", dev.getMemoryBusWidth());
    spdlog::info("GPU#0 Memory Clock Rate {}", dev.getMemoryClockRate());
    spdlog::info("GPU#0 Shared Multiprocessor Count {}", dev.getSharedMultiprocessorCount());
    spdlog::info("GPU#0 Async Engine Count {}", dev.getAsyncEngineCount());
    spdlog::debug("Running with: threads_per_block={}, total_elements={}", threads_per_block, total_elements);

    for (auto i = 0l; i < total_elements; ++i)
    {
        x[i] = 10. * (i + 1);
    }

    run_copy_kernel_basic(threads_per_block, x, y);

    return EXIT_SUCCESS;
}