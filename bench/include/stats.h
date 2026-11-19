#pragma once
#include <span>
#include <cstddef>
#include "timer.h"

namespace AschLLMBench
{
    struct Stats
    {
        float mean;
        float median;
        float stddev;
        float minimum;
        float maximum;
        float coeffient_of_variation;
        std::size_t count;

        Stats static compute_stats(std::span<const Timer> timers);
    };
} // namespace AschLLMBench
