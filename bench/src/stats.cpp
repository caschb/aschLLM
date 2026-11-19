#include "stats.h"

#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>
#include <cmath>

namespace AschLLMBench
{
    Stats Stats::compute_stats(std::span<const Timer> timers)
    {
        // TODO: handle empty timers
        Stats stats;

        std::vector<float> durations(timers.size());
        std::transform(timers.begin(), timers.end(), durations.begin(), [](const Timer &t)
                       { return t.get_duration(); });
        std::sort(durations.begin(), durations.end());

        stats.count = durations.size();
        stats.maximum = durations.back();
        stats.minimum = durations.front();
        stats.mean = std::reduce(durations.begin(), durations.end()) / durations.size();
        stats.median = durations.size() % 2 ? durations[durations.size() / 2] : 0.5f * (durations[durations.size() / 2 - 1] + durations[durations.size() / 2]);
        float sv = std::transform_reduce(durations.begin(), durations.end(), 0.f, std::plus(), [m = stats.mean](float d)
                                         { return (d - m) * (d - m); });
        stats.stddev = durations.size() > 1 ? std::sqrt(sv / (durations.size() - 1)) : 0.f;
        stats.coeffient_of_variation = stats.stddev / stats.mean;
        return stats;
    }
} // namespace AschLLMBench
