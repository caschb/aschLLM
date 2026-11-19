#pragma once
#include "aschllm/core/event.h"
#include "aschllm/core/stream.h"
#include <utility>

namespace AschLLMBench
{
    class Timer
    {
    private:
        AschLLM::Event start;
        AschLLM::Event stop;

    public:
        void start_timer(const AschLLM::Stream &stream) const;
        void stop_timer(const AschLLM::Stream &stream) const;
        float get_duration() const;

    };
} // namespace AschLLMBench
