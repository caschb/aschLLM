#pragma once
#include <cstddef>

#include "aschllm/core/device_buffer.h"
#include "aschllm/core/stream.h"
#include "aschllm/core/device_info.h"

namespace AschLLMBench
{
    class L2Flusher
    {
    private:
        AschLLM::DeviceBuffer<std::byte> l2buffer;

    public:
        explicit L2Flusher(const AschLLM::DeviceInfo &info) : l2buffer{2 * info.getL2CacheSize()} {}

        void flush_stream(const AschLLM::Stream &stream);
    };
} // namespace AschLLMBench
