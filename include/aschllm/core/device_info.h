#pragma once
#include <cuda_runtime.h>
#include <string_view>

namespace AschLLM
{
    class DeviceInfo
    {
    private:
        cudaDeviceProp properties;
        int device_id;
        int clock_rate;
        int memory_clock_rate;
        int max_shared_memory_per_block;

    public:
        DeviceInfo(int device_id);

        int getDeviceId() const;
        int getL2CacheSize() const;
        int getMemoryBusWidth() const;
        int getSharedMultiprocessorCount() const;
        int getClockRate() const;
        int getMemoryClockRate() const;
        int getMaxSharedMemoryPerBlock() const;
        std::string_view getDeviceName() const;
    };
}
