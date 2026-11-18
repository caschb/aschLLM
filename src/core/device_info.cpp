#include "aschllm/core/device_info.h"
#include "aschllm/core/error_handling.h"
#include <cuda_runtime.h>
#include <iostream>

namespace AschLLM
{
    DeviceInfo::DeviceInfo(int device_identifier)
    {
        this->device_id = device_identifier;
        CUDA_CHECK(cudaGetDeviceProperties(&this->properties, device_identifier));
        CUDA_CHECK(cudaDeviceGetAttribute(&this->clock_rate, cudaDeviceAttr::cudaDevAttrClockRate, device_identifier));
        CUDA_CHECK(cudaDeviceGetAttribute(&this->memory_clock_rate, cudaDeviceAttr::cudaDevAttrMemoryClockRate, device_identifier));
        CUDA_CHECK(cudaDeviceGetAttribute(&this->max_shared_memory_per_block, cudaDeviceAttr::cudaDevAttrMaxSharedMemoryPerBlock, device_identifier));
    }
    int DeviceInfo::getDeviceId() const
    {
        return this->device_id;
    }
    int DeviceInfo::getL2CacheSize() const
    {
        return this->properties.l2CacheSize;
    }
    int DeviceInfo::getMemoryBusWidth() const
    {
        return this->properties.memoryBusWidth;
    }
    int DeviceInfo::getSharedMultiprocessorCount() const
    {
        return this->properties.multiProcessorCount;
    }
    int DeviceInfo::getClockRate() const
    {
        return this->clock_rate;
    }
    int DeviceInfo::getMemoryClockRate() const
    {
        return this->memory_clock_rate;
    }
    int DeviceInfo::getMaxSharedMemoryPerBlock() const
    {
        return this->max_shared_memory_per_block;
    }
    std::string_view DeviceInfo::getDeviceName() const
    {
        auto name = this->properties.name;
        return std::string_view(name);
    }
} // namespace AschLLM
