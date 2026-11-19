#pragma once
#include <cuda_runtime.h>
#include <utility>

namespace AschLLM
{
    class Event
    {
    private:
        cudaEvent_t event = nullptr;

    public:
        Event();

        Event(unsigned int flags);

        Event(const Event &) = delete;
        Event &operator=(const Event &) = delete;

        Event(Event &&other) noexcept : event(std::exchange(other.event, nullptr)) {}
        Event &operator=(Event &&other) noexcept
        {
            std::swap(event, other.event);
            return *this;
        }

        cudaEvent_t get_cuda_event() const;
        ~Event();
    };
}