#include <cstddef>
#include <numeric>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "aschllm/core/device_buffer.h"

TEST(BufferTest, CorrectSizeAndLength)
{
    const AschLLM::DeviceBuffer<float> dev_buf{10};
    size_t length = dev_buf.get_length();
    size_t size = dev_buf.get_size();
    EXPECT_EQ(length, 10);
    EXPECT_EQ(size, sizeof(float) * 10);
}

TEST(BufferTest, AsyncRoundTrip)
{
    constexpr size_t capacity = 1000000;
    AschLLM::PinnedHostBuffer<float> first_pinned_buf{capacity};
    AschLLM::PinnedHostBuffer<float> second_pinned_buf{capacity};
    AschLLM::DeviceBuffer<float> dev_buf{capacity};
    AschLLM::Stream stream;

    std::iota(first_pinned_buf.get_buffer(), first_pinned_buf.get_buffer() + first_pinned_buf.get_length(), 12);
    std::fill_n(second_pinned_buf.get_buffer(), second_pinned_buf.get_length(), -1);
    dev_buf.copy_host_to_dev_async(first_pinned_buf, stream);
    dev_buf.copy_dev_to_host_async(second_pinned_buf, stream);
    stream.synchronize();
    EXPECT_THAT(second_pinned_buf.as_span(), testing::ElementsAreArray(first_pinned_buf.as_span()));
}