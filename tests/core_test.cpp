#include <cstddef>
#include <gtest/gtest.h>
#include "aschllm/core/device_buffer.h"

class DeviceBufferTest : public testing::Test
{
protected:
    const AschLLM::DeviceBuffer<float> buf{10};
};

TEST_F(DeviceBufferTest, CorrectLength)
{
    size_t length = buf.get_length();
    EXPECT_EQ(length, 10);
}

TEST_F(DeviceBufferTest, CorrectSize)
{
    size_t size = buf.get_size();
    EXPECT_EQ(size, sizeof(float) * 10);
}