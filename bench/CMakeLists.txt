add_library(aschLLMBench_lib src/l2_flusher.cpp src/timer.cpp)

target_include_directories(aschLLMBench_lib PUBLIC include)
target_link_libraries(aschLLMBench_lib PUBLIC aschLLM_lib)

add_executable(aschLLMBench bench.cpp)

target_link_libraries(aschLLMBench PRIVATE aschLLMBench_lib)