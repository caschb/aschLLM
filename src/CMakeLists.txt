find_package(CUDAToolkit)

add_library(aschLLM_lib STATIC
    core/error_handling.cpp core/stream.cpp core/device_info.cpp core/event.cpp)

option(ASCHLLM_SYNC_AFTER_LAUNCH "Sync after every kernel launch" OFF)

target_compile_definitions(aschLLM_lib PUBLIC
    $<$<BOOL:${ASCHLLM_SYNC_AFTER_LAUNCH}>:ASCHLLM_SYNC_AFTER_LAUNCH>)

add_executable(aschLLM main.cpp kern.cu)

target_link_libraries(aschLLM_lib PUBLIC aschLLM_flags CUDA::cudart spdlog::spdlog_header_only)
target_link_libraries(aschLLM PRIVATE aschLLM_lib)
