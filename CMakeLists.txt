cmake_minimum_required(VERSION 4.0)

project(aschLLM
    VERSION 0.1.0
    DESCRIPTION "A custom LLM inference engine for learning"
    HOMEPAGE_URL "https://github.com/caschb/aschLLM"
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ASCHLLM_PTXAS_VERBOSE "Register info" OFF)

add_library(aschLLM_flags INTERFACE)

target_compile_options(aschLLM_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Wshadow>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${ASCHLLM_PTXAS_VERBOSE}>>:-Xptxas=-v>)

target_compile_options(aschLLM_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wshadow>)

target_include_directories(aschLLM_flags INTERFACE include)
include(FetchContent)
FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz
    FIND_PACKAGE_ARGS NAMES spdlog 
)
FetchContent_MakeAvailable(spdlog)
add_subdirectory(src)

add_subdirectory(bench)

if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
include(CTest)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
    FIND_PACKAGE_ARGS NAMES GTest
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)
endif()
