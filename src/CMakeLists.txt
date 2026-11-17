add_library(aschLLM_lib STATIC
    kern.cu)

add_executable(aschLLM main.cpp)

target_link_libraries(aschLLM_lib PUBLIC aschLLM_flags)
target_link_libraries(aschLLM PRIVATE aschLLM_lib aschLLM_flags)