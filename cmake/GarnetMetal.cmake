# Native backend foundation; Qwen graph integration is a separate milestone.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
find_library(GARNET_METAL_FRAMEWORK Metal REQUIRED)
find_library(GARNET_MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(GARNET_FOUNDATION_FRAMEWORK Foundation REQUIRED)
add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)
target_include_directories(garnet_metal PUBLIC src/backends/metal)
target_compile_options(garnet_metal PRIVATE -fobjc-arc)
target_link_libraries(garnet_metal PUBLIC
    "${GARNET_METAL_FRAMEWORK}" "${GARNET_MPS_FRAMEWORK}"
    "${GARNET_FOUNDATION_FRAMEWORK}")
enable_testing()
add_executable(garnet_metal_matmul_test test2026/metal/matmul_test.cpp)
target_link_libraries(garnet_metal_matmul_test PRIVATE garnet_metal)
add_test(NAME metal_mps_matmul_reference COMMAND garnet_metal_matmul_test)
