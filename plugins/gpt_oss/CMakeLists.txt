cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(garnet_gpt_oss_ops LANGUAGES CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    enable_testing()
endif()
include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
set(GARNET_TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../ThirdPartySDK/TensorRT"
    CACHE PATH "TensorRT SDK directory")
add_library(garnet_gpt_oss_ops SHARED module/module.cpp tensorrt/gpt_oss_plugin.cpp
    cuda/gpt_oss_kernels.cu cuda/gpt_oss_gemv.cu cuda/gpt_oss_marlin.cu cuda/tp_collective.cpp
    cuda/tp_bf16.cu
    cuda/tp_direct.cu)
target_include_directories(garnet_gpt_oss_ops PRIVATE include "${GARNET_TENSORRT_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin")
target_compile_definitions(garnet_gpt_oss_ops PRIVATE NOMINMAX)
target_compile_options(garnet_gpt_oss_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
if(WIN32)
    target_link_libraries(garnet_gpt_oss_ops PRIVATE CUDA::cudart "${GARNET_TENSORRT_ROOT}/lib/nvinfer_10.lib")
else()
    find_library(GARNET_GPT_OSS_TRT_LIBRARY NAMES nvinfer
        PATHS "${GARNET_TENSORRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    target_link_libraries(garnet_gpt_oss_ops PRIVATE CUDA::cudart ${GARNET_GPT_OSS_TRT_LIBRARY} ${CMAKE_DL_LIBS})
    set_target_properties(garnet_gpt_oss_ops PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()
option(GARNET_GPT_OSS_ENABLE_NCCL "Enable GPT-OSS two-GPU tensor-parallel collectives" OFF)
if(GARNET_GPT_OSS_ENABLE_NCCL)
    if(WIN32)
        message(FATAL_ERROR "GPT-OSS NCCL tensor parallelism currently requires Linux")
    endif()
    find_path(GARNET_NCCL_INCLUDE_DIR nccl.h REQUIRED)
    find_library(GARNET_NCCL_LIBRARY NAMES nccl REQUIRED)
    target_include_directories(garnet_gpt_oss_ops PRIVATE "${GARNET_NCCL_INCLUDE_DIR}")
    target_link_libraries(garnet_gpt_oss_ops PRIVATE "${GARNET_NCCL_LIBRARY}")
    target_compile_definitions(garnet_gpt_oss_ops PRIVATE GARNET_GPT_OSS_ENABLE_NCCL=1)
    set_property(TARGET garnet_gpt_oss_ops APPEND PROPERTY BUILD_RPATH
        "/usr/lib/x86_64-linux-gnu")
    set_property(TARGET garnet_gpt_oss_ops APPEND PROPERTY INSTALL_RPATH
        "/usr/lib/x86_64-linux-gnu")
endif()
install(TARGETS garnet_gpt_oss_ops
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT garnet_gpt_oss_ops
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_gpt_oss_ops)
option(GARNET_GPT_OSS_BUILD_TESTS "Build GPT-OSS operator correctness tests" ON)
option(GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL "Build experimental native FlashInfer attention adapter" OFF)
if(GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL)
    if(CUDAToolkit_VERSION VERSION_LESS "12.9")
        message(FATAL_ERROR "The optional pinned FlashInfer prefill headers require CUDA Toolkit12.9 or newer")
    endif()
    target_sources(garnet_gpt_oss_ops PRIVATE cuda/gpt_oss_flash_prefill.cu)
    target_include_directories(garnet_gpt_oss_ops PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include")
    target_compile_definitions(garnet_gpt_oss_ops PRIVATE GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL=1)
endif()
if(GARNET_GPT_OSS_BUILD_TESTS)
    add_executable(garnet_gpt_oss_expert_weight_shard
        ../../test2026/gpt_oss/expert_weight_shard.cpp)
    target_include_directories(garnet_gpt_oss_expert_weight_shard PRIVATE
        ../../src/backends/tensorrt)
    add_test(NAME garnet_gpt_oss_expert_weight_shard COMMAND garnet_gpt_oss_expert_weight_shard)
    add_executable(garnet_gpt_oss_kernel_parity
        ../../test2026/gpt_oss/kernel_parity.cpp cuda/gpt_oss_kernels.cu
        cuda/gpt_oss_gemv.cu cuda/gpt_oss_marlin.cu)
    target_include_directories(garnet_gpt_oss_kernel_parity PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin" ../../src/backends/tensorrt)
    target_link_libraries(garnet_gpt_oss_kernel_parity PRIVATE CUDA::cudart)
    target_compile_definitions(garnet_gpt_oss_kernel_parity PRIVATE NOMINMAX GARNET_GPT_OSS_KERNEL_TEST)
    if(GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL)
        target_sources(garnet_gpt_oss_kernel_parity PRIVATE cuda/gpt_oss_flash_prefill.cu)
        target_include_directories(garnet_gpt_oss_kernel_parity PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include")
        target_compile_definitions(garnet_gpt_oss_kernel_parity PRIVATE GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL=1)
    endif()
    target_compile_options(garnet_gpt_oss_kernel_parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    add_test(NAME garnet_gpt_oss_kernel_parity COMMAND garnet_gpt_oss_kernel_parity)
    add_executable(garnet_gpt_oss_kernel_benchmark
        ../../test2026/gpt_oss/kernel_benchmark.cu cuda/gpt_oss_kernels.cu)
    target_include_directories(garnet_gpt_oss_kernel_benchmark PRIVATE include)
    target_link_libraries(garnet_gpt_oss_kernel_benchmark PRIVATE CUDA::cudart)
    target_compile_definitions(garnet_gpt_oss_kernel_benchmark PRIVATE NOMINMAX GARNET_GPT_OSS_KERNEL_TEST)
    add_executable(garnet_gpt_oss_attention_batch_benchmark
        ../../test2026/gpt_oss/attention_batch_benchmark.cu cuda/gpt_oss_kernels.cu)
    target_include_directories(garnet_gpt_oss_attention_batch_benchmark PRIVATE include)
    target_link_libraries(garnet_gpt_oss_attention_batch_benchmark PRIVATE CUDA::cudart)
    target_compile_definitions(garnet_gpt_oss_attention_batch_benchmark PRIVATE NOMINMAX)
    if(GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL)
        target_sources(garnet_gpt_oss_attention_batch_benchmark PRIVATE cuda/gpt_oss_flash_prefill.cu)
        target_include_directories(garnet_gpt_oss_attention_batch_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer/include")
        target_compile_definitions(garnet_gpt_oss_attention_batch_benchmark PRIVATE GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL=1)
        target_compile_options(garnet_gpt_oss_attention_batch_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()
    add_executable(garnet_gpt_oss_marlin_decode_benchmark
        ../../test2026/gpt_oss/marlin_decode_benchmark.cu
        cuda/gpt_oss_kernels.cu cuda/gpt_oss_marlin.cu)
    target_include_directories(garnet_gpt_oss_marlin_decode_benchmark PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/marlin")
    target_link_libraries(garnet_gpt_oss_marlin_decode_benchmark PRIVATE CUDA::cudart)
    target_compile_definitions(garnet_gpt_oss_marlin_decode_benchmark PRIVATE NOMINMAX)
    target_compile_options(garnet_gpt_oss_marlin_decode_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(GARNET_GPT_OSS_ENABLE_NCCL)
        add_executable(garnet_gpt_oss_tp_workspace_test ../../test2026/gpt_oss/tp_workspace.cpp)
        target_include_directories(garnet_gpt_oss_tp_workspace_test PRIVATE
            include tensorrt "${GARNET_TENSORRT_ROOT}/include")
        target_link_libraries(garnet_gpt_oss_tp_workspace_test PRIVATE garnet_gpt_oss_ops CUDA::cudart)
        add_test(NAME garnet_gpt_oss_tp_workspace_test COMMAND garnet_gpt_oss_tp_workspace_test)
        add_executable(garnet_gpt_oss_tp_collective_test
            ../../test2026/gpt_oss/tp_collective.cpp cuda/tp_collective.cpp
            cuda/tp_direct.cu cuda/tp_bf16.cu)
        target_include_directories(garnet_gpt_oss_tp_collective_test PRIVATE include
            "${GARNET_NCCL_INCLUDE_DIR}")
        target_link_libraries(garnet_gpt_oss_tp_collective_test PRIVATE CUDA::cudart "${GARNET_NCCL_LIBRARY}")
        target_compile_definitions(garnet_gpt_oss_tp_collective_test PRIVATE GARNET_GPT_OSS_ENABLE_NCCL=1)
        add_test(NAME garnet_gpt_oss_tp_collective_test COMMAND garnet_gpt_oss_tp_collective_test)
        add_executable(garnet_gpt_oss_tp_direct_benchmark
            ../../test2026/gpt_oss/tp_direct_benchmark.cu cuda/tp_collective.cpp
            cuda/tp_direct.cu cuda/tp_bf16.cu)
        target_include_directories(garnet_gpt_oss_tp_direct_benchmark PRIVATE include cuda
            "${GARNET_NCCL_INCLUDE_DIR}")
        target_link_libraries(garnet_gpt_oss_tp_direct_benchmark PRIVATE CUDA::cudart "${GARNET_NCCL_LIBRARY}")
        target_compile_definitions(garnet_gpt_oss_tp_direct_benchmark PRIVATE GARNET_GPT_OSS_ENABLE_NCCL=1)
    endif()
endif()

set_target_properties(garnet_gpt_oss_ops PROPERTIES OUTPUT_NAME garnet_gpt_oss)
target_include_directories(garnet_gpt_oss_ops PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../xlang3/sdk")
if(TARGET xlang3_runtime)
    target_link_libraries(garnet_gpt_oss_ops PRIVATE xlang3_runtime)
else()
    set(GARNET_XLANG_RUNTIME_LIBRARY "" CACHE FILEPATH "XLang3 runtime link library for standalone module build")
    if(NOT GARNET_XLANG_RUNTIME_LIBRARY)
        message(FATAL_ERROR "Standalone native module requires GARNET_XLANG_RUNTIME_LIBRARY")
    endif()
    target_link_libraries(garnet_gpt_oss_ops PRIVATE "${GARNET_XLANG_RUNTIME_LIBRARY}")
endif()
