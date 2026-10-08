cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(garnet_gpt_oss_ops LANGUAGES CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    enable_testing()
endif()
find_package(CUDAToolkit REQUIRED)
option(GARNET_GPT_OSS_BUILD_XQA_PARITY "Build private XQA native parity test; inference remains unchanged" OFF)
if(GARNET_GPT_OSS_BUILD_XQA_PARITY)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(GARNET_XQA_UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer_xqa")
    set(GARNET_XQA_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/gpt_oss/prepare_xqa_sources.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${GARNET_XQA_GENERATOR}" "${GARNET_XQA_UPSTREAM}/SHA256SUMS")
    file(SHA256 "${GARNET_XQA_GENERATOR}" GARNET_XQA_GENERATOR_SHA)
    file(SHA256 "${GARNET_XQA_UPSTREAM}/SHA256SUMS" GARNET_XQA_UPSTREAM_SHA)
    string(SHA256 GARNET_XQA_GENERATED_KEY "${GARNET_XQA_GENERATOR_SHA}:${GARNET_XQA_UPSTREAM_SHA}")
    set(GARNET_XQA_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/xqa-${GARNET_XQA_GENERATED_KEY}")
    file(STRINGS "${GARNET_XQA_UPSTREAM}/SHA256SUMS" GARNET_XQA_MANIFEST_LINES)
    set(GARNET_XQA_DEPENDENCIES "${GARNET_XQA_UPSTREAM}/SHA256SUMS" "${GARNET_XQA_GENERATOR}")
    set(GARNET_XQA_GENERATED_FILES)
    foreach(GARNET_XQA_RECORD IN LISTS GARNET_XQA_MANIFEST_LINES)
        string(REGEX REPLACE "^[0-9a-f]+  " "" GARNET_XQA_FILE "${GARNET_XQA_RECORD}")
        list(APPEND GARNET_XQA_DEPENDENCIES "${GARNET_XQA_UPSTREAM}/${GARNET_XQA_FILE}")
        list(APPEND GARNET_XQA_GENERATED_FILES "${GARNET_XQA_GENERATED}/${GARNET_XQA_FILE}")
    endforeach()
    add_custom_command(OUTPUT "${GARNET_XQA_GENERATED}/garnet-adaptation.json"
        BYPRODUCTS ${GARNET_XQA_GENERATED_FILES}
        COMMAND "${Python3_EXECUTABLE}" "${GARNET_XQA_GENERATOR}" "${GARNET_XQA_UPSTREAM}" "${GARNET_XQA_GENERATED}"
        DEPENDS ${GARNET_XQA_DEPENDENCIES} VERBATIM)
    # Verification runs on every explicit native-test build. Existing damaged
    # evidence is rejected rather than silently regenerated or overwritten.
    add_custom_target(garnet_gpt_oss_xqa_generated_verified
        COMMAND "${Python3_EXECUTABLE}" "${GARNET_XQA_GENERATOR}" "${GARNET_XQA_UPSTREAM}" "${GARNET_XQA_GENERATED}" --verify-existing
        DEPENDS "${GARNET_XQA_GENERATED}/garnet-adaptation.json" VERBATIM)
    add_executable(garnet_gpt_oss_xqa_attention_parity
        ../../test2026/gpt_oss/xqa_attention_parity.cu
        cuda/gpt_oss_xqa_attention.cu cuda/gpt_oss_xqa_bridge.cu)
    add_dependencies(garnet_gpt_oss_xqa_attention_parity garnet_gpt_oss_xqa_generated_verified)
    target_include_directories(garnet_gpt_oss_xqa_attention_parity PRIVATE include cuda "${GARNET_XQA_GENERATED}")
    target_compile_options(garnet_gpt_oss_xqa_attention_parity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(garnet_gpt_oss_xqa_attention_parity PRIVATE NOMINMAX NDEBUG=1
        BEAM_WIDTH=1 HEAD_ELEMS=64 HEAD_GRP_SIZE=8 INPUT_FP16=0 CACHE_ELEM_ENUM=0
        TOKENS_PER_PAGE=16 SLIDING_WINDOW=1 LOW_PREC_OUTPUT=0 SPEC_DEC=0 USE_INPUT_KV=0
        USE_CUSTOM_BARRIER=1 USE_SM90_MHA=0)
    target_link_libraries(garnet_gpt_oss_xqa_attention_parity PRIVATE CUDA::cudart)
    add_test(NAME garnet_gpt_oss_xqa_attention_parity COMMAND garnet_gpt_oss_xqa_attention_parity)
endif()
include(GNUInstallDirs)
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
    add_executable(garnet_gpt_oss_xqa_page_contract ../../test2026/gpt_oss/xqa_page_contract.cpp)
    target_include_directories(garnet_gpt_oss_xqa_page_contract PRIVATE include)
    add_test(NAME garnet_gpt_oss_xqa_page_contract COMMAND garnet_gpt_oss_xqa_page_contract)
    add_executable(garnet_gpt_oss_router_dispatch ../../test2026/gpt_oss/router_dispatch.cpp)
    target_include_directories(garnet_gpt_oss_router_dispatch PRIVATE include)
    add_test(NAME garnet_gpt_oss_router_dispatch COMMAND garnet_gpt_oss_router_dispatch)
    add_executable(garnet_gpt_oss_marlin_chunks ../../test2026/gpt_oss/marlin_chunks.cpp)
    target_include_directories(garnet_gpt_oss_marlin_chunks PRIVATE include)
    add_test(NAME garnet_gpt_oss_marlin_chunks COMMAND garnet_gpt_oss_marlin_chunks)
    find_package(Threads REQUIRED)
    add_executable(garnet_gpt_oss_marlin_weight_pack ../../test2026/gpt_oss/marlin_weight_pack.cpp)
    target_include_directories(garnet_gpt_oss_marlin_weight_pack PRIVATE include)
    target_link_libraries(garnet_gpt_oss_marlin_weight_pack PRIVATE Threads::Threads)
    add_test(NAME garnet_gpt_oss_marlin_weight_pack COMMAND garnet_gpt_oss_marlin_weight_pack)
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
        add_executable(garnet_gpt_oss_tp_bf16_wire_benchmark
            ../../test2026/gpt_oss/tp_bf16_wire_benchmark.cu cuda/tp_collective.cpp
            cuda/tp_direct.cu cuda/tp_bf16.cu)
        target_include_directories(garnet_gpt_oss_tp_bf16_wire_benchmark PRIVATE include cuda
            "${GARNET_NCCL_INCLUDE_DIR}")
        target_link_libraries(garnet_gpt_oss_tp_bf16_wire_benchmark PRIVATE CUDA::cudart "${GARNET_NCCL_LIBRARY}")
        target_compile_definitions(garnet_gpt_oss_tp_bf16_wire_benchmark PRIVATE GARNET_GPT_OSS_ENABLE_NCCL=1)
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
