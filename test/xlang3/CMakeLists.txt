# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(garnet_tensor_tests LANGUAGES CXX)
endif()
enable_testing()
find_package(CUDAToolkit REQUIRED)

set(XLANG3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../xlang3" CACHE PATH "XLang3 repository")
set(XLANG3_RUNTIME_LIBRARY "" CACHE FILEPATH "Built XLang3 runtime import/shared library")
if(TARGET xlang3_runtime)
    set(GARNET_TEST_RUNTIME xlang3_runtime)
elseif(EXISTS "${XLANG3_RUNTIME_LIBRARY}")
    set(GARNET_TEST_RUNTIME "${XLANG3_RUNTIME_LIBRARY}")
else()
    message(FATAL_ERROR "Build xlang3_runtime first and set XLANG3_RUNTIME_LIBRARY")
endif()
add_library(garnet_tensor_capture STATIC
    ../../src/tensor/graph_capture.cpp ../../src/tensor/garnet_tensor.cpp
    ../../src/tensor/tensor_helper.cpp ../../src/model/compiled_graph_capture.cpp)
target_include_directories(garnet_tensor_capture PUBLIC
    ../../src/tensor "${XLANG3_ROOT}/sdk")
target_compile_features(garnet_tensor_capture PUBLIC cxx_std_17)
target_compile_definitions(garnet_tensor_capture PUBLIC XLANG3_USE_SHARED)
target_link_libraries(garnet_tensor_capture PUBLIC ${GARNET_TEST_RUNTIME} CUDA::cudart)
add_executable(garnet_tensor_tests tensor_tests.cpp)
target_link_libraries(garnet_tensor_tests PRIVATE garnet_tensor_capture)
add_test(NAME garnet_tensor_capture
    COMMAND garnet_tensor_tests "${CMAKE_CURRENT_SOURCE_DIR}/tensor_graph.py")
set_tests_properties(garnet_tensor_capture PROPERTIES TIMEOUT 60)
add_executable(garnet_tensor_readiness_tests tensor_readiness_tests.cpp)
target_link_libraries(garnet_tensor_readiness_tests PRIVATE garnet_tensor_capture)
add_test(NAME garnet_tensor_readiness COMMAND garnet_tensor_readiness_tests)
set_tests_properties(garnet_tensor_readiness PROPERTIES TIMEOUT 60)

if(EXISTS "${GARNET_TENSORRT_ROOT}/include/NvInfer.h")
    find_library(GARNET_TEST_TRT_LIBRARY NAMES nvinfer_10 nvinfer
        PATHS "${GARNET_TENSORRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    add_executable(garnet_trt_context_pool_tests trt_context_pool_tests.cpp)
    target_compile_features(garnet_trt_context_pool_tests PRIVATE cxx_std_17)
    target_include_directories(garnet_trt_context_pool_tests PRIVATE
        ../../src/backends/tensorrt "${GARNET_TENSORRT_ROOT}/include")
    target_link_libraries(garnet_trt_context_pool_tests PRIVATE
        "${GARNET_TEST_TRT_LIBRARY}" CUDA::cudart)
    if(WIN32)
        add_test(NAME garnet_trt_context_pool COMMAND "${CMAKE_COMMAND}" -E env
            "PATH=${GARNET_TENSORRT_ROOT}/bin;${CUDAToolkit_BIN_DIR};$ENV{PATH}"
            "$<TARGET_FILE:garnet_trt_context_pool_tests>")
    else()
        add_test(NAME garnet_trt_context_pool COMMAND garnet_trt_context_pool_tests)
    endif()
    set_tests_properties(garnet_trt_context_pool PROPERTIES TIMEOUT 120)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(test_library_path "${GARNET_TENSORRT_ROOT}/lib")
    if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        string(APPEND test_library_path ":$ENV{LD_LIBRARY_PATH}")
    endif()
    if(EXISTS "/usr/lib/wsl/lib/libcuda.so")
        string(PREPEND test_library_path "/usr/lib/wsl/lib:")
    endif()
    get_property(tensor_tests DIRECTORY PROPERTY TESTS)
    foreach(tensor_test IN LISTS tensor_tests)
        set_property(TEST "${tensor_test}" APPEND PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${test_library_path}")
    endforeach()
endif()
