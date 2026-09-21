# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project("garnet" LANGUAGES CXX CUDA)

include(GNUInstallDirs)
enable_testing()

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)

# Ensure that CUDA and CXX code uses C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Match the bundled native crypto/network archives. Runtime-owned XLang3
# objects cross the DLL boundary only through the SDK C ABI.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # Older parent projects can initialize compiler flags before CMP0091.
    # Keep NVCC's host runtime consistent with the C++ translation units too.
    foreach(language C CXX CUDA)
        foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            set(flag CMAKE_${language}_FLAGS_${configuration})
            string(REGEX REPLACE "([/-])MD" "\\1MT" ${flag} "${${flag}}")
        endforeach()
    endforeach()
endif()

# Append our cmake modules path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH=" ${CMAKE_MODULE_PATH})

# Build against the public XLang3 SDK, never old runtime implementation sources.
set(XLANG3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../xlang3" CACHE PATH "XLang3 repository")
if(NOT TARGET xlang3_runtime)
    add_subdirectory("${XLANG3_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/xlang3")
endif()
set(xlang_sdk_folder "${XLANG3_ROOT}/sdk")
set(GARNET_TENSORRT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ThirdPartySDK/TensorRT"
    CACHE PATH "TensorRT SDK directory")
include_directories(${xlang_sdk_folder})

# OpenVINO is an optional compiled backend. The Python wheel contains the
# native C++ SDK as well as the Python API, so discover its CMake package when
# a system/archive installation was not configured explicitly.
find_package(OpenVINO QUIET COMPONENTS Runtime)
if(NOT OpenVINO_FOUND)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                "import pathlib,openvino; print(pathlib.Path(openvino.__file__).parent/'cmake')"
            OUTPUT_VARIABLE GARNET_OPENVINO_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(GARNET_OPENVINO_CMAKE_DIR)
            set(OpenVINO_DIR "${GARNET_OPENVINO_CMAKE_DIR}")
            find_package(OpenVINO QUIET COMPONENTS Runtime)
        endif()
    endif()
endif()

include_directories(${CUDA_INCLUDE_DIRS})
# Include project directories.
include_directories("src/common")
include_directories("src/core")
include_directories("src/entry")
include_directories("src/include")
include_directories("src/model")
include_directories("src/runtime/kv")
include_directories("src/runtime/scheduler")
include_directories("src/tensor")
include_directories("src/test")
include_directories("src/log")
include_directories("src/cuda")
include_directories("src/image")
include_directories("src/image/qwen_vl")
include_directories("src/tokenizer")
include_directories("third_party")
include_directories("src/backends/tensorrt")
include_directories("src/backends/openvino")
include_directories("src/backends/cpu")

# Optionally include xlang and Garnet includes.
include_directories(
    ${xlang_sdk_folder}
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/tensor
    ${PROJECT_SOURCE_DIR}/src/test
    ${PROJECT_SOURCE_DIR}/src/log
    ${PROJECT_SOURCE_DIR}/src/model
    ${PROJECT_SOURCE_DIR}/src/runtime/kv
    ${PROJECT_SOURCE_DIR}/src/runtime/scheduler
    ${PROJECT_SOURCE_DIR}/src/image
    ${PROJECT_SOURCE_DIR}/src/image/qwen_vl
    ${PROJECT_SOURCE_DIR}/src/tokenizer
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/src/backends/tensorrt
    ${PROJECT_SOURCE_DIR}/src/backends/openvino
    ${PROJECT_SOURCE_DIR}/src/backends/cpu
    "${GARNET_TENSORRT_ROOT}/include"
)

# Use absolute paths for source files.
FILE(GLOB Common_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")
FILE(GLOB_RECURSE Core_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
FILE(GLOB Entry_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/entry/*.cpp")
FILE(GLOB Model_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp")
FILE(GLOB_RECURSE Runtime_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp")
FILE(GLOB Tensor_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
FILE(GLOB Log_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/log/*.cpp")
FILE(GLOB_RECURSE Image_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/image/*.cpp")
FILE(GLOB_RECURSE Tokenizer_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer/*.cpp")
FILE(GLOB_RECURSE Cuda_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
list(FILTER Cuda_SRC EXCLUDE REGEX ".*/test\\.cu$")
FILE(GLOB_RECURSE Backend_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/*.cpp")

# Create the shared library target.
add_library(${PROJECT_NAME} 
    SHARED
    ${Entry_SRC}
    ${Tensor_SRC}
    ${Cuda_SRC}
    ${Common_SRC}
    ${Core_SRC}
    ${Model_SRC}
    ${Runtime_SRC}
    ${Log_SRC}
    ${Image_SRC}
    ${Tokenizer_SRC}
    ${Backend_SRC}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE GARNET_SERVING_EXPORTS=1)
target_compile_definitions(${PROJECT_NAME} PRIVATE XLANG3_USE_SHARED)
target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
target_link_libraries(${PROJECT_NAME} xlang3_runtime)

if(OpenVINO_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE GARNET_WITH_OPENVINO=1)
    target_link_libraries(${PROJECT_NAME} openvino::runtime)
    message(STATUS "Garnet OpenVINO backend enabled: ${OpenVINO_VERSION}")

    # Keep the runtime and device plugins beside garnet.dll so loading Garnet
    # never depends on a user-specific Python DLL search path.
    get_filename_component(GARNET_OPENVINO_PACKAGE_DIR
        "${OpenVINO_DIR}/.." ABSOLUTE)
    file(GLOB GARNET_OPENVINO_DLLS
        "${GARNET_OPENVINO_PACKAGE_DIR}/libs/*.dll")
    foreach(GARNET_OPENVINO_DLL IN LISTS GARNET_OPENVINO_DLLS)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${GARNET_OPENVINO_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        )
    endforeach()
    install(FILES ${GARNET_OPENVINO_DLLS}
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT garnet_runtime
    )
else()
    message(STATUS "Garnet OpenVINO backend disabled (OpenVINO SDK not found)")
endif()

add_executable(garnet_qwen_vl_gpu_path_perf
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_gpu_path_perf.cpp"
)

add_executable(garnet_qwen_vl_prepare_prompt_perf
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_prepare_prompt_perf.cpp"
)

add_executable(garnet_device_kv_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/device_kv_native_smoke.cpp"
)
target_link_libraries(garnet_device_kv_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

add_executable(garnet_qwen_tokenizer_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_tokenizer_native_smoke.cpp"
)

add_executable(garnet_debug_logits_sampler_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/logits_sampler_native_smoke.cpp"
)
target_link_libraries(garnet_debug_logits_sampler_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

add_executable(garnet_repetition_sampler_test
    test2026/native/repetition_sampler_test.cpp
    src/cuda/logits_sampler_kernel.cu)
target_include_directories(garnet_repetition_sampler_test PRIVATE src/cuda)
target_link_libraries(garnet_repetition_sampler_test PRIVATE CUDA::cudart)
set_target_properties(garnet_repetition_sampler_test PROPERTIES
    CUDA_ARCHITECTURES "80;86;89")
add_test(NAME garnet_repetition_sampler COMMAND garnet_repetition_sampler_test)
set_tests_properties(garnet_repetition_sampler PROPERTIES TIMEOUT 120)

add_executable(garnet_qwen_vl_request_kv_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_request_kv_native_smoke.cpp"
)
target_link_libraries(garnet_qwen_vl_request_kv_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
if(CMAKE_DL_LIBS)
    foreach(native_tool IN ITEMS garnet_qwen_vl_gpu_path_perf garnet_qwen_vl_prepare_prompt_perf
            garnet_device_kv_native_smoke garnet_qwen_tokenizer_native_smoke
            garnet_debug_logits_sampler_native_smoke garnet_qwen_vl_request_kv_native_smoke)
        target_link_libraries(${native_tool} ${CMAKE_DL_LIBS})
    endforeach()
endif()

add_executable(garnet_continuous_batch_scheduler_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/continuous_batch_scheduler_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/kv/paged_kv_pool.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/scheduler/continuous_batch_scheduler.cpp"
)
target_include_directories(garnet_continuous_batch_scheduler_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/kv"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/scheduler"
)
target_link_libraries(garnet_continuous_batch_scheduler_test ${CUDA_LIBRARIES})

add_executable(garnet_weight_quantization_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/weight_quantization_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/weight_quantization.cpp"
)

add_executable(garnet_q4q8_linear_v2_benchmark
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/q4q8_linear_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
)

add_executable(garnet_q4q8_linear_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/q4q8_linear_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/weight_quantization.cpp"
)

add_executable(garnet_qwen3_native_prepare_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen3_native_prepare_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/qwen3_native_decode_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/weight_quantization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/safetensors_index.cpp"
)

add_executable(garnet_model_catalog_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/model_catalog_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/model_catalog.cpp"
)

target_include_directories(garnet_model_catalog_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

target_include_directories(garnet_q4q8_linear_v2_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

target_include_directories(garnet_q4q8_linear_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

target_include_directories(garnet_qwen3_native_prepare_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
    message(STATUS "Garnet native CPU INT4 OpenMP enabled")
endif()

if(MSVC)
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/qwen3_native_decode_engine.cpp"
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/q4q8_linear.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/qwen3_native_decode_engine.cpp"
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
    )
endif()

target_include_directories(garnet_weight_quantization_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

add_subdirectory(test/xlang3)
add_executable(garnet_image_tests
    test/xlang3/image_tests.cpp
    src/image/qwen_vl/qwen_vl_image_preprocessor.cpp
    src/image/qwen_vl/qwen_vl_vision_metadata.cpp
    src/image/cuda/jpeg_decode_nvjpeg.cpp
    src/cuda/qwen_vl_image_preprocess_kernel.cu
    src/cuda/qwen_vl_vision_metadata_kernel.cu)
target_link_libraries(garnet_image_tests PRIVATE garnet_tensor_capture
    CUDA::cudart CUDA::nvjpeg CUDA::nppc CUDA::nppig)
target_compile_definitions(garnet_image_tests PRIVATE NOMINMAX)
set_target_properties(garnet_image_tests PROPERTIES CUDA_ARCHITECTURES "80;86;89")
if(MSVC)
    target_compile_options(garnet_image_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/utf-8>)
endif()
add_test(NAME garnet_image_preprocessing COMMAND garnet_image_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/fixtures/vlm_screenshots/Q1.jpg")
set_tests_properties(garnet_image_preprocessing PROPERTIES TIMEOUT 120)
add_test(NAME garnet_weight_quantization COMMAND garnet_weight_quantization_test)
add_test(NAME garnet_q4q8_linear COMMAND garnet_q4q8_linear_test)
add_test(NAME garnet_model_catalog COMMAND garnet_model_catalog_test
    "${CMAKE_CURRENT_SOURCE_DIR}/xModel")
add_test(NAME garnet_continuous_batch_scheduler COMMAND garnet_continuous_batch_scheduler_test)
add_test(NAME garnet_device_kv_cabi COMMAND garnet_device_kv_native_smoke "$<TARGET_FILE:garnet>")
add_test(NAME garnet_logits_sampler_cabi COMMAND garnet_debug_logits_sampler_native_smoke "$<TARGET_FILE:garnet>")
add_test(NAME garnet_native_api COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/native_api.py")
set_tests_properties(garnet_native_api PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_production_capture COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/models/run_production_capture.py")
set_tests_properties(garnet_production_capture PROPERTIES TIMEOUT 120
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
add_test(NAME garnet_serving_python_defaults COMMAND "$<TARGET_FILE:xlang3>"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/serving_defaults.py")
set_tests_properties(garnet_serving_python_defaults PROPERTIES TIMEOUT 60
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:garnet>")
set_tests_properties(garnet_weight_quantization garnet_q4q8_linear
    garnet_model_catalog garnet_continuous_batch_scheduler
    garnet_device_kv_cabi garnet_logits_sampler_cabi PROPERTIES TIMEOUT 120)

find_program(GARNET_POWERSHELL NAMES powershell pwsh REQUIRED)
if(WIN32)
    set(GARNET_TENSORRT_RUNTIME_DIR "${GARNET_TENSORRT_ROOT}/bin")
else()
    set(GARNET_TENSORRT_RUNTIME_DIR "${GARNET_TENSORRT_ROOT}/lib")
endif()
    foreach(backend tensorrt openvino)
        if(backend STREQUAL "openvino" AND NOT OpenVINO_FOUND)
            continue()
        endif()
        add_test(NAME garnet_models_${backend}
            COMMAND "${GARNET_POWERSHELL}" -NoProfile -ExecutionPolicy Bypass
                -File "${CMAKE_CURRENT_SOURCE_DIR}/test/xlang3/run_model_tests.ps1"
                -RuntimePath "$<TARGET_FILE:xlang3>" -Backend "${backend}"
                -WorkRoot "${CMAKE_CURRENT_BINARY_DIR}/model-tests/${backend}"
                -TensorRTDirectory "${GARNET_TENSORRT_RUNTIME_DIR}"
                -CudaDirectory "${CUDAToolkit_BIN_DIR}")
        set_tests_properties(garnet_models_${backend} PROPERTIES TIMEOUT 300)
    endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(garnet_test_library_path "${GARNET_TENSORRT_ROOT}/lib")
    if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
        string(APPEND garnet_test_library_path ":$ENV{LD_LIBRARY_PATH}")
    endif()
    if(EXISTS "/usr/lib/wsl/lib/libcuda.so")
        string(PREPEND garnet_test_library_path "/usr/lib/wsl/lib:")
    endif()
    get_property(garnet_tests DIRECTORY PROPERTY TESTS)
    foreach(garnet_test IN LISTS garnet_tests)
        set_property(TEST "${garnet_test}" APPEND PROPERTY
            ENVIRONMENT "LD_LIBRARY_PATH=${garnet_test_library_path}")
    endforeach()
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor /utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/utf-8>
    )
endif()


# On Windows, link against cudart (the CUDA runtime) explicitly.
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cuda_LIBRARY}
            cublas.lib "${GARNET_TENSORRT_ROOT}/lib/nvinfer_10.lib"
            nppc.lib nppig.lib nvjpeg.lib
            ${CUDA_CUDA_LIBRARY} # This specifically adds the cuda.lib driver library
    ) 
    # Keep the multi-backend runtime loadable on machines without NVIDIA
    # software. These DLLs are resolved only if a CUDA/TensorRT operation is
    # actually invoked; OpenVINO CPU/Intel-GPU execution does not require them.
    if(MSVC)
        target_link_libraries(${PROJECT_NAME} delayimp)
        target_link_options(${PROJECT_NAME} PRIVATE
            "/DELAYLOAD:cublas64_13.dll"
            "/DELAYLOAD:nvinfer_10.dll"
            "/DELAYLOAD:nppig64_13.dll"
            "/DELAYLOAD:nvjpeg64_13.dll"
        )
    endif()

    # CUDA 13 installs several runtime DLLs under bin/x64. Deploy every
    # directly linked vendor runtime beside Garnet so applications and native
    # tests do not depend on a machine-wide PATH entry.
    set(GARNET_VENDOR_DLLS
        "${GARNET_TENSORRT_ROOT}/bin/nvinfer_10.dll")
    # TensorRT loads the GPU-architecture builder resource dynamically when a
    # model is compiled. Ship these beside Garnet so a clean installation can
    # generate its own engines on the local CUDA device.
    file(GLOB GARNET_TENSORRT_BUILDER_DLLS
        "${GARNET_TENSORRT_ROOT}/bin/nvinfer_builder_resource_*_10.dll")
    if(NOT GARNET_TENSORRT_BUILDER_DLLS)
        message(FATAL_ERROR "TensorRT builder resource DLLs were not found")
    endif()
    list(APPEND GARNET_VENDOR_DLLS ${GARNET_TENSORRT_BUILDER_DLLS})
    foreach(pattern cublas64_*.dll nppig64_*.dll nvjpeg64_*.dll)
        file(GLOB matches
            "${CUDAToolkit_BIN_DIR}/${pattern}"
            "${CUDAToolkit_BIN_DIR}/x64/${pattern}")
        if(NOT matches)
            message(FATAL_ERROR "Garnet runtime dependency not found: ${pattern}")
        endif()
        list(APPEND GARNET_VENDOR_DLLS ${matches})
    endforeach()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${GARNET_VENDOR_DLLS} "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND_EXPAND_LISTS VERBATIM)
else()
    find_library(GARNET_TENSORRT_LIBRARY NAMES nvinfer
        PATHS "${GARNET_TENSORRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    target_link_libraries(${PROJECT_NAME} pthread dl m
        "${GARNET_TENSORRT_LIBRARY}" CUDA::cudart CUDA::cuda_driver
        CUDA::cublas CUDA::nppc CUDA::nppig CUDA::nvjpeg)
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,--no-undefined")
endif()

# Enable separable CUDA compilation and specify architectures.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89"
    RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>"
    LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>"
    ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>"
)

# copy the src/cuda folder to Lib/Garnet/cuda.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Lib/Garnet/cuda")
file(COPY "${PROJECT_SOURCE_DIR}/src/cuda" DESTINATION "${CMAKE_BINARY_DIR}/bin/Lib/Garnet/")

# Model programs are release assets, while checkpoint weights and compiled
# engines remain separately installed and machine-specific. Mirror only the
# production model programs into the build tree and install package.
option(GARNET_INSTALL_XMODELS "Install production Garnet xModel packages" ON)
set(GARNET_BUILD_MODEL_DIR "${CMAKE_BINARY_DIR}/models")
file(REMOVE_RECURSE "${GARNET_BUILD_MODEL_DIR}")
file(COPY "${PROJECT_SOURCE_DIR}/xModel/"
    DESTINATION "${GARNET_BUILD_MODEL_DIR}"
    FILES_MATCHING
        PATTERN "debug_*.py" EXCLUDE
        PATTERN "*.py"
        PATTERN "*.json"
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT garnet_runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_development
)
if(GARNET_INSTALL_XMODELS)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/xModel/"
        DESTINATION "models"
        COMPONENT garnet_xmodels
        FILES_MATCHING
            PATTERN "debug_*.py" EXCLUDE
            PATTERN "*.py"
            PATTERN "*.json"
    )
endif()
