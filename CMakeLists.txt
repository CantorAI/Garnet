cmake_minimum_required(VERSION 3.18)
project("garnet" LANGUAGES CXX CUDA)

include(GNUInstallDirs)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)

# Ensure that CUDA and CXX code uses C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# For MSVC, make sure all code uses the dynamic runtime
if(MSVC)
    # This affects all targets
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    # Force all C/C++ code to use dynamic runtime
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MDd")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MD")
endif()

# Append our cmake modules path.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH=" ${CMAKE_MODULE_PATH})

# Find and include the xlang package.
find_package(xlang REQUIRED)
set(xlang_sdk_folder "${XLANG_INCLUDE_DIR}")
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
include_directories("${xlang_sdk_folder}/../Tensor")
include_directories("${xlang_sdk_folder}/../Core")
include_directories("src/backends/tensorrt")
include_directories("src/backends/openvino")
include_directories("src/backends/cpu")

# Optionally include xlang and Garnet includes.
include_directories(
    ${XLANG_INCLUDE_DIR}
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
    "D:/CantorAI/ThirdPartySDK/TensorRT/include"
)

# Use absolute paths for source files.
FILE(GLOB XLANG_API_SRC "${xlang_sdk_folder}/value.cpp")
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
    ${XLANG_API_SRC}
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

add_executable(garnet_qwen_vl_request_kv_native_smoke
    "${CMAKE_CURRENT_SOURCE_DIR}/test2026/native/qwen_vl_request_kv_native_smoke.cpp"
)
target_link_libraries(garnet_qwen_vl_request_kv_native_smoke ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

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
endif()

target_include_directories(garnet_weight_quantization_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model"
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()


# On Windows, link against cudart (the CUDA runtime) explicitly.
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cuda_LIBRARY}
            cublas.lib "D:/CantorAI/ThirdPartySDK/TensorRT/lib/nvinfer_10.lib"
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
    # Add debug-specific libraries for Windows Debug build
    if(MSVC)
        # Debug build
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG")
        
        # Use the debug runtime for debug builds
        target_link_libraries(${PROJECT_NAME} $<$<CONFIG:Debug>:msvcrtd>)
        target_link_libraries(${PROJECT_NAME} $<$<CONFIG:Release>:msvcrt>)
    endif()
else()
    target_link_libraries(${PROJECT_NAME} pthread dl m ${CUDA_LIBRARIES} ${CUDA_cuda_LIBRARY})
endif()

# Enable separable CUDA compilation and specify architectures.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;89"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# For MSVC, ensure consistent runtime settings for all source files
if(MSVC)
    # Apply to all source files directly
    foreach(src_file ${XLANG_API_SRC} ${Entry_SRC} ${Tensor_SRC})
        if(CMAKE_BUILD_TYPE MATCHES Debug)
            set_source_files_properties(${src_file} PROPERTIES COMPILE_FLAGS "/MDd")
        else()
            set_source_files_properties(${src_file} PROPERTIES COMPILE_FLAGS "/MD")
        endif()
    endforeach()
    
    # Set target properties for MSVC runtime
    set_target_properties(${PROJECT_NAME} PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
    
    # For debug builds, also link with the debug version of the C runtime
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_link_libraries(${PROJECT_NAME} debug msvcrtd)
    endif()
endif()

# copy the src/cuda folder to Lib/Garnet/cuda.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Lib/Garnet/cuda")
file(COPY "${PROJECT_SOURCE_DIR}/src/cuda" DESTINATION "${CMAKE_BINARY_DIR}/bin/Lib/Garnet/")

# Model programs are release assets, while checkpoint weights and compiled
# engines remain separately installed and machine-specific. Mirror only the
# production .x/.json contract into the build tree and install package.
option(GARNET_INSTALL_XMODELS "Install production Garnet xModel packages" ON)
set(GARNET_BUILD_MODEL_DIR "${CMAKE_BINARY_DIR}/models")
file(REMOVE_RECURSE "${GARNET_BUILD_MODEL_DIR}")
file(COPY "${PROJECT_SOURCE_DIR}/xModel/"
    DESTINATION "${GARNET_BUILD_MODEL_DIR}"
    FILES_MATCHING
        PATTERN "debug_*.x" EXCLUDE
        PATTERN "*.x"
        PATTERN "*.json"
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT garnet_runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT garnet_development
)
install(FILES "${PROJECT_SOURCE_DIR}/src/include/garnet_serving.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/garnet"
    COMPONENT garnet_development
)
if(GARNET_INSTALL_XMODELS)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/xModel/"
        DESTINATION "models"
        COMPONENT garnet_xmodels
        FILES_MATCHING
            PATTERN "debug_*.x" EXCLUDE
            PATTERN "*.x"
            PATTERN "*.json"
    )
endif()
