cmake_minimum_required(VERSION 3.18)
project("garnet" LANGUAGES CXX CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

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

include_directories(${CUDA_INCLUDE_DIRS})
# Include project directories.
include_directories("src/common")
include_directories("src/entry")
include_directories("src/compiler")
include_directories("src/include")
include_directories("src/model")
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
include_directories("src/trt")

# Optionally include xlang and Garnet includes.
include_directories(
    ${XLANG_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src/include
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/compiler
    ${PROJECT_SOURCE_DIR}/src/tensor
    ${PROJECT_SOURCE_DIR}/src/test
    ${PROJECT_SOURCE_DIR}/src/log
    ${PROJECT_SOURCE_DIR}/src/model
    ${PROJECT_SOURCE_DIR}/src/image
    ${PROJECT_SOURCE_DIR}/src/image/qwen_vl
    ${PROJECT_SOURCE_DIR}/src/tokenizer
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/src/trt
    "D:/CantorAI/ThirdPartySDK/TensorRT/include"
)

# Use absolute paths for source files.
FILE(GLOB XLANG_API_SRC "${xlang_sdk_folder}/value.cpp")
FILE(GLOB Common_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")
FILE(GLOB Entry_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/entry/*.cpp")
FILE(GLOB Model_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp")
FILE(GLOB Tensor_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
FILE(GLOB Log_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/log/*.cpp")
FILE(GLOB_RECURSE Image_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/image/*.cpp")
FILE(GLOB_RECURSE Tokenizer_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer/*.cpp")
FILE(GLOB_RECURSE Cuda_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
list(FILTER Cuda_SRC EXCLUDE REGEX ".*/test\\.cu$")
FILE(GLOB Trt_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/trt/*.cpp")

# Create the shared library target.
add_library(${PROJECT_NAME} 
    SHARED
    ${XLANG_API_SRC}
    ${Entry_SRC}
    ${Tensor_SRC}
    ${Cuda_SRC}
    ${Common_SRC}
    ${Model_SRC}
    ${Log_SRC}
    ${Image_SRC}
    ${Tokenizer_SRC}
    ${Trt_SRC}
)

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

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()


# On Windows, link against cudart (the CUDA runtime) explicitly.
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cuda_LIBRARY}
            nvrtc.lib cublas.lib "D:/CantorAI/ThirdPartySDK/TensorRT/lib/nvinfer_10.lib"
            nppc.lib nppig.lib nvjpeg.lib
            ${CUDA_CUDA_LIBRARY} # This specifically adds the cuda.lib driver library
    ) 
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
