# Build Garnet against a preserved XLang3 SDK runtime without rebuilding its VM.
cmake_minimum_required(VERSION 3.20)
project(garnet_qwen38_study LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT WIN32)
    message(FATAL_ERROR "This study wrapper currently describes native Windows dependencies")
endif()
set(XLANG3_RUNTIME_DIR "" CACHE PATH "Preserved XLang3 runtime DLL, import library and executable")
set(XLANG3_ROOT "" CACHE PATH "Matching XLang3 SDK/source root")
foreach(name xlang3_runtime.dll xlang3_runtime.lib xlang3.exe)
    if(NOT EXISTS "${XLANG3_RUNTIME_DIR}/${name}")
        message(FATAL_ERROR "Missing preserved runtime dependency: ${name}")
    endif()
endforeach()
if(NOT EXISTS "${XLANG3_ROOT}/sdk/xlang3/xlang3.h")
    message(FATAL_ERROR "XLANG3_ROOT must contain the matching public SDK")
endif()
add_library(xlang3_runtime SHARED IMPORTED GLOBAL)
set_target_properties(xlang3_runtime PROPERTIES
    IMPORTED_IMPLIB "${XLANG3_RUNTIME_DIR}/xlang3_runtime.lib"
    IMPORTED_LOCATION "${XLANG3_RUNTIME_DIR}/xlang3_runtime.dll")
add_executable(xlang3 IMPORTED GLOBAL)
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
file(COPY "${XLANG3_RUNTIME_DIR}/xlang3_runtime.dll" "${XLANG3_RUNTIME_DIR}/xlang3.exe"
    DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set_target_properties(xlang3 PROPERTIES IMPORTED_LOCATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/xlang3.exe")
get_filename_component(GARNET_STUDY_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${GARNET_STUDY_SOURCE}" "${CMAKE_BINARY_DIR}/garnet")
add_executable(garnet_nvfp4_projection_test nvfp4_projection_test.cu
    "${GARNET_STUDY_SOURCE}/src/cuda/modelopt_nvfp4.cu")
target_include_directories(garnet_nvfp4_projection_test PRIVATE "${GARNET_STUDY_SOURCE}/src/cuda")
target_link_libraries(garnet_nvfp4_projection_test PRIVATE CUDA::cudart)
set_target_properties(garnet_nvfp4_projection_test PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
add_test(NAME garnet_nvfp4_projection COMMAND garnet_nvfp4_projection_test)
add_executable(garnet_modelopt_trt_test modelopt_trt_test.cpp modelopt_trt_projection.cpp)
target_include_directories(garnet_modelopt_trt_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${GARNET_TENSORRT_ROOT}/include")
target_link_libraries(garnet_modelopt_trt_test PRIVATE CUDA::cudart "${GARNET_TENSORRT_ROOT}/lib/nvinfer_10.lib")
target_compile_definitions(garnet_modelopt_trt_test PRIVATE NOMINMAX)
target_compile_features(garnet_modelopt_trt_test PRIVATE cxx_std_17)
add_test(NAME garnet_modelopt_trt_projection COMMAND garnet_modelopt_trt_test)
add_executable(garnet_modelopt_cublaslt_test modelopt_cublaslt_test.cpp
    "${GARNET_STUDY_SOURCE}/src/backends/tensorrt/modelopt_cublaslt.cpp"
    "${GARNET_STUDY_SOURCE}/src/cuda/modelopt_nvfp4.cu")
target_include_directories(garnet_modelopt_cublaslt_test PRIVATE
    "${GARNET_STUDY_SOURCE}/src/backends/tensorrt" "${GARNET_STUDY_SOURCE}/src/cuda")
target_link_libraries(garnet_modelopt_cublaslt_test PRIVATE CUDA::cudart CUDA::cublasLt)
target_compile_features(garnet_modelopt_cublaslt_test PRIVATE cxx_std_17)
set_target_properties(garnet_modelopt_cublaslt_test PROPERTIES CUDA_STANDARD 17)
add_test(NAME garnet_modelopt_cublaslt_projection COMMAND garnet_modelopt_cublaslt_test)
add_executable(garnet_modelopt_plugin_test modelopt_plugin_test.cpp
    "${GARNET_STUDY_SOURCE}/src/backends/tensorrt/modelopt_nvfp4_plugin.cpp"
    "${GARNET_STUDY_SOURCE}/src/backends/tensorrt/modelopt_cublaslt.cpp"
    "${GARNET_STUDY_SOURCE}/src/cuda/modelopt_nvfp4.cu")
target_include_directories(garnet_modelopt_plugin_test PRIVATE "${GARNET_STUDY_SOURCE}/src/backends/tensorrt"
    "${GARNET_STUDY_SOURCE}/src/cuda" "${GARNET_TENSORRT_ROOT}/include")
target_link_libraries(garnet_modelopt_plugin_test PRIVATE CUDA::cudart CUDA::cublasLt "${GARNET_TENSORRT_ROOT}/lib/nvinfer_10.lib")
target_compile_features(garnet_modelopt_plugin_test PRIVATE cxx_std_17)
set_target_properties(garnet_modelopt_plugin_test PROPERTIES CUDA_STANDARD 17)
add_test(NAME garnet_modelopt_native_plugin COMMAND garnet_modelopt_plugin_test)
