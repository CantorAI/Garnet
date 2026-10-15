# CPU-only runtime uses the public XLang SDK and the actual Garnet backend.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(Threads REQUIRED)
set(XLANG3_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../xlang3/sdk" CACHE PATH "Public XLang3 SDK")
set(GARNET_XLANG3_RUNTIME_LIBRARY "" CACHE FILEPATH "Existing XLang3 shared runtime")
if(NOT EXISTS "${XLANG3_SDK_ROOT}/xlang3/xlang3.h")
    message(FATAL_ERROR "XLANG3_SDK_ROOT must point to the public SDK")
endif()
if(NOT EXISTS "${GARNET_XLANG3_RUNTIME_LIBRARY}")
    message(FATAL_ERROR "GARNET_XLANG3_RUNTIME_LIBRARY must name the existing shared runtime")
endif()
add_library(garnet SHARED
    src/entry/cpu_main.cpp src/backends/cpu/compiled_graph.cpp
    src/tensor/garnet_tensor.cpp src/tensor/graph_capture.cpp
    src/core/lowering_context.cpp src/model/compiled_graph_capture.cpp
    src/backends/openvino/openvino_builder.cpp src/backends/cpu/gated_delta.cpp
    src/backends/cpu/q4q8_linear.cpp src/backends/cpu/qwen3_native_decode_engine.cpp
    src/model/weight_quantization.cpp src/model/safetensors_index.cpp
    src/tokenizer/qwen_tokenizer.cpp)
target_include_directories(garnet PRIVATE "${XLANG3_SDK_ROOT}" src/tensor src/core
    src/model src/tokenizer src/backends/cpu src/backends/openvino third_party)
target_compile_definitions(garnet PRIVATE GARNET_CPU_ONLY=1 GARNET_WITH_OPENVINO=1 XLANG3_USE_SHARED=1)
target_link_libraries(garnet PRIVATE openvino::runtime Threads::Threads "${GARNET_XLANG3_RUNTIME_LIBRARY}")
option(GARNET_CPU_DOTPROD "Target Raspberry Pi 5 Cortex-A76 dot-product instructions" OFF)
if(GARNET_CPU_DOTPROD)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(FATAL_ERROR "GARNET_CPU_DOTPROD requires ARM64")
    endif()
    target_compile_options(garnet PRIVATE -mcpu=cortex-a76)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(MSVC)
        target_compile_options(garnet PRIVATE /arch:AVX2)
    else()
        target_compile_options(garnet PRIVATE -mavx2 -mf16c)
    endif()
endif()
include(GNUInstallDirs)
install(TARGETS garnet LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
