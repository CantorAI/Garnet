cmake_minimum_required(VERSION 3.20)
project(garnet_pi5b_cpu_validation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
get_filename_component(GARNET_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_library(garnet_native_cpu STATIC
    "${GARNET_ROOT}/src/backends/cpu/gated_delta.cpp"
    "${GARNET_ROOT}/src/backends/cpu/q4q8_linear.cpp"
    "${GARNET_ROOT}/src/backends/cpu/qwen3_native_decode_engine.cpp"
    "${GARNET_ROOT}/src/model/weight_quantization.cpp"
    "${GARNET_ROOT}/src/model/safetensors_index.cpp")
target_include_directories(garnet_native_cpu PUBLIC
    "${GARNET_ROOT}/src/backends/cpu" "${GARNET_ROOT}/src/model"
    "${GARNET_ROOT}/third_party")
target_link_libraries(garnet_native_cpu PUBLIC Threads::Threads)
option(GARNET_PI5B_DOTPROD "Target the Pi 5 Cortex-A76 instruction set" OFF)
if(GARNET_PI5B_DOTPROD)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(FATAL_ERROR "Pi instruction target requires ARM64")
    endif()
    target_compile_options(garnet_native_cpu PRIVATE -mcpu=cortex-a76)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(MSVC)
        target_compile_options(garnet_native_cpu PRIVATE /arch:AVX2)
    else()
        target_compile_options(garnet_native_cpu PRIVATE -mavx2 -mf16c)
    endif()
endif()
add_executable(q4q8_arm_smoke q4q8_arm_smoke.cpp)
target_link_libraries(q4q8_arm_smoke PRIVATE garnet_native_cpu)
enable_testing()
add_test(NAME q4q8_reference COMMAND q4q8_arm_smoke)
add_executable(gated_delta_probe gated_delta_probe.cpp)
target_link_libraries(gated_delta_probe PRIVATE garnet_native_cpu)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME gated_delta_reference COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta.py" "$<TARGET_FILE:gated_delta_probe>")
endif()
find_package(OpenVINO QUIET COMPONENTS Runtime)
if(OpenVINO_FOUND)
    add_executable(linear_attention_probe linear_attention_probe.cpp)
    target_include_directories(linear_attention_probe PRIVATE
        "${GARNET_ROOT}/src/backends/openvino" "${GARNET_ROOT}/third_party")
    target_link_libraries(linear_attention_probe PRIVATE openvino::runtime)
    if(Python3_Interpreter_FOUND)
        add_test(NAME qwen35_linear_attention_reference COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/test_linear_attention.py" "$<TARGET_FILE:linear_attention_probe>")
    endif()
    add_executable(gated_delta_openvino_probe gated_delta_openvino_probe.cpp)
    target_include_directories(gated_delta_openvino_probe PRIVATE
        "${GARNET_ROOT}/src/backends/openvino" "${GARNET_ROOT}/third_party")
    target_link_libraries(gated_delta_openvino_probe PRIVATE openvino::runtime)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gated_delta_openvino_reference COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta.py" "$<TARGET_FILE:gated_delta_openvino_probe>")
    endif()
endif()
if(OpenVINO_FOUND)
    set(XLANG3_SDK_ROOT "${GARNET_ROOT}/../xlang3/sdk" CACHE PATH "XLang3 public SDK headers")
    add_library(garnet_openvino_lowering OBJECT "${GARNET_ROOT}/src/backends/openvino/openvino_builder.cpp")
    target_compile_definitions(garnet_openvino_lowering PUBLIC GARNET_CPU_ONLY=1 GARNET_WITH_OPENVINO=1)
    target_include_directories(garnet_openvino_lowering PUBLIC "${XLANG3_SDK_ROOT}"
        "${GARNET_ROOT}/src/backends/cpu" "${GARNET_ROOT}/src/backends/openvino"
        "${GARNET_ROOT}/src/model" "${GARNET_ROOT}/src/core"
        "${GARNET_ROOT}/src/tensor" "${GARNET_ROOT}/third_party")
    target_link_libraries(garnet_openvino_lowering PRIVATE openvino::runtime)
    set(XLANG3_RUNTIME_LIBRARY "" CACHE FILEPATH "Existing XLang3 runtime for graph execution validation")
    if(XLANG3_RUNTIME_LIBRARY)
        add_executable(graph_fixture_probe graph_fixture_probe.cpp
            "${GARNET_ROOT}/src/core/lowering_context.cpp"
            "${GARNET_ROOT}/src/tensor/graph_capture.cpp"
            "${GARNET_ROOT}/src/tensor/garnet_tensor.cpp"
            "${GARNET_ROOT}/src/model/compiled_graph_capture.cpp")
        target_link_libraries(graph_fixture_probe PRIVATE garnet_openvino_lowering
            garnet_native_cpu openvino::runtime "${XLANG3_RUNTIME_LIBRARY}")
        if(Python3_Interpreter_FOUND)
            add_test(NAME qwen35_linear_captured_checkpoint COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_LIST_DIR}/test_linear_graph.py" "$<TARGET_FILE:graph_fixture_probe>" "${GARNET_ROOT}")
            set_tests_properties(qwen35_linear_captured_checkpoint PROPERTIES ENVIRONMENT
                "GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f32;GARNET_OPENVINO_CPU_THREADS=2")
            add_test(NAME qwen35_attention_captured_checkpoint COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_LIST_DIR}/test_attention_graph.py" "$<TARGET_FILE:graph_fixture_probe>" "${GARNET_ROOT}")
            set_tests_properties(qwen35_attention_captured_checkpoint PROPERTIES ENVIRONMENT
                "GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f32;GARNET_OPENVINO_CPU_THREADS=2")
        endif()
        add_executable(gated_delta_graph_probe gated_delta_graph_probe.cpp
            "${GARNET_ROOT}/src/core/lowering_context.cpp"
            "${GARNET_ROOT}/src/tensor/graph_capture.cpp"
            "${GARNET_ROOT}/src/tensor/garnet_tensor.cpp"
            "${GARNET_ROOT}/src/model/compiled_graph_capture.cpp")
        target_link_libraries(gated_delta_graph_probe PRIVATE garnet_openvino_lowering
            garnet_native_cpu openvino::runtime "${XLANG3_RUNTIME_LIBRARY}")
        add_test(NAME gated_delta_xmodel_graph COMMAND gated_delta_graph_probe
            "${GARNET_ROOT}" "${CMAKE_CURRENT_LIST_DIR}/run_gated_delta_graph.py")
        set_tests_properties(gated_delta_xmodel_graph PROPERTIES ENVIRONMENT
            "GARNET_OPENVINO_CPU_INFERENCE_PRECISION=f32;GARNET_OPENVINO_CPU_THREADS=2")
    endif()
endif()
