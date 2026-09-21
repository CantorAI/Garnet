# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(garnet_cpp_example LANGUAGES CXX)

set(XLANG3_ROOT "" CACHE PATH "Path to an XLang3 source checkout")
if(NOT EXISTS "${XLANG3_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "Set -DXLANG3_ROOT=<path-to-xlang3>")
endif()

if(NOT TARGET xlang3_runtime)
    add_subdirectory("${XLANG3_ROOT}" "${CMAKE_BINARY_DIR}/xlang3" EXCLUDE_FROM_ALL)
endif()

add_executable(garnet_cpp_example main.cpp)
target_compile_features(garnet_cpp_example PRIVATE cxx_std_17)
target_link_libraries(garnet_cpp_example PRIVATE xlang3_runtime)
