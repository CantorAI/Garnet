# CMakeList.txt : CMake project for Garnet, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)
project ("garnet")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Function to find the libTorch directory
function(find_libtorch_dir var)
    # Start by looking in the current directory and move up the directory tree
    set(libtorch_found FALSE)
    set(search_dir "${CMAKE_CURRENT_SOURCE_DIR}")

    while(NOT libtorch_found AND NOT "${search_dir}" STREQUAL "")
        # Check for the appropriate build type directory within libTorch
        if(CMAKE_BUILD_TYPE MATCHES Debug)
            find_path(TORCH_DIR NAMES libTorch/Debug PATHS "${search_dir}" NO_DEFAULT_PATH)
        else()
            find_path(TORCH_DIR NAMES libTorch/Release PATHS "${search_dir}" NO_DEFAULT_PATH)
        endif()

        if(TORCH_DIR)
            set(libtorch_found TRUE)
        else()
            # Move up one directory
            get_filename_component(search_dir "${search_dir}" DIRECTORY)
        endif()
    endwhile()

    # If found, return the path, otherwise return an empty string
    if(libtorch_found)
        set(${var} "${TORCH_DIR}" PARENT_SCOPE)
    else()
        set(${var} "" PARENT_SCOPE)
    endif()
endfunction()

# Use the function to find libTorch
find_libtorch_dir(found_torch_dir)

if(NOT "${found_torch_dir}" STREQUAL "")
    # Append the specific build directory to the path
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(Torch_DIR "${found_torch_dir}/libTorch/Debug/share/cmake/Torch")
    else()
        set(Torch_DIR "${found_torch_dir}/libTorch/Release/share/cmake/Torch")
    endif()
    message(STATUS "Torch_DIR set to ${Torch_DIR}")
else()
    message(FATAL_ERROR "libTorch directory not found.")
endif()

find_package(Torch REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH=" ${CMAKE_MODULE_PATH} )
find_package(xlang REQUIRED)
set(xlang_sdk_folder  "${XLANG_INCLUDE_DIR}")
include_directories(${xlang_sdk_folder})

FILE(GLOB XLANG_API_SRC "${xlang_sdk_folder}/value.cpp")
FILE(GLOB Main_SRC "src/*.cpp")

add_library(${PROJECT_NAME} 
	SHARED
	${XLANG_API_SRC}
	${Main_SRC}
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
else()
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" pthread dl)
endif()

add_subdirectory ("test/TorchTest")