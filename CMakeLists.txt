cmake_minimum_required(VERSION 3.18)
project("garnet" LANGUAGES CXX CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Ensure that CUDA code uses the shared (dynamic) runtime.
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

# Use absolute paths for source files.
FILE(GLOB XLANG_API_SRC "${xlang_sdk_folder}/value.cpp")
FILE(GLOB Common_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp")
FILE(GLOB Entry_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/entry/*.cpp")
FILE(GLOB Model_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp")
FILE(GLOB Tensor_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
FILE(GLOB Log_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/log/*.cpp")
FILE(GLOB Cuda_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
FILE(GLOB Test_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/test/cudalib_test.cpp")

# Create the shared library target.
add_library(${PROJECT_NAME} 
    SHARED
    ${XLANG_API_SRC}
    ${Entry_SRC}
    ${Tensor_SRC}
    ${Cuda_SRC}
    ${Test_SRC}
    ${Common_SRC}
    ${Model_SRC}
    ${Log_SRC}
)


# On Windows, link against cudart (the CUDA runtime) explicitly.
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_cuda_LIBRARY}
            nvrtc.lib
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

