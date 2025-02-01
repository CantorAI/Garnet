 # CMakeList.txt : CMake project for Garnet, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)
project ("garnet")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message("CMAKE_MODULE_PATH=" ${CMAKE_MODULE_PATH} )
find_package(xlang REQUIRED)
set(xlang_sdk_folder  "${XLANG_INCLUDE_DIR}")
include_directories(${xlang_sdk_folder})

include_directories("src/entry")
include_directories("src/include")
include_directories("src/model")

FILE(GLOB XLANG_API_SRC "${xlang_sdk_folder}/value.cpp")
FILE(GLOB Entry_SRC "src/entry/*.cpp")

add_library(${PROJECT_NAME} 
	SHARED
	${XLANG_API_SRC}
	${Entry_SRC}
)
if (WIN32)
    target_link_libraries(${PROJECT_NAME})
else()
    target_link_libraries(${PROJECT_NAME} pthread dl)
endif()

