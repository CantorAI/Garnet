#!/bin/bash
set -e

# Move to parent workspace folder (GarnetDev)
cd "$(dirname "$0")/../../../"

BUILD_DIR="out/build/linux-Debug"
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR/CMakeCache.txt"

cd "$BUILD_DIR"
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=bin -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=bin ../../../Garnet
ninja
