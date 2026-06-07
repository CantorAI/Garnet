#!/bin/bash
set -e

echo "WARNING: Garnet currently requires CUDA. macOS natively does not support CUDA."
echo "This build script assumes you have a decoupled CPU build target or an external GPU setup."

# Move to parent workspace folder (GarnetDev)
cd "$(dirname "$0")/../../../"

BUILD_DIR="out/build/macos-Debug"
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR/CMakeCache.txt"

cd "$BUILD_DIR"
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=bin -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=bin ../../../Garnet
ninja
