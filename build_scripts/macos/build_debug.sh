#!/bin/bash
set -e

echo "WARNING: Garnet currently requires CUDA. macOS natively does not support CUDA."
echo "This build script assumes you have a decoupled CPU build target or an external GPU setup."

# Move to Garnet root relative to this script
cd "$(dirname "$0")/../../"

BUILD_DIR="out/build/macos-Debug"
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR/CMakeCache.txt"

cd "$BUILD_DIR"
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug ../../../
ninja
