#!/bin/bash
set -e

# Move to Garnet root relative to this script
cd "$(dirname "$0")/../../"

BUILD_DIR="out/build/linux-Release"
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR/CMakeCache.txt"

cd "$BUILD_DIR"
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ../../../
ninja
