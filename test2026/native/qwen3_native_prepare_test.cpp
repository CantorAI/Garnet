#include "qwen3_native_decode_engine.h"
#include "safetensors_index.h"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: qwen3_native_prepare_test <checkpoint> "
                     "[packed-cache]\n";
        return 2;
    }
    std::string error;
    Garnet::SafeTensorsIndex index;
    if (!index.Open(argv[1], error)) {
        std::cerr << error << '\n';
        return 3;
    }
    const auto start = std::chrono::steady_clock::now();
    Garnet::Qwen3NativeDecodeEngine engine;
    const std::filesystem::path packedCache =
        argc == 3 ? argv[2] : std::filesystem::path();
    if (!engine.Prepare(index, packedCache, error)) {
        std::cerr << error << '\n';
        return 4;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "native Qwen3 prepare passed: packed_bytes="
              << engine.PackedBytes() << " seconds=" << seconds << '\n';
    return 0;
}
