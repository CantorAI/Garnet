#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using EncodeFn = int (*)(
    const char* modelDir,
    const char* text,
    long long* outputIds,
    int outputCapacity,
    int* outputCount,
    char* errorMessage,
    int errorMessageCapacity);

using DecodeFn = int (*)(
    const char* modelDir,
    const long long* tokenIds,
    int tokenCount,
    int skipSpecialTokens,
    char* outputText,
    int outputTextCapacity,
    int* outputByteCount,
    char* errorMessage,
    int errorMessageCapacity);

using TokenIdFn = long long (*)(
    const char* modelDir,
    const char* token);

namespace
{
    std::string DefaultModelDir()
    {
        return "D:\\CantorAI\\xWorld\\models\\hf_models\\qwen2.5-0.5b";
    }

    std::string DefaultDllPath()
    {
        return "D:\\CantorAI\\Garnet\\out\\build\\x64-Debug\\bin\\garnet.dll";
    }
}

int main(int argc, char** argv)
{
    std::string modelDir = argc > 1 ? argv[1] : DefaultModelDir();
    std::string dllPath = argc > 2 ? argv[2] : DefaultDllPath();
    const char* text = "Describe visible objects.";

    HMODULE dll = LoadLibraryA(dllPath.c_str());
    if (!dll) {
        std::cerr << "failed to load " << dllPath << ", error=" << GetLastError() << "\n";
        return 1;
    }

    auto encode = reinterpret_cast<EncodeFn>(GetProcAddress(dll, "GarnetQwenTokenizerEncode"));
    auto decode = reinterpret_cast<DecodeFn>(GetProcAddress(dll, "GarnetQwenTokenizerDecode"));
    auto tokenId = reinterpret_cast<TokenIdFn>(GetProcAddress(dll, "GarnetQwenTokenizerTokenId"));
    if (!encode || !decode || !tokenId) {
        std::cerr << "missing tokenizer exports\n";
        FreeLibrary(dll);
        return 2;
    }

    char error[2048] = {};
    std::vector<long long> ids(256);
    int count = 0;
    int rc = encode(modelDir.c_str(), text, ids.data(), static_cast<int>(ids.size()), &count, error, static_cast<int>(sizeof(error)));
    if (rc != 0 || count <= 0) {
        std::cerr << "encode failed rc=" << rc << " count=" << count << " error=" << error << "\n";
        FreeLibrary(dll);
        return 3;
    }
    ids.resize(static_cast<size_t>(count));

    std::vector<char> decoded(4096);
    int byteCount = 0;
    rc = decode(
        modelDir.c_str(),
        ids.data(),
        count,
        1,
        decoded.data(),
        static_cast<int>(decoded.size()),
        &byteCount,
        error,
        static_cast<int>(sizeof(error)));
    if (rc != 0 || byteCount <= 0) {
        std::cerr << "decode failed rc=" << rc << " byte_count=" << byteCount << " error=" << error << "\n";
        FreeLibrary(dll);
        return 4;
    }

    long long imagePadId = tokenId(modelDir.c_str(), "<|image_pad|>");
    if (imagePadId < 0) {
        std::cerr << "missing Qwen image pad token\n";
        FreeLibrary(dll);
        return 5;
    }

    std::cout << "Garnet Qwen tokenizer native smoke passed\n";
    std::cout << "input_text: " << text << "\n";
    std::cout << "token_count: " << count << "\n";
    std::cout << "decoded_text: " << decoded.data() << "\n";
    std::cout << "image_pad_id: " << imagePadId << "\n";

    FreeLibrary(dll);
    return 0;
}
