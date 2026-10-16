// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace Garnet
{
    class SafeTensorsMappedFile
    {
        void* m_fileHandle = nullptr;
        void* m_mappingHandle = nullptr;
        const unsigned char* m_data = nullptr;
        std::uint64_t m_size = 0;

    public:
        SafeTensorsMappedFile() = default;
        ~SafeTensorsMappedFile();
        SafeTensorsMappedFile(const SafeTensorsMappedFile&) = delete;
        SafeTensorsMappedFile& operator=(const SafeTensorsMappedFile&) = delete;

        bool Open(const std::filesystem::path& filePath, std::string& errorMessage);
        void Close();
        const void* DataAt(std::uint64_t offset, std::uint64_t size) const;
        bool IsOpen() const { return m_data != nullptr; }
    };

    struct SafeTensorMetadata
    {
        std::string dataType;
        std::vector<long long> shape;
        std::filesystem::path filePath;
        std::uint64_t dataOffset = 0;
        std::uint64_t dataSize = 0;
    };

    class SafeTensorsIndex
    {
        std::filesystem::path m_filePath;
        std::uint64_t m_dataStart = 0;
        std::uint64_t m_tensorBytes = 0;
        std::unordered_map<std::string, SafeTensorMetadata> m_tensors;

    public:
        // Accepts a single .safetensors file, a Hugging Face
        // model.safetensors.index.json, or a directory containing either.
        bool Open(const std::filesystem::path& filePath, std::string& errorMessage);

        const std::filesystem::path& FilePath() const { return m_filePath; }
        std::uint64_t DataStart() const { return m_dataStart; }
        std::uint64_t TensorBytes() const { return m_tensorBytes; }
        size_t TensorCount() const { return m_tensors.size(); }
        const std::unordered_map<std::string, SafeTensorMetadata>& Entries() const { return m_tensors; }
        const SafeTensorMetadata* Find(const std::string& name) const;
    };
}
