#include "safetensors_index.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <limits>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Garnet
{
    SafeTensorsMappedFile::~SafeTensorsMappedFile()
    {
        Close();
    }

    bool SafeTensorsMappedFile::Open(
        const std::filesystem::path& filePath,
        std::string& errorMessage)
    {
        Close();
#if defined(_WIN32)
        HANDLE file = CreateFileW(
            filePath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            errorMessage = "failed to open safetensors mapping file";
            return false;
        }
        LARGE_INTEGER fileSize{};
        if (!GetFileSizeEx(file, &fileSize) || fileSize.QuadPart <= 0) {
            CloseHandle(file);
            errorMessage = "failed to query safetensors mapping size";
            return false;
        }
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) {
            CloseHandle(file);
            errorMessage = "failed to create safetensors file mapping";
            return false;
        }
        const void* data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (!data) {
            CloseHandle(mapping);
            CloseHandle(file);
            errorMessage = "failed to map safetensors file";
            return false;
        }
        m_fileHandle = file;
        m_mappingHandle = mapping;
        m_data = static_cast<const unsigned char*>(data);
        m_size = static_cast<std::uint64_t>(fileSize.QuadPart);
#else
        const int file = open(filePath.c_str(), O_RDONLY);
        if (file < 0) {
            errorMessage = "failed to open safetensors mapping file";
            return false;
        }
        struct stat fileStatus{};
        if (fstat(file, &fileStatus) != 0 || fileStatus.st_size <= 0) {
            close(file);
            errorMessage = "failed to query safetensors mapping size";
            return false;
        }
        const void* data = mmap(
            nullptr,
            static_cast<size_t>(fileStatus.st_size),
            PROT_READ,
            MAP_PRIVATE,
            file,
            0);
        if (data == MAP_FAILED) {
            close(file);
            errorMessage = "failed to map safetensors file";
            return false;
        }
        m_fileHandle = reinterpret_cast<void*>(static_cast<intptr_t>(file + 1));
        m_data = static_cast<const unsigned char*>(data);
        m_size = static_cast<std::uint64_t>(fileStatus.st_size);
#endif
        errorMessage.clear();
        return true;
    }

    void SafeTensorsMappedFile::Close()
    {
#if defined(_WIN32)
        if (m_data) UnmapViewOfFile(m_data);
        if (m_mappingHandle) CloseHandle(static_cast<HANDLE>(m_mappingHandle));
        if (m_fileHandle) CloseHandle(static_cast<HANDLE>(m_fileHandle));
#else
        if (m_data) munmap(const_cast<unsigned char*>(m_data), static_cast<size_t>(m_size));
        if (m_fileHandle) {
            close(static_cast<int>(reinterpret_cast<intptr_t>(m_fileHandle)) - 1);
        }
#endif
        m_fileHandle = nullptr;
        m_mappingHandle = nullptr;
        m_data = nullptr;
        m_size = 0;
    }

    const void* SafeTensorsMappedFile::DataAt(
        std::uint64_t offset,
        std::uint64_t size) const
    {
        if (!m_data || offset > m_size || size > m_size - offset) {
            return nullptr;
        }
        return m_data + offset;
    }

    bool SafeTensorsIndex::Open(
        const std::filesystem::path& filePath,
        std::string& errorMessage)
    {
        m_filePath.clear();
        m_dataStart = 0;
        m_tensorBytes = 0;
        m_tensors.clear();

        std::error_code filesystemError;
        const std::uint64_t fileSize = std::filesystem::file_size(filePath, filesystemError);
        if (filesystemError || fileSize < sizeof(std::uint64_t)) {
            errorMessage = "safetensors file is missing or too small";
            return false;
        }

        std::ifstream stream(filePath, std::ios::binary);
        std::uint64_t headerSize = 0;
        stream.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));
        constexpr std::uint64_t kMaximumHeaderSize = 256ULL << 20;
        if (!stream || headerSize == 0 || headerSize > kMaximumHeaderSize ||
            headerSize > fileSize - sizeof(headerSize)) {
            errorMessage = "invalid safetensors header length";
            return false;
        }

        std::string header(static_cast<size_t>(headerSize), '\0');
        stream.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (!stream) {
            errorMessage = "failed to read safetensors header";
            return false;
        }

        nlohmann::json root;
        try {
            root = nlohmann::json::parse(header);
        }
        catch (const std::exception& exception) {
            errorMessage = std::string("invalid safetensors JSON header: ") + exception.what();
            return false;
        }
        if (!root.is_object()) {
            errorMessage = "safetensors header root must be an object";
            return false;
        }

        m_dataStart = sizeof(headerSize) + headerSize;
        for (auto iterator = root.begin(); iterator != root.end(); ++iterator) {
            if (iterator.key() == "__metadata__") {
                continue;
            }
            const auto& value = iterator.value();
            if (!value.is_object() || !value.contains("dtype") ||
                !value.contains("shape") || !value.contains("data_offsets") ||
                !value["dtype"].is_string() || !value["shape"].is_array() ||
                !value["data_offsets"].is_array() || value["data_offsets"].size() != 2) {
                errorMessage = "invalid safetensors tensor metadata: " + iterator.key();
                return false;
            }

            SafeTensorMetadata metadata;
            metadata.dataType = value["dtype"].get<std::string>();
            for (const auto& dimension : value["shape"]) {
                if (!dimension.is_number_integer() || dimension.get<long long>() < 0) {
                    errorMessage = "invalid safetensors shape: " + iterator.key();
                    return false;
                }
                metadata.shape.push_back(dimension.get<long long>());
            }
            const std::uint64_t begin = value["data_offsets"][0].get<std::uint64_t>();
            const std::uint64_t end = value["data_offsets"][1].get<std::uint64_t>();
            if (end < begin || end > fileSize - m_dataStart) {
                errorMessage = "safetensors data range exceeds file: " + iterator.key();
                return false;
            }
            metadata.dataOffset = m_dataStart + begin;
            metadata.dataSize = end - begin;
            if (metadata.dataSize > std::numeric_limits<std::uint64_t>::max() - m_tensorBytes) {
                errorMessage = "safetensors tensor byte count overflow";
                return false;
            }
            m_tensorBytes += metadata.dataSize;
            m_tensors.emplace(iterator.key(), std::move(metadata));
        }
        if (m_tensors.empty()) {
            errorMessage = "safetensors file contains no tensors";
            return false;
        }
        m_filePath = std::filesystem::absolute(filePath).lexically_normal();
        errorMessage.clear();
        return true;
    }

    const SafeTensorMetadata* SafeTensorsIndex::Find(const std::string& name) const
    {
        const auto found = m_tensors.find(name);
        return found == m_tensors.end() ? nullptr : &found->second;
    }
}
