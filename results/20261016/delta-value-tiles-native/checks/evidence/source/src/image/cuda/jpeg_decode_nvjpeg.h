// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

namespace Garnet::Image::Cuda
{
    struct GpuImageRGB8
    {
        unsigned char* data = nullptr;
        int width = 0;
        int height = 0;
        int pitchBytes = 0;
        int deviceId = 0;
    };

    class NvJpegDecoder
    {
    public:
        NvJpegDecoder();
        ~NvJpegDecoder();

        bool Initialize(int cudaDevice = 0, std::string* error = nullptr);

        cudaError_t Decode(
            const unsigned char* jpegData,
            size_t jpegSize,
            cudaStream_t stream,
            GpuImageRGB8* result,
            std::string* error = nullptr);

        void Release(GpuImageRGB8* image);

    private:
        void* m_handle = nullptr;
        void* m_state = nullptr;
        int m_cudaDevice = 0;
        cudaEvent_t m_completion = nullptr;
        bool m_pending = false;
    };

    bool ReadFileBytes(const char* path, unsigned char** data, size_t* size, std::string* error = nullptr);

    cudaError_t DecodeJpegToDeviceRGB8(
        const unsigned char* jpegData,
        size_t jpegSize,
        cudaStream_t stream,
        GpuImageRGB8* result,
        std::string* error);

    // Releases the decoder owned by the calling inference thread. Call this
    // before unloading Garnet; nvJPEG teardown is not safe from a DLL/TLS
    // destructor during Windows process shutdown.
    void ShutdownThreadNvJpegDecoder();

    void FreeDecodedImage(GpuImageRGB8* result);
}
