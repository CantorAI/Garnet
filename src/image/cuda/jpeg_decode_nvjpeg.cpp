#include "jpeg_decode_nvjpeg.h"

#include <nvjpeg.h>
#include <fstream>

namespace Garnet::Image::Cuda
{
    namespace
    {
        thread_local NvJpegDecoder* g_threadDecoder = nullptr;

        cudaError_t NvJpegToCuda(nvjpegStatus_t status)
        {
            return status == NVJPEG_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
        }

        void SetError(std::string* error, const std::string& message)
        {
            if (error) {
                *error = message;
            }
        }
    }

    NvJpegDecoder::NvJpegDecoder() = default;

    NvJpegDecoder::~NvJpegDecoder()
    {
        if (m_state) {
            nvjpegJpegStateDestroy(reinterpret_cast<nvjpegJpegState_t>(m_state));
            m_state = nullptr;
        }
        if (m_handle) {
            nvjpegDestroy(reinterpret_cast<nvjpegHandle_t>(m_handle));
            m_handle = nullptr;
        }
    }

    bool NvJpegDecoder::Initialize(int cudaDevice, std::string* error)
    {
        m_cudaDevice = cudaDevice;
        if (m_handle && m_state) {
            return true;
        }

        int oldDevice = 0;
        cudaGetDevice(&oldDevice);
        if (oldDevice != m_cudaDevice) {
            cudaSetDevice(m_cudaDevice);
        }

        nvjpegHandle_t handle = nullptr;
        nvjpegJpegState_t state = nullptr;
        nvjpegStatus_t nvStatus = nvjpegCreateSimple(&handle);
        if (nvStatus == NVJPEG_STATUS_SUCCESS) {
            nvStatus = nvjpegJpegStateCreate(handle, &state);
        }

        if (oldDevice != m_cudaDevice) {
            cudaSetDevice(oldDevice);
        }

        if (nvStatus != NVJPEG_STATUS_SUCCESS) {
            if (state) {
                nvjpegJpegStateDestroy(state);
            }
            if (handle) {
                nvjpegDestroy(handle);
            }
            SetError(error, "failed to initialize nvJPEG decoder");
            return false;
        }

        m_handle = handle;
        m_state = state;
        return true;
    }

    cudaError_t NvJpegDecoder::Decode(
        const unsigned char* jpegData,
        size_t jpegSize,
        cudaStream_t stream,
        GpuImageRGB8* result,
        std::string* error)
    {
        if (!jpegData || jpegSize == 0 || !result) {
            SetError(error, "invalid nvJPEG decode arguments");
            return cudaErrorInvalidValue;
        }
        if (!Initialize(m_cudaDevice, error)) {
            return cudaErrorUnknown;
        }

        auto handle = reinterpret_cast<nvjpegHandle_t>(m_handle);
        auto state = reinterpret_cast<nvjpegJpegState_t>(m_state);
        int componentCount = 0;
        nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
        int widths[NVJPEG_MAX_COMPONENT] = {};
        int heights[NVJPEG_MAX_COMPONENT] = {};
        nvjpegStatus_t nvStatus = nvjpegGetImageInfo(handle, jpegData, jpegSize, &componentCount, &subsampling, widths, heights);
        if (nvStatus != NVJPEG_STATUS_SUCCESS || widths[0] <= 0 || heights[0] <= 0) {
            SetError(error, "nvjpegGetImageInfo failed");
            return NvJpegToCuda(nvStatus);
        }

        result->width = widths[0];
        result->height = heights[0];
        result->pitchBytes = result->width * 3;
        size_t outputBytes = static_cast<size_t>(result->pitchBytes) * static_cast<size_t>(result->height);

        int oldDevice = 0;
        cudaGetDevice(&oldDevice);
        if (oldDevice != m_cudaDevice) {
            cudaSetDevice(m_cudaDevice);
        }

        cudaError_t cudaStatus = cudaMalloc(&result->data, outputBytes);
        if (oldDevice != m_cudaDevice) {
            cudaSetDevice(oldDevice);
        }
        if (cudaStatus != cudaSuccess) {
            SetError(error, "cudaMalloc failed for decoded RGB image");
            return cudaStatus;
        }

        nvjpegImage_t destination{};
        destination.channel[0] = result->data;
        destination.pitch[0] = static_cast<unsigned int>(result->pitchBytes);
        nvStatus = nvjpegDecode(handle, state, jpegData, jpegSize, NVJPEG_OUTPUT_RGBI, &destination, stream);
        if (nvStatus != NVJPEG_STATUS_SUCCESS) {
            cudaFree(result->data);
            result->data = nullptr;
            SetError(error, "nvjpegDecode failed");
            return NvJpegToCuda(nvStatus);
        }

        return cudaSuccess;
    }

    void NvJpegDecoder::Release(GpuImageRGB8* image)
    {
        FreeDecodedImage(image);
    }

    bool ReadFileBytes(const char* path, unsigned char** data, size_t* size, std::string* error)
    {
        if (!path || !data || !size) {
            SetError(error, "invalid file read arguments");
            return false;
        }
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            SetError(error, "failed to open JPEG file");
            return false;
        }
        std::streamsize fileSize = file.tellg();
        if (fileSize <= 0) {
            SetError(error, "JPEG file is empty");
            return false;
        }
        file.seekg(0, std::ios::beg);
        unsigned char* buffer = new unsigned char[static_cast<size_t>(fileSize)];
        if (!file.read(reinterpret_cast<char*>(buffer), fileSize)) {
            delete[] buffer;
            SetError(error, "failed to read JPEG file");
            return false;
        }
        *data = buffer;
        *size = static_cast<size_t>(fileSize);
        return true;
    }

    cudaError_t DecodeJpegToDeviceRGB8(
        const unsigned char* jpegData,
        size_t jpegSize,
        cudaStream_t stream,
        GpuImageRGB8* result,
        std::string* error)
    {
        if (!g_threadDecoder) {
            g_threadDecoder = new NvJpegDecoder();
        }
        return g_threadDecoder->Decode(jpegData, jpegSize, stream, result, error);
    }

    void ShutdownThreadNvJpegDecoder()
    {
        delete g_threadDecoder;
        g_threadDecoder = nullptr;
    }

    void FreeDecodedImage(GpuImageRGB8* result)
    {
        if (!result) {
            return;
        }
        if (result->data) {
            cudaFree(result->data);
            result->data = nullptr;
        }
        result->width = 0;
        result->height = 0;
        result->pitchBytes = 0;
    }
}
