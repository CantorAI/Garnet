#include "jpeg_decode_nvjpeg.h"

#include <nvjpeg.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>

namespace Garnet::Image::Cuda
{
    namespace
    {
        thread_local std::map<int, NvJpegDecoder*> g_threadDecoders;

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
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != m_cudaDevice && cudaSetDevice(m_cudaDevice) != cudaSuccess) return;
        if (m_pending) cudaEventSynchronize(m_completion);
        if (m_state) {
            nvjpegJpegStateDestroy(reinterpret_cast<nvjpegJpegState_t>(m_state));
            m_state = nullptr;
        }
        if (m_handle) {
            nvjpegDestroy(reinterpret_cast<nvjpegHandle_t>(m_handle));
            m_handle = nullptr;
        }
        if (m_completion) cudaEventDestroy(m_completion);
        if (previous != m_cudaDevice) cudaSetDevice(previous);
    }

    bool NvJpegDecoder::Initialize(int cudaDevice, std::string* error)
    {
        if (m_handle && m_state) {
            if (m_cudaDevice == cudaDevice) return true;
            SetError(error, "nvJPEG decoder belongs to another CUDA device");
            return false;
        }

        int oldDevice = 0;
        if (cudaGetDevice(&oldDevice) != cudaSuccess ||
            (oldDevice != cudaDevice && cudaSetDevice(cudaDevice) != cudaSuccess)) {
            SetError(error, "cannot select nvJPEG CUDA device");
            return false;
        }
        m_cudaDevice = cudaDevice;

        nvjpegHandle_t handle = nullptr;
        nvjpegJpegState_t state = nullptr;
        nvjpegStatus_t nvStatus = nvjpegCreateSimple(&handle);
        if (nvStatus == NVJPEG_STATUS_SUCCESS) {
            nvStatus = nvjpegJpegStateCreate(handle, &state);
        }

        cudaError_t eventStatus = cudaSuccess;
        if (nvStatus == NVJPEG_STATUS_SUCCESS)
            eventStatus = cudaEventCreateWithFlags(&m_completion, cudaEventDisableTiming);
        if (nvStatus != NVJPEG_STATUS_SUCCESS || eventStatus != cudaSuccess) {
            if (state) {
                nvjpegJpegStateDestroy(state);
            }
            if (handle) {
                nvjpegDestroy(handle);
            }
            SetError(error, "failed to initialize nvJPEG decoder");
            if (oldDevice != m_cudaDevice) cudaSetDevice(oldDevice);
            return false;
        }

        m_handle = handle;
        m_state = state;
        if (oldDevice != m_cudaDevice) cudaSetDevice(oldDevice);
        return true;
    }

    cudaError_t NvJpegDecoder::Decode(
        const unsigned char* jpegData,
        size_t jpegSize,
        cudaStream_t stream,
        GpuImageRGB8* result,
        std::string* error)
    {
        if (!jpegData || jpegSize == 0 || !result || result->data) {
            SetError(error, "invalid nvJPEG decode arguments");
            return cudaErrorInvalidValue;
        }
        int currentDevice = 0;
        auto deviceStatus = cudaGetDevice(&currentDevice);
        if (deviceStatus != cudaSuccess) return deviceStatus;
        if (!Initialize(currentDevice, error)) {
            return cudaErrorUnknown;
        }
        if (m_pending) {
            auto status = cudaEventSynchronize(m_completion);
            if (status != cudaSuccess) return status;
            m_pending = false;
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
            return nvStatus == NVJPEG_STATUS_SUCCESS ? cudaErrorInvalidValue : NvJpegToCuda(nvStatus);
        }
        if (widths[0] > (std::numeric_limits<int>::max)() / 3)
            return cudaErrorInvalidValue;

        result->width = widths[0];
        result->height = heights[0];
        result->pitchBytes = result->width * 3;
        result->deviceId = m_cudaDevice;
        size_t outputBytes = static_cast<size_t>(result->pitchBytes) * static_cast<size_t>(result->height);

        cudaError_t cudaStatus = cudaMalloc(&result->data, outputBytes);
        if (cudaStatus != cudaSuccess) {
            SetError(error, "cudaMalloc failed for decoded RGB image");
            return cudaStatus;
        }

        nvjpegImage_t destination{};
        destination.channel[0] = result->data;
        destination.pitch[0] = static_cast<unsigned int>(result->pitchBytes);
        nvStatus = nvjpegDecode(handle, state, jpegData, jpegSize, NVJPEG_OUTPUT_RGBI, &destination, stream);
        if (nvStatus != NVJPEG_STATUS_SUCCESS) {
            cudaStreamSynchronize(stream);
            cudaFree(result->data);
            result->data = nullptr;
            SetError(error, "nvjpegDecode failed");
            return NvJpegToCuda(nvStatus);
        }
        cudaStatus = cudaEventRecord(m_completion, stream);
        if (cudaStatus != cudaSuccess) {
            cudaStreamSynchronize(stream);
            cudaFree(result->data);
            result->data = nullptr;
            return cudaStatus;
        }
        m_pending = true;

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
        int device = 0;
        auto status = cudaGetDevice(&device);
        if (status != cudaSuccess) return status;
        auto& decoder = g_threadDecoders[device];
        if (!decoder) {
            decoder = new NvJpegDecoder();
        }
        return decoder->Decode(jpegData, jpegSize, stream, result, error);
    }

    void ShutdownThreadNvJpegDecoder()
    {
        for (auto& entry : g_threadDecoders) delete entry.second;
        g_threadDecoders.clear();
    }

    void FreeDecodedImage(GpuImageRGB8* result)
    {
        if (!result) {
            return;
        }
        if (result->data) {
            int previous = 0;
            if (cudaGetDevice(&previous) != cudaSuccess) return;
            if (previous != result->deviceId && cudaSetDevice(result->deviceId) != cudaSuccess) return;
            const auto status = cudaFree(result->data);
            if (previous != result->deviceId) cudaSetDevice(previous);
            if (status != cudaSuccess) return;
            result->data = nullptr;
        }
        result->width = 0;
        result->height = 0;
        result->pitchBytes = 0;
    }
}
