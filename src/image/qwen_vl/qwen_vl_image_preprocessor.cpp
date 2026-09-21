// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qwen_vl_image_preprocessor.h"
#include "qwen_vl_vision_metadata.h"
#include "../cuda/jpeg_decode_nvjpeg.h"
#include "../../tensor/garnet_tensor.h"
#include "../../tensor/tensor_helper.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <nppi_geometry_transforms.h>

extern "C" cudaError_t runQwenVLNormalizePatchLayoutFP32(
    const float* input,
    float* output,
    int height,
    int width,
    int inputChannels,
    int channelOrder,
    int patchSize,
    int temporalPatchSize,
    int mergeSize,
    float inputScale,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2,
    cudaStream_t stream);

extern "C" cudaError_t runQwenVLResizeNormalizePatchLayoutFP32(
    const float* input,
    float* output,
    int srcHeight,
    int srcWidth,
    int dstHeight,
    int dstWidth,
    int inputChannels,
    int channelOrder,
    int patchSize,
    int temporalPatchSize,
    int mergeSize,
    float inputScale,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2,
    cudaStream_t stream);

extern "C" cudaError_t runQwenVLResizeNormalizePatchLayoutRGB8(
    const unsigned char* input,
    float* output,
    int srcHeight,
    int srcWidth,
    int srcPitchBytes,
    int dstHeight,
    int dstWidth,
    int patchSize,
    int temporalPatchSize,
    int mergeSize,
    float inputScale,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2,
    cudaStream_t stream);

#if defined(_WIN32)
#define GARNET_IMAGE_EXPORT __declspec(dllexport)
#else
#define GARNET_IMAGE_EXPORT
#endif

namespace Garnet::Image::QwenVL
{
    namespace
    {
        int RoundToInt(double value)
        {
            return static_cast<int>(std::floor(value + 0.5));
        }

        X::Tensor MakeTensor(X3PackageHost* host, X3TensorDType dtype, const std::vector<int64_t>& shape)
        {
            return X::Tensor::Create(host, dtype, shape);
        }

        int ChannelOrder(PixelFormat format)
        {
            switch (format) {
            case PixelFormat::BGR_FLOAT32_HWC:
                return 1;
            case PixelFormat::RGBA_FLOAT32_HWC:
                return 2;
            case PixelFormat::BGRA_FLOAT32_HWC:
                return 3;
            default:
                return 0;
            }
        }

        DevicePreprocessResult PreprocessJpegBytesToDeviceBufferImpl(
            const unsigned char* jpegBytes,
            size_t jpegSize,
            int minPixels,
            int maxPixels)
        {
            constexpr int patchSize = 16;
            constexpr int temporalPatchSize = 2;
            constexpr int mergeSize = 2;
            if (!jpegBytes || jpegSize == 0 || minPixels <= 0 || maxPixels <= 0) {
                throw std::invalid_argument("invalid Qwen-VL JPEG preprocess arguments");
            }

            cudaStream_t stream = nullptr;
            cudaError_t status = cudaStreamCreate(&stream);
            if (status != cudaSuccess) {
                throw std::runtime_error("failed to create CUDA stream");
            }

            Cuda::GpuImageRGB8 decoded;
            std::string decodeError;
            status = Cuda::DecodeJpegToDeviceRGB8(jpegBytes, jpegSize, stream, &decoded, &decodeError);
            if (status != cudaSuccess) {
                cudaStreamDestroy(stream);
                throw std::runtime_error(decodeError.empty() ? cudaGetErrorString(status) : decodeError);
            }

            DevicePreprocessResult result;
            result.sourceHeight = decoded.height;
            result.sourceWidth = decoded.width;

            SmartResizeResult resize;
            try {
                resize = SmartResize(decoded.height, decoded.width, patchSize * mergeSize, minPixels, maxPixels);
            }
            catch (...) {
                Cuda::FreeDecodedImage(&decoded);
                cudaStreamDestroy(stream);
                throw;
            }

            int gridH = resize.height / patchSize;
            int gridW = resize.width / patchSize;
            if (gridH <= 0 || gridW <= 0 || gridH % mergeSize != 0 || gridW % mergeSize != 0) {
                Cuda::FreeDecodedImage(&decoded);
                cudaStreamDestroy(stream);
                throw std::runtime_error("resized JPEG grid is invalid for Qwen-VL merge size");
            }

            result.resizedHeight = resize.height;
            result.resizedWidth = resize.width;
            result.patchCount = gridH * gridW;
            result.featureDim = 3 * temporalPatchSize * patchSize * patchSize;
            result.outputBytes = static_cast<size_t>(result.patchCount) * static_cast<size_t>(result.featureDim) * sizeof(float);
            result.imageGridTHW[0] = 1;
            result.imageGridTHW[1] = gridH;
            result.imageGridTHW[2] = gridW;

            status = cudaMalloc(&result.pixelValuesDevice, result.outputBytes);
            if (status != cudaSuccess) {
                Cuda::FreeDecodedImage(&decoded);
                cudaStreamDestroy(stream);
                throw std::runtime_error("failed to allocate CUDA output buffer");
            }

            status = runQwenVLResizeNormalizePatchLayoutRGB8(
                decoded.data,
                result.pixelValuesDevice,
                decoded.height,
                decoded.width,
                decoded.pitchBytes,
                resize.height,
                resize.width,
                patchSize,
                temporalPatchSize,
                mergeSize,
                255.0f,
                0.5f,
                0.5f,
                0.5f,
                0.5f,
                0.5f,
                0.5f,
                stream);
            if (status == cudaSuccess) {
                status = cudaStreamSynchronize(stream);
            }
            Cuda::FreeDecodedImage(&decoded);
            cudaStreamDestroy(stream);
            if (status != cudaSuccess) {
                cudaFree(result.pixelValuesDevice);
                result.pixelValuesDevice = nullptr;
                throw std::runtime_error(cudaGetErrorString(status));
            }
            return result;
        }
    }

    DevicePreprocessResult PreprocessJpegFileToDeviceBuffer(
        const std::string& jpegPath,
        int minPixels,
        int maxPixels)
    {
        if (jpegPath.empty()) {
            throw std::invalid_argument("invalid Qwen-VL JPEG path");
        }
        unsigned char* jpegBytes = nullptr;
        size_t jpegSize = 0;
        std::string readError;
        if (!Cuda::ReadFileBytes(jpegPath.c_str(), &jpegBytes, &jpegSize, &readError)) {
            throw std::runtime_error(
                readError.empty() ? "failed to read JPEG file" : readError);
        }
        try {
            DevicePreprocessResult result = PreprocessJpegBytesToDeviceBufferImpl(
                jpegBytes, jpegSize, minPixels, maxPixels);
            delete[] jpegBytes;
            return result;
        }
        catch (...) {
            delete[] jpegBytes;
            throw;
        }
    }

    DevicePreprocessResult PreprocessJpegBytesToDeviceBuffer(
        const unsigned char* jpegData,
        size_t jpegSize,
        int minPixels,
        int maxPixels)
    {
        return PreprocessJpegBytesToDeviceBufferImpl(
            jpegData, jpegSize, minPixels, maxPixels);
    }

    SmartResizeResult SmartResize(
        int height,
        int width,
        int factor,
        int minPixels,
        int maxPixels)
    {
        if (height <= 0 || width <= 0 || factor <= 0 || minPixels <= 0 || maxPixels <= 0) {
            throw std::invalid_argument("invalid Qwen-VL smart resize arguments");
        }
        double aspect = static_cast<double>(std::max(height, width)) / static_cast<double>(std::min(height, width));
        if (aspect > 200.0) {
            throw std::invalid_argument("Qwen-VL image aspect ratio is too large");
        }

        int hBar = RoundToInt(static_cast<double>(height) / factor) * factor;
        int wBar = RoundToInt(static_cast<double>(width) / factor) * factor;
        if (hBar * wBar > maxPixels) {
            double beta = std::sqrt(static_cast<double>(height) * static_cast<double>(width) / static_cast<double>(maxPixels));
            hBar = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
            wBar = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
        }
        else if (hBar * wBar < minPixels) {
            double beta = std::sqrt(static_cast<double>(minPixels) / (static_cast<double>(height) * static_cast<double>(width)));
            hBar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
            wBar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
        }
        return { hBar, wBar };
    }

    PreprocessResult PreprocessRawImageTensor(
        X::Value rawImageValue,
        int height,
        int width,
        const QwenVLImagePreprocessConfig& config)
    {
        if (height <= 0 || width <= 0 || config.patchSize <= 0 ||
            config.temporalPatchSize <= 0 || config.mergeSize <= 0 ||
            !std::isfinite(config.inputScale) || config.inputScale <= 0) {
            throw std::invalid_argument("invalid raw image dimensions or preprocessing configuration");
        }
        for (int channel = 0; channel < 3; ++channel) {
            if (!std::isfinite(config.mean[channel]) || !std::isfinite(config.std[channel]) || config.std[channel] <= 0) {
                throw std::invalid_argument("invalid image normalization parameters");
            }
        }
        if (height % config.patchSize != 0 || width % config.patchSize != 0) {
            throw std::invalid_argument("raw image dimensions must be divisible by Qwen-VL patch_size");
        }

        X::Tensor rawImage(rawImageValue);
        auto rawInfo = rawImage.Info();
        if (rawInfo.dtype != X3_TENSOR_FLOAT32 || rawInfo.symbolic || !rawInfo.data) {
            throw std::invalid_argument("raw image tensor must be float32 HWC");
        }

        int inputChannels = ChannelCount(config.pixelFormat);
        const int64_t pixelCount = static_cast<int64_t>(height) * width;
        if (pixelCount > (std::numeric_limits<int>::max)() / inputChannels) {
            throw std::invalid_argument("raw image exceeds CUDA kernel indexing limits");
        }
        int64_t expectedCount = pixelCount * inputChannels;
        int64_t count = 1;
        int64_t stride = sizeof(float);
        for (uint32_t axis = rawInfo.rank; axis-- > 0;) {
            if (rawInfo.shape[axis] <= 0 || rawInfo.shape[axis] > expectedCount / count ||
                (rawInfo.shape[axis] > 1 && rawInfo.strides[axis] != stride)) {
                throw std::invalid_argument("raw image must have contiguous HWC storage");
            }
            count *= rawInfo.shape[axis];
            stride *= rawInfo.shape[axis];
        }
        if (count != expectedCount) {
            throw std::invalid_argument("raw image tensor count does not match height * width * channels");
        }

        int gridH = height / config.patchSize;
        int gridW = width / config.patchSize;
        if (gridH % config.mergeSize || gridW % config.mergeSize) {
            throw std::invalid_argument("raw image grid must be divisible by merge size");
        }
        int64_t featureSize = 3;
        const int64_t featureLimit = (std::numeric_limits<int>::max)() / (static_cast<int64_t>(gridH) * gridW);
        for (int dimension : {config.temporalPatchSize, config.patchSize, config.patchSize}) {
            if (featureSize > featureLimit / dimension) {
                throw std::invalid_argument("image output exceeds CUDA kernel indexing limits");
            }
            featureSize *= dimension;
        }
        int featureDim = static_cast<int>(featureSize);
        int patchCount = gridH * gridW;

        X::Tensor pixelValues = TensorHelper::CreateGPU(rawImage.host(), X3_TENSOR_FLOAT32, { patchCount, featureDim });
        X::Tensor imageGrid = MakeTensor(rawImage.host(), X3_TENSOR_INT64, { 1, 3 });

        X::Tensor input = TensorHelper::CopyToGPU(rawImage);
        int currentDevice = 0;
        if (cudaGetDevice(&currentDevice) != cudaSuccess || input.Info().device_id != currentDevice) {
            throw std::invalid_argument("raw image GPU must match the current CUDA device");
        }
        const float* dInput = static_cast<const float*>(TensorHelper::GetGPUMemory(input));
        float* dOutput = static_cast<float*>(TensorHelper::GetGPUMemory(pixelValues));
        cudaStream_t stream = cudaStreamPerThread;
        auto use = TensorHelper::AcquireGPU({{input, X3_TENSOR_READ}, {pixelValues, X3_TENSOR_WRITE}}, stream);
        cudaError_t cudaStatus = cudaSuccess;

        if (cudaStatus == cudaSuccess) {
            cudaStatus = runQwenVLNormalizePatchLayoutFP32(
                dInput,
                dOutput,
                height,
                width,
                inputChannels,
                ChannelOrder(config.pixelFormat),
                config.patchSize,
                config.temporalPatchSize,
                config.mergeSize,
                config.inputScale,
                config.mean[0],
                config.mean[1],
                config.mean[2],
                config.std[0],
                config.std[1],
                config.std[2],
                stream);
        }

        use.Finish();
        if (cudaStatus == cudaSuccess) {
            cudaStatus = cudaStreamSynchronize(stream);
        }
        const char* syncEnv = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
        bool syncCPU = !syncEnv || !(syncEnv[0] == '0' && syncEnv[1] == '\0');
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("Qwen-VL CUDA image preprocessing failed: ") + cudaGetErrorString(cudaStatus));
        }
        if (syncCPU) pixelValues = TensorHelper::CopyToCPU(pixelValues);

        auto* gridData = static_cast<int64_t*>(imageGrid.Info().data);
        gridData[0] = 1;
        gridData[1] = gridH;
        gridData[2] = gridW;

        PreprocessResult result;
        result.pixelValues = X::Value(pixelValues);
        result.imageGridTHW = X::Value(imageGrid);
        auto metadata = BuildVisionMetadataTensors(rawImage.host(), 1, gridH, gridW, config.mergeSize);
        result.bilinearIndices = metadata.bilinearIndices;
        result.bilinearWeights = metadata.bilinearWeights;
        result.visionPositionIds = metadata.positionIds;
        result.visionCuSeqlens = metadata.cuSeqlens;
        result.sourceHeight = height;
        result.sourceWidth = width;
        result.resizedHeight = height;
        result.resizedWidth = width;
        result.patchSize = config.patchSize;
        result.temporalPatchSize = config.temporalPatchSize;
        result.mergeSize = config.mergeSize;
        return result;
    }

    PreprocessResult PreprocessJpegDeviceBufferToTensor(
        X3PackageHost* host,
        DevicePreprocessResult deviceResult)
    {
        // This function consumes the allocation even when wrapping fails.
        X::Tensor pixelValues;
        try {
            int device = 0;
            auto status = cudaGetDevice(&device);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            if (deviceResult.patchCount <= 0 || deviceResult.featureDim <= 0 ||
                deviceResult.outputBytes != static_cast<uint64_t>(deviceResult.patchCount) * deviceResult.featureDim * sizeof(float)) {
                throw std::invalid_argument("invalid JPEG device tensor dimensions");
            }
            int64_t shape[] = {deviceResult.patchCount, deviceResult.featureDim};
            int64_t strides[] = {static_cast<int64_t>(deviceResult.featureDim) * sizeof(float), sizeof(float)};
            X3TensorInfo info{};
            info.size = sizeof(info);
            info.dtype = X3_TENSOR_FLOAT32;
            info.rank = 2;
            info.shape = shape;
            info.strides = strides;
            info.data = deviceResult.pixelValuesDevice;
            info.byte_size = deviceResult.outputBytes;
            info.device_type = TensorHelper::CudaDevice;
            info.device_id = device;
            pixelValues = TensorHelper::WrapGPU(host, info, deviceResult.pixelValuesDevice, device);
        }
        catch (...) {
            cudaFree(deviceResult.pixelValuesDevice);
            throw;
        }

        const char* syncEnv = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
        bool syncCPU = !syncEnv || !(syncEnv[0] == '0' && syncEnv[1] == '\0');
        if (syncCPU) {
            pixelValues = TensorHelper::CopyToCPU(pixelValues);
        }

        X::Tensor imageGrid = MakeTensor(host, X3_TENSOR_INT64, { 1, 3 });
        auto* gridData = static_cast<int64_t*>(imageGrid.Info().data);
        gridData[0] = deviceResult.imageGridTHW[0];
        gridData[1] = deviceResult.imageGridTHW[1];
        gridData[2] = deviceResult.imageGridTHW[2];

        PreprocessResult result;
        result.pixelValues = X::Value(pixelValues);
        result.imageGridTHW = X::Value(imageGrid);
        auto metadata = BuildVisionMetadataTensors(
            host,
            static_cast<int>(deviceResult.imageGridTHW[0]),
            static_cast<int>(deviceResult.imageGridTHW[1]),
            static_cast<int>(deviceResult.imageGridTHW[2]),
            deviceResult.mergeSize);
        result.bilinearIndices = metadata.bilinearIndices;
        result.bilinearWeights = metadata.bilinearWeights;
        result.visionPositionIds = metadata.positionIds;
        result.visionCuSeqlens = metadata.cuSeqlens;
        result.sourceHeight = deviceResult.sourceHeight;
        result.sourceWidth = deviceResult.sourceWidth;
        result.resizedHeight = deviceResult.resizedHeight;
        result.resizedWidth = deviceResult.resizedWidth;
        result.patchSize = deviceResult.patchSize;
        result.temporalPatchSize = deviceResult.temporalPatchSize;
        result.mergeSize = deviceResult.mergeSize;
        return result;
    }

    PreprocessResult PreprocessJpegFileToTensor(
        X3PackageHost* host,
        const std::string& jpegPath,
        int minPixels,
        int maxPixels)
    {
        return PreprocessJpegDeviceBufferToTensor(
            host,
            PreprocessJpegFileToDeviceBuffer(jpegPath, minPixels, maxPixels));
    }

    PreprocessResult PreprocessJpegBytesToTensor(
        X3PackageHost* host,
        const unsigned char* jpegData,
        size_t jpegSize,
        int minPixels,
        int maxPixels)
    {
        return PreprocessJpegDeviceBufferToTensor(
            host,
            PreprocessJpegBytesToDeviceBuffer(
                jpegData, jpegSize, minPixels, maxPixels));
    }
}

extern "C" GARNET_IMAGE_EXPORT int GarnetQwenVLPreprocessRGBF32(
    const float* input,
    int height,
    int width,
    int inputChannels,
    int channelOrder,
    float inputScale,
    float* output,
    long long* imageGridTHW,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message);
        }
    };

    constexpr int patchSize = 16;
    constexpr int temporalPatchSize = 2;
    constexpr int mergeSize = 2;
    if (!input || !output || !imageGridTHW || height <= 0 || width <= 0 || inputChannels < 3) {
        setError("invalid Qwen-VL preprocess C ABI arguments");
        return 1;
    }
    if (height % patchSize != 0 || width % patchSize != 0) {
        setError("height/width must be divisible by patch_size=16");
        return 2;
    }

    int gridH = height / patchSize;
    int gridW = width / patchSize;
    if (gridH % mergeSize != 0 || gridW % mergeSize != 0) {
        setError("grid_h/grid_w must be divisible by merge_size=2");
        return 3;
    }

    int expectedCount = height * width * inputChannels;
    int patchCount = gridH * gridW;
    int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
    size_t inputBytes = static_cast<size_t>(expectedCount) * sizeof(float);
    size_t outputBytes = static_cast<size_t>(patchCount) * static_cast<size_t>(featureDim) * sizeof(float);

    float* dInput = nullptr;
    float* dOutput = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t status = cudaStreamCreate(&stream);
    if (status != cudaSuccess ||
        cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
        cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
        if (dInput) cudaFree(dInput);
        if (dOutput) cudaFree(dOutput);
        if (stream) cudaStreamDestroy(stream);
        setError("failed to allocate CUDA buffers");
        return 4;
    }

    status = cudaMemcpyAsync(dInput, input, inputBytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
        status = runQwenVLNormalizePatchLayoutFP32(
            dInput,
            dOutput,
            height,
            width,
            inputChannels,
            channelOrder,
            patchSize,
            temporalPatchSize,
            mergeSize,
            inputScale,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(output, dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(stream);
    }

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaStreamDestroy(stream);
    if (status != cudaSuccess) {
        setError(cudaGetErrorString(status));
        return 5;
    }

    imageGridTHW[0] = 1;
    imageGridTHW[1] = gridH;
    imageGridTHW[2] = gridW;
    setError("");
    return 0;
}

extern "C" GARNET_IMAGE_EXPORT int GarnetQwenVLResizePreprocessRGBF32(
    const float* input,
    int srcHeight,
    int srcWidth,
    int inputChannels,
    int channelOrder,
    int minPixels,
    int maxPixels,
    float inputScale,
    float* output,
    long long* imageGridTHW,
    int* resizedHeight,
    int* resizedWidth,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message);
        }
    };

    constexpr int patchSize = 16;
    constexpr int temporalPatchSize = 2;
    constexpr int mergeSize = 2;
    if (!input || !output || !imageGridTHW || !resizedHeight || !resizedWidth ||
        srcHeight <= 0 || srcWidth <= 0 || inputChannels < 3) {
        setError("invalid Qwen-VL resize preprocess C ABI arguments");
        return 1;
    }

    Garnet::Image::QwenVL::SmartResizeResult resize;
    try {
        resize = Garnet::Image::QwenVL::SmartResize(
            srcHeight,
            srcWidth,
            patchSize * mergeSize,
            minPixels,
            maxPixels);
    }
    catch (const std::exception& exc) {
        setError(exc.what());
        return 2;
    }

    int gridH = resize.height / patchSize;
    int gridW = resize.width / patchSize;
    if (gridH % mergeSize != 0 || gridW % mergeSize != 0) {
        setError("resized grid is not divisible by merge_size=2");
        return 3;
    }

    int inputCount = srcHeight * srcWidth * inputChannels;
    int patchCount = gridH * gridW;
    int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
    size_t inputBytes = static_cast<size_t>(inputCount) * sizeof(float);
    size_t resizedBytes = static_cast<size_t>(resize.height) * static_cast<size_t>(resize.width) * static_cast<size_t>(inputChannels) * sizeof(float);
    size_t outputBytes = static_cast<size_t>(patchCount) * static_cast<size_t>(featureDim) * sizeof(float);

    float* dInput = nullptr;
    float* dResized = nullptr;
    float* dOutput = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t status = cudaStreamCreate(&stream);
    if (status != cudaSuccess ||
        cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
        cudaMalloc(&dResized, resizedBytes) != cudaSuccess ||
        cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
        if (dInput) cudaFree(dInput);
        if (dResized) cudaFree(dResized);
        if (dOutput) cudaFree(dOutput);
        if (stream) cudaStreamDestroy(stream);
        setError("failed to allocate CUDA resize/preprocess buffers");
        return 4;
    }

    status = cudaMemcpyAsync(dInput, input, inputBytes, cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess) {
        int device = 0;
        cudaDeviceProp props{};
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        NppStreamContext nppContext{};
        nppContext.hStream = stream;
        nppContext.nCudaDeviceId = device;
        nppContext.nMultiProcessorCount = props.multiProcessorCount;
        nppContext.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
        nppContext.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
        nppContext.nSharedMemPerBlock = props.sharedMemPerBlock;
        nppContext.nCudaDevAttrComputeCapabilityMajor = props.major;
        nppContext.nCudaDevAttrComputeCapabilityMinor = props.minor;
        unsigned int streamFlags = 0;
        cudaStreamGetFlags(stream, &streamFlags);
        nppContext.nStreamFlags = streamFlags;

        NppiSize srcSize{ srcWidth, srcHeight };
        NppiRect srcRect{ 0, 0, srcWidth, srcHeight };
        NppiSize dstSize{ resize.width, resize.height };
        NppiRect dstRect{ 0, 0, resize.width, resize.height };
        NppStatus nppStatus = nppiResize_32f_C3R_Ctx(
            dInput,
            srcWidth * inputChannels * static_cast<int>(sizeof(float)),
            srcSize,
            srcRect,
            dResized,
            resize.width * inputChannels * static_cast<int>(sizeof(float)),
            dstSize,
            dstRect,
            NPPI_INTER_CUBIC,
            nppContext);
        if (nppStatus != NPP_SUCCESS) {
            status = cudaErrorUnknown;
            setError("NPP resize failed");
        }
    }
    if (status == cudaSuccess) {
        status = runQwenVLNormalizePatchLayoutFP32(
            dResized,
            dOutput,
            resize.height,
            resize.width,
            inputChannels,
            channelOrder,
            patchSize,
            temporalPatchSize,
            mergeSize,
            inputScale,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(output, dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) {
        status = cudaStreamSynchronize(stream);
    }

    cudaFree(dInput);
    cudaFree(dResized);
    cudaFree(dOutput);
    cudaStreamDestroy(stream);
    if (status != cudaSuccess) {
        setError(cudaGetErrorString(status));
        return 5;
    }

    imageGridTHW[0] = 1;
    imageGridTHW[1] = gridH;
    imageGridTHW[2] = gridW;
    *resizedHeight = resize.height;
    *resizedWidth = resize.width;
    setError("");
    return 0;
}

namespace
{
    int QwenVLPreprocessJpegBytesImpl(
        const unsigned char* jpegData,
        size_t jpegSize,
        int minPixels,
        int maxPixels,
        float* output,
        long long* imageGridTHW,
        int* sourceHeight,
        int* sourceWidth,
        int* resizedHeight,
        int* resizedWidth,
        char* errorMessage,
        int errorMessageCapacity)
    {
        auto setError = [&](const char* message) {
            if (errorMessage && errorMessageCapacity > 0) {
                std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message);
            }
        };

        constexpr int patchSize = 16;
        constexpr int temporalPatchSize = 2;
        constexpr int mergeSize = 2;
        if (!jpegData || jpegSize == 0 || !output || !imageGridTHW || !sourceHeight || !sourceWidth ||
            !resizedHeight || !resizedWidth || minPixels <= 0 || maxPixels <= 0) {
            setError("invalid Qwen-VL JPEG preprocess arguments");
            return 1;
        }

        cudaStream_t stream = nullptr;
        cudaError_t status = cudaStreamCreate(&stream);
        if (status != cudaSuccess) {
            setError("failed to create CUDA stream");
            return 2;
        }

        Garnet::Image::Cuda::GpuImageRGB8 decoded;
        std::string decodeError;
        status = Garnet::Image::Cuda::DecodeJpegToDeviceRGB8(jpegData, jpegSize, stream, &decoded, &decodeError);
        if (status != cudaSuccess) {
            cudaStreamDestroy(stream);
            setError(decodeError.empty() ? cudaGetErrorString(status) : decodeError.c_str());
            return 3;
        }

        Garnet::Image::QwenVL::SmartResizeResult resize;
        try {
            resize = Garnet::Image::QwenVL::SmartResize(
                decoded.height,
                decoded.width,
                patchSize * mergeSize,
                minPixels,
                maxPixels);
        }
        catch (const std::exception& exc) {
            Garnet::Image::Cuda::FreeDecodedImage(&decoded);
            cudaStreamDestroy(stream);
            setError(exc.what());
            return 4;
        }

        int gridH = resize.height / patchSize;
        int gridW = resize.width / patchSize;
        if (gridH % mergeSize != 0 || gridW % mergeSize != 0) {
            Garnet::Image::Cuda::FreeDecodedImage(&decoded);
            cudaStreamDestroy(stream);
            setError("resized JPEG grid is not divisible by merge_size=2");
            return 5;
        }

        int patchCount = gridH * gridW;
        int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
        size_t outputBytes = static_cast<size_t>(patchCount) * static_cast<size_t>(featureDim) * sizeof(float);
        int decodedHeight = decoded.height;
        int decodedWidth = decoded.width;
        float* dOutput = nullptr;
        status = cudaMalloc(&dOutput, outputBytes);
        if (status != cudaSuccess) {
            Garnet::Image::Cuda::FreeDecodedImage(&decoded);
            cudaStreamDestroy(stream);
            setError("failed to allocate CUDA output buffer");
            return 6;
        }

        status = runQwenVLResizeNormalizePatchLayoutRGB8(
            decoded.data,
            dOutput,
            decoded.height,
            decoded.width,
            decoded.pitchBytes,
            resize.height,
            resize.width,
            patchSize,
            temporalPatchSize,
            mergeSize,
            255.0f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            0.5f,
            stream);

        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(output, dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        }
        if (status == cudaSuccess) {
            status = cudaStreamSynchronize(stream);
        }

        cudaFree(dOutput);
        Garnet::Image::Cuda::FreeDecodedImage(&decoded);
        cudaStreamDestroy(stream);
        if (status != cudaSuccess) {
            setError(cudaGetErrorString(status));
            return 7;
        }

        imageGridTHW[0] = 1;
        imageGridTHW[1] = gridH;
        imageGridTHW[2] = gridW;
        *sourceHeight = decodedHeight;
        *sourceWidth = decodedWidth;
        *resizedHeight = resize.height;
        *resizedWidth = resize.width;
        setError("");
        return 0;
    }
}

extern "C" GARNET_IMAGE_EXPORT int GarnetQwenVLPreprocessJpegBytes(
    const unsigned char* jpegData,
    size_t jpegSize,
    int minPixels,
    int maxPixels,
    float* output,
    long long* imageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    char* errorMessage,
    int errorMessageCapacity)
{
    return QwenVLPreprocessJpegBytesImpl(
        jpegData,
        jpegSize,
        minPixels,
        maxPixels,
        output,
        imageGridTHW,
        sourceHeight,
        sourceWidth,
        resizedHeight,
        resizedWidth,
        errorMessage,
        errorMessageCapacity);
}

extern "C" GARNET_IMAGE_EXPORT int GarnetQwenVLPreprocessJpegFile(
    const char* jpegPath,
    int minPixels,
    int maxPixels,
    float* output,
    long long* imageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message);
        }
    };

    unsigned char* bytes = nullptr;
    size_t byteCount = 0;
    std::string readError;
    if (!Garnet::Image::Cuda::ReadFileBytes(jpegPath, &bytes, &byteCount, &readError)) {
        setError(readError.empty() ? "failed to read JPEG file" : readError.c_str());
        return 1;
    }

    int rc = QwenVLPreprocessJpegBytesImpl(
        bytes,
        byteCount,
        minPixels,
        maxPixels,
        output,
        imageGridTHW,
        sourceHeight,
        sourceWidth,
        resizedHeight,
        resizedWidth,
        errorMessage,
        errorMessageCapacity);
    delete[] bytes;
    return rc;
}

extern "C" GARNET_IMAGE_EXPORT int GarnetQwenVLPreprocessJpegFileDevice(
    const char* jpegPath,
    int minPixels,
    int maxPixels,
    void** pixelValuesDevice,
    size_t* outputBytes,
    long long* imageGridTHW,
    int* sourceHeight,
    int* sourceWidth,
    int* resizedHeight,
    int* resizedWidth,
    int* patchCount,
    int* featureDim,
    char* errorMessage,
    int errorMessageCapacity)
{
    auto setError = [&](const char* message) {
        if (errorMessage && errorMessageCapacity > 0) {
            std::snprintf(errorMessage, static_cast<size_t>(errorMessageCapacity), "%s", message);
        }
    };

    if (!jpegPath || !pixelValuesDevice || !outputBytes || !imageGridTHW ||
        !sourceHeight || !sourceWidth || !resizedHeight || !resizedWidth ||
        !patchCount || !featureDim) {
        setError("invalid Qwen-VL device preprocess C ABI arguments");
        return 1;
    }

    *pixelValuesDevice = nullptr;
    *outputBytes = 0;
    try {
        auto result = Garnet::Image::QwenVL::PreprocessJpegFileToDeviceBuffer(jpegPath, minPixels, maxPixels);
        *pixelValuesDevice = result.pixelValuesDevice;
        *outputBytes = result.outputBytes;
        imageGridTHW[0] = result.imageGridTHW[0];
        imageGridTHW[1] = result.imageGridTHW[1];
        imageGridTHW[2] = result.imageGridTHW[2];
        *sourceHeight = result.sourceHeight;
        *sourceWidth = result.sourceWidth;
        *resizedHeight = result.resizedHeight;
        *resizedWidth = result.resizedWidth;
        *patchCount = result.patchCount;
        *featureDim = result.featureDim;
        setError("");
        return 0;
    }
    catch (const std::exception& exc) {
        setError(exc.what());
        return 2;
    }
}

extern "C" GARNET_IMAGE_EXPORT int GarnetFreeDeviceBuffer(void* deviceBuffer)
{
    if (!deviceBuffer) {
        return 0;
    }
    return cudaFree(deviceBuffer) == cudaSuccess ? 0 : 1;
}
