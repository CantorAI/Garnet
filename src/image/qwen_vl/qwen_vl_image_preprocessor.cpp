#include "qwen_vl_image_preprocessor.h"
#include "../../tensor/garnet_tensor.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
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

        X::Tensor MakeTensor(X::TensorDataType dtype, const std::vector<int>& shape)
        {
            X::Tensor tensor;
            X::Port::vector<int> xshape;
            for (int dim : shape) {
                xshape.push_back(dim);
            }
            tensor->SetDataType(dtype);
            tensor->SetShape(xshape);
            X::Value initData;
            tensor->Create(initData);
            return tensor;
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
        if (!rawImageValue.IsTensor()) {
            throw std::invalid_argument("raw image must be an X tensor");
        }
        if (height <= 0 || width <= 0) {
            throw std::invalid_argument("raw image height/width must be positive");
        }
        if (height % config.patchSize != 0 || width % config.patchSize != 0) {
            throw std::invalid_argument("raw image dimensions must be divisible by Qwen-VL patch_size");
        }

        X::Tensor rawImage(rawImageValue);
        if (rawImage->GetDataType() != X::TensorDataType::FLOAT32) {
            throw std::invalid_argument("raw image tensor must be float32 HWC");
        }

        int inputChannels = ChannelCount(config.pixelFormat);
        int expectedCount = height * width * inputChannels;
        if (rawImage->GetCount() != expectedCount) {
            throw std::invalid_argument("raw image tensor count does not match height * width * channels");
        }

        int gridH = height / config.patchSize;
        int gridW = width / config.patchSize;
        int featureDim = 3 * config.temporalPatchSize * config.patchSize * config.patchSize;
        int patchCount = gridH * gridW;

        X::Tensor pixelValues = MakeTensor(X::TensorDataType::FLOAT32, { patchCount, featureDim });
        X::Tensor imageGrid = MakeTensor(X::TensorDataType::INT64, { 1, 3 });

        float* dInput = nullptr;
        float* dOutput = nullptr;
        cudaStream_t stream = nullptr;
        size_t inputBytes = static_cast<size_t>(expectedCount) * sizeof(float);
        size_t outputBytes = static_cast<size_t>(patchCount) * static_cast<size_t>(featureDim) * sizeof(float);
        cudaError_t cudaStatus = cudaStreamCreate(&stream);
        if (cudaStatus != cudaSuccess ||
            cudaMalloc(&dInput, inputBytes) != cudaSuccess ||
            cudaMalloc(&dOutput, outputBytes) != cudaSuccess) {
            if (dInput) cudaFree(dInput);
            if (dOutput) cudaFree(dOutput);
            if (stream) cudaStreamDestroy(stream);
            throw std::runtime_error("failed to allocate CUDA image preprocessing buffers");
        }

        cudaStatus = cudaMemcpyAsync(dInput, rawImage->GetData(), inputBytes, cudaMemcpyHostToDevice, stream);

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

        if (cudaStatus == cudaSuccess) {
            cudaStatus = cudaMemcpyAsync(pixelValues->GetData(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        }

        if (cudaStatus == cudaSuccess) {
            cudaStatus = cudaStreamSynchronize(stream);
        }
        cudaFree(dInput);
        cudaFree(dOutput);
        cudaStreamDestroy(stream);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error(std::string("Qwen-VL CUDA image preprocessing failed: ") + cudaGetErrorString(cudaStatus));
        }

        auto* gridData = reinterpret_cast<long long*>(imageGrid->GetData());
        gridData[0] = 1;
        gridData[1] = gridH;
        gridData[2] = gridW;

        PreprocessResult result;
        result.pixelValues = X::Value(pixelValues);
        result.imageGridTHW = X::Value(imageGrid);
        result.resizedHeight = height;
        result.resizedWidth = width;
        result.patchSize = config.patchSize;
        result.temporalPatchSize = config.temporalPatchSize;
        result.mergeSize = config.mergeSize;
        return result;
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
