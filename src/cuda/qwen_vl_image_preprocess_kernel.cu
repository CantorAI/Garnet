#include <cuda_runtime.h>

namespace
{
    __device__ __forceinline__ float CubicWeight(float x)
    {
        // Matches the common bicubic family used by image libraries closely enough for
        // model input; exact PIL/torchvision parity is tracked separately in tests.
        constexpr float a = -0.75f;
        x = fabsf(x);
        if (x <= 1.0f) {
            return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
        }
        if (x < 2.0f) {
            return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
        }
        return 0.0f;
    }

    __device__ __forceinline__ int ClampInt(int value, int low, int high)
    {
        return max(low, min(value, high));
    }

    __device__ __forceinline__ float ReadChannel(
        const float* input,
        int pixelBase,
        int inputChannels,
        int channelOrder,
        int channel)
    {
        int sourceChannel = channel;
        if (channelOrder == 1 || channelOrder == 3) {
            sourceChannel = 2 - channel;
        }
        return input[pixelBase + sourceChannel];
    }

    __device__ __forceinline__ float ReadChannelRGB8(
        const unsigned char* input,
        int y,
        int x,
        int pitchBytes,
        int channel)
    {
        const unsigned char* pixel = input + y * pitchBytes + x * 3;
        return static_cast<float>(pixel[channel]);
    }

    __device__ float SampleBicubic(
        const float* input,
        int srcHeight,
        int srcWidth,
        int inputChannels,
        int channelOrder,
        float y,
        float x,
        int channel)
    {
        int yBase = static_cast<int>(floorf(y));
        int xBase = static_cast<int>(floorf(x));
        float accum = 0.0f;
        float weightSum = 0.0f;
        for (int dy = -1; dy <= 2; ++dy) {
            int sy = ClampInt(yBase + dy, 0, srcHeight - 1);
            float wy = CubicWeight(y - static_cast<float>(yBase + dy));
            for (int dx = -1; dx <= 2; ++dx) {
                int sx = ClampInt(xBase + dx, 0, srcWidth - 1);
                float wx = CubicWeight(x - static_cast<float>(xBase + dx));
                float w = wy * wx;
                int pixelBase = (sy * srcWidth + sx) * inputChannels;
                accum += ReadChannel(input, pixelBase, inputChannels, channelOrder, channel) * w;
                weightSum += w;
            }
        }
        return weightSum != 0.0f ? accum / weightSum : 0.0f;
    }

    __device__ float SampleBicubicRGB8(
        const unsigned char* input,
        int srcHeight,
        int srcWidth,
        int pitchBytes,
        float y,
        float x,
        int channel)
    {
        int yBase = static_cast<int>(floorf(y));
        int xBase = static_cast<int>(floorf(x));
        float accum = 0.0f;
        float weightSum = 0.0f;
        for (int dy = -1; dy <= 2; ++dy) {
            int sy = ClampInt(yBase + dy, 0, srcHeight - 1);
            float wy = CubicWeight(y - static_cast<float>(yBase + dy));
            for (int dx = -1; dx <= 2; ++dx) {
                int sx = ClampInt(xBase + dx, 0, srcWidth - 1);
                float wx = CubicWeight(x - static_cast<float>(xBase + dx));
                float w = wy * wx;
                accum += ReadChannelRGB8(input, sy, sx, pitchBytes, channel) * w;
                weightSum += w;
            }
        }
        return weightSum != 0.0f ? accum / weightSum : 0.0f;
    }

    __global__ void QwenVLNormalizePatchLayoutKernel(
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
        int patchCount,
        int featureDim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = patchCount * featureDim;
        if (idx >= total) {
            return;
        }

        int feature = idx % featureDim;
        int patchIndex = idx / featureDim;

        int patchArea = patchSize * patchSize;
        int channelStride = temporalPatchSize * patchArea;
        int channel = feature / channelStride;
        int rem = feature - channel * channelStride;
        rem %= patchArea;
        int patchY = rem / patchSize;
        int patchX = rem - patchY * patchSize;

        int gridW = width / patchSize;
        int outerW = gridW / mergeSize;
        int mergePair = patchIndex % (mergeSize * mergeSize);
        int outerIndex = patchIndex / (mergeSize * mergeSize);
        int mergeY = mergePair / mergeSize;
        int mergeX = mergePair - mergeY * mergeSize;
        int outerY = outerIndex / outerW;
        int outerX = outerIndex - outerY * outerW;

        int imageY = (outerY * mergeSize + mergeY) * patchSize + patchY;
        int imageX = (outerX * mergeSize + mergeX) * patchSize + patchX;
        int pixelBase = (imageY * width + imageX) * inputChannels;

        float mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
        float stdv = channel == 0 ? std0 : (channel == 1 ? std1 : std2);
        float value = ReadChannel(input, pixelBase, inputChannels, channelOrder, channel);
        output[idx] = (value / inputScale - mean) / stdv;
    }

    __global__ void QwenVLResizeNormalizePatchLayoutKernel(
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
        int patchCount,
        int featureDim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = patchCount * featureDim;
        if (idx >= total) {
            return;
        }

        int feature = idx % featureDim;
        int patchIndex = idx / featureDim;

        int patchArea = patchSize * patchSize;
        int channelStride = temporalPatchSize * patchArea;
        int channel = feature / channelStride;
        int rem = feature - channel * channelStride;
        rem %= patchArea;
        int patchY = rem / patchSize;
        int patchX = rem - patchY * patchSize;

        int gridW = dstWidth / patchSize;
        int outerW = gridW / mergeSize;
        int mergePair = patchIndex % (mergeSize * mergeSize);
        int outerIndex = patchIndex / (mergeSize * mergeSize);
        int mergeY = mergePair / mergeSize;
        int mergeX = mergePair - mergeY * mergeSize;
        int outerY = outerIndex / outerW;
        int outerX = outerIndex - outerY * outerW;

        int dstY = (outerY * mergeSize + mergeY) * patchSize + patchY;
        int dstX = (outerX * mergeSize + mergeX) * patchSize + patchX;

        float srcY = (static_cast<float>(dstY) + 0.5f) * static_cast<float>(srcHeight) / static_cast<float>(dstHeight) - 0.5f;
        float srcX = (static_cast<float>(dstX) + 0.5f) * static_cast<float>(srcWidth) / static_cast<float>(dstWidth) - 0.5f;
        float value = SampleBicubic(input, srcHeight, srcWidth, inputChannels, channelOrder, srcY, srcX, channel);
        value = fminf(255.0f, fmaxf(0.0f, value));

        float mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
        float stdv = channel == 0 ? std0 : (channel == 1 ? std1 : std2);
        output[idx] = (value / inputScale - mean) / stdv;
    }

    __global__ void QwenVLResizeNormalizePatchLayoutRGB8Kernel(
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
        int patchCount,
        int featureDim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = patchCount * featureDim;
        if (idx >= total) {
            return;
        }

        int feature = idx % featureDim;
        int patchIndex = idx / featureDim;

        int patchArea = patchSize * patchSize;
        int channelStride = temporalPatchSize * patchArea;
        int channel = feature / channelStride;
        int rem = feature - channel * channelStride;
        rem %= patchArea;
        int patchY = rem / patchSize;
        int patchX = rem - patchY * patchSize;

        int gridW = dstWidth / patchSize;
        int outerW = gridW / mergeSize;
        int mergePair = patchIndex % (mergeSize * mergeSize);
        int outerIndex = patchIndex / (mergeSize * mergeSize);
        int mergeY = mergePair / mergeSize;
        int mergeX = mergePair - mergeY * mergeSize;
        int outerY = outerIndex / outerW;
        int outerX = outerIndex - outerY * outerW;

        int dstY = (outerY * mergeSize + mergeY) * patchSize + patchY;
        int dstX = (outerX * mergeSize + mergeX) * patchSize + patchX;

        float srcY = (static_cast<float>(dstY) + 0.5f) * static_cast<float>(srcHeight) / static_cast<float>(dstHeight) - 0.5f;
        float srcX = (static_cast<float>(dstX) + 0.5f) * static_cast<float>(srcWidth) / static_cast<float>(dstWidth) - 0.5f;
        float value = SampleBicubicRGB8(input, srcHeight, srcWidth, srcPitchBytes, srcY, srcX, channel);
        value = fminf(255.0f, fmaxf(0.0f, value));

        float mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
        float stdv = channel == 0 ? std0 : (channel == 1 ? std1 : std2);
        output[idx] = (value / inputScale - mean) / stdv;
    }
}

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
    cudaStream_t stream)
{
    if (!input || !output || height <= 0 || width <= 0 || inputChannels < 3 ||
        patchSize <= 0 || temporalPatchSize <= 0 || mergeSize <= 0 ||
        height % patchSize != 0 || width % patchSize != 0) {
        return cudaErrorInvalidValue;
    }

    int gridH = height / patchSize;
    int gridW = width / patchSize;
    if (gridH % mergeSize != 0 || gridW % mergeSize != 0) {
        return cudaErrorInvalidValue;
    }

    int patchCount = gridH * gridW;
    int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
    int total = patchCount * featureDim;
    int block = 256;
    int grid = (total + block - 1) / block;
    QwenVLNormalizePatchLayoutKernel<<<grid, block, 0, stream>>>(
        input,
        output,
        height,
        width,
        inputChannels,
        channelOrder,
        patchSize,
        temporalPatchSize,
        mergeSize,
        inputScale,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        patchCount,
        featureDim);
    return cudaGetLastError();
}

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
    cudaStream_t stream)
{
    if (!input || !output || srcHeight <= 0 || srcWidth <= 0 || srcPitchBytes < srcWidth * 3 ||
        dstHeight <= 0 || dstWidth <= 0 || patchSize <= 0 || temporalPatchSize <= 0 || mergeSize <= 0 ||
        dstHeight % patchSize != 0 || dstWidth % patchSize != 0) {
        return cudaErrorInvalidValue;
    }
    int gridH = dstHeight / patchSize;
    int gridW = dstWidth / patchSize;
    if (gridH % mergeSize != 0 || gridW % mergeSize != 0) {
        return cudaErrorInvalidValue;
    }
    int patchCount = gridH * gridW;
    int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
    int total = patchCount * featureDim;
    int block = 256;
    int grid = (total + block - 1) / block;
    QwenVLResizeNormalizePatchLayoutRGB8Kernel<<<grid, block, 0, stream>>>(
        input,
        output,
        srcHeight,
        srcWidth,
        srcPitchBytes,
        dstHeight,
        dstWidth,
        patchSize,
        temporalPatchSize,
        mergeSize,
        inputScale,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        patchCount,
        featureDim);
    return cudaGetLastError();
}

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
    cudaStream_t stream)
{
    if (!input || !output || srcHeight <= 0 || srcWidth <= 0 || dstHeight <= 0 || dstWidth <= 0 ||
        inputChannels < 3 || patchSize <= 0 || temporalPatchSize <= 0 || mergeSize <= 0 ||
        dstHeight % patchSize != 0 || dstWidth % patchSize != 0) {
        return cudaErrorInvalidValue;
    }
    int gridH = dstHeight / patchSize;
    int gridW = dstWidth / patchSize;
    if (gridH % mergeSize != 0 || gridW % mergeSize != 0) {
        return cudaErrorInvalidValue;
    }
    int patchCount = gridH * gridW;
    int featureDim = 3 * temporalPatchSize * patchSize * patchSize;
    int total = patchCount * featureDim;
    int block = 256;
    int grid = (total + block - 1) / block;
    QwenVLResizeNormalizePatchLayoutKernel<<<grid, block, 0, stream>>>(
        input,
        output,
        srcHeight,
        srcWidth,
        dstHeight,
        dstWidth,
        inputChannels,
        channelOrder,
        patchSize,
        temporalPatchSize,
        mergeSize,
        inputScale,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        patchCount,
        featureDim);
    return cudaGetLastError();
}
