#include "cuda_lib.h"

#include <cuda_runtime.h>

constexpr int GARNET_LINEAR_TILE = 16;

__global__ void garnet_linear_bias_transpose_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int rows,
    int inFeatures,
    int outFeatures) {
    int row = blockIdx.y * GARNET_LINEAR_TILE + threadIdx.y;
    int out = blockIdx.x * GARNET_LINEAR_TILE + threadIdx.x;

    __shared__ float aTile[GARNET_LINEAR_TILE][GARNET_LINEAR_TILE];
    __shared__ float wTile[GARNET_LINEAR_TILE][GARNET_LINEAR_TILE];

    float sum = 0.0f;
    for (int kBase = 0; kBase < inFeatures; kBase += GARNET_LINEAR_TILE) {
        int aK = kBase + threadIdx.x;
        int wK = kBase + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < rows && aK < inFeatures)
            ? input[row * inFeatures + aK]
            : 0.0f;
        wTile[threadIdx.y][threadIdx.x] = (out < outFeatures && wK < inFeatures)
            ? weight[out * inFeatures + wK]
            : 0.0f;
        __syncthreads();

        for (int k = 0; k < GARNET_LINEAR_TILE; ++k) {
            sum += aTile[threadIdx.y][k] * wTile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && out < outFeatures) {
        output[row * outFeatures + out] = sum + bias[out];
    }
}

extern "C" cudaError_t runLinearBiasTransposeFP32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int inFeatures,
    int outFeatures,
    cudaStream_t stream) {
    if (!input || !weight || !bias || !output || rows <= 0 || inFeatures <= 0 || outFeatures <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(GARNET_LINEAR_TILE, GARNET_LINEAR_TILE);
    dim3 grid(
        (outFeatures + GARNET_LINEAR_TILE - 1) / GARNET_LINEAR_TILE,
        (rows + GARNET_LINEAR_TILE - 1) / GARNET_LINEAR_TILE);
    garnet_linear_bias_transpose_fp32_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, rows, inFeatures, outFeatures);
    return cudaGetLastError();
}

__global__ void garnet_linear_transpose_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int rows,
    int inFeatures,
    int outFeatures) {
    int row = blockIdx.y * GARNET_LINEAR_TILE + threadIdx.y;
    int out = blockIdx.x * GARNET_LINEAR_TILE + threadIdx.x;

    __shared__ float aTile[GARNET_LINEAR_TILE][GARNET_LINEAR_TILE];
    __shared__ float wTile[GARNET_LINEAR_TILE][GARNET_LINEAR_TILE];

    float sum = 0.0f;
    for (int kBase = 0; kBase < inFeatures; kBase += GARNET_LINEAR_TILE) {
        int aK = kBase + threadIdx.x;
        int wK = kBase + threadIdx.y;
        aTile[threadIdx.y][threadIdx.x] = (row < rows && aK < inFeatures)
            ? input[row * inFeatures + aK]
            : 0.0f;
        wTile[threadIdx.y][threadIdx.x] = (out < outFeatures && wK < inFeatures)
            ? weight[out * inFeatures + wK]
            : 0.0f;
        __syncthreads();

        for (int k = 0; k < GARNET_LINEAR_TILE; ++k) {
            sum += aTile[threadIdx.y][k] * wTile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && out < outFeatures) {
        output[row * outFeatures + out] = sum;
    }
}

extern "C" cudaError_t runLinearTransposeFP32(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int inFeatures,
    int outFeatures,
    cudaStream_t stream) {
    if (!input || !weight || !output || rows <= 0 || inFeatures <= 0 || outFeatures <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(GARNET_LINEAR_TILE, GARNET_LINEAR_TILE);
    dim3 grid(
        (outFeatures + GARNET_LINEAR_TILE - 1) / GARNET_LINEAR_TILE,
        (rows + GARNET_LINEAR_TILE - 1) / GARNET_LINEAR_TILE);
    garnet_linear_transpose_fp32_kernel<<<grid, block, 0, stream>>>(
        input, weight, output, rows, inFeatures, outFeatures);
    return cudaGetLastError();
}

__global__ void garnet_silu_mul_fp32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    float g = gate[idx];
    output[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

extern "C" cudaError_t runSiluMulFP32(
    const float* gate,
    const float* up,
    float* output,
    int count,
    cudaStream_t stream) {
    if (!gate || !up || !output || count <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    garnet_silu_mul_fp32_kernel<<<gridSize, blockSize, 0, stream>>>(gate, up, output, count);
    return cudaGetLastError();
}
