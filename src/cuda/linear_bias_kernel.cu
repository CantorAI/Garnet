// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_lib.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mutex>

constexpr int GARNET_LINEAR_TILE = 16;

namespace
{
    cublasHandle_t gBFloat16GemmHandle = nullptr;
    cudaStream_t gBFloat16GemmStream = nullptr;
    std::mutex gBFloat16GemmMutex;

    struct BFloat16Workspace
    {
        __nv_bfloat16* data = nullptr;
        size_t capacity = 0;
    };

    BFloat16Workspace gBFloat16Workspace;

    __global__ void garnet_fp32_to_bf16_kernel(
        const float* input,
        __nv_bfloat16* output,
        int count) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count) output[index] = __float2bfloat16(input[index]);
    }

    __global__ void garnet_add_bf16_bias_fp32_kernel(
        float* output,
        const __nv_bfloat16* bias,
        int rows,
        int columns) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int count = rows * columns;
        if (index < count) output[index] += __bfloat162float(bias[index % columns]);
    }

    cudaError_t RunCublasLinearTransposeBF16WeightFP32(
        const float* input,
        const __nv_bfloat16* weight,
        float* output,
        int rows,
        int inFeatures,
        int outFeatures,
        cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(gBFloat16GemmMutex);
        if (!gBFloat16GemmHandle && cublasCreate(&gBFloat16GemmHandle) != CUBLAS_STATUS_SUCCESS) {
            return cudaErrorInitializationError;
        }
        if (gBFloat16GemmStream != stream) {
            if (cublasSetStream(gBFloat16GemmHandle, stream) != CUBLAS_STATUS_SUCCESS) {
                return cudaErrorInvalidResourceHandle;
            }
            if (cublasSetWorkspace(gBFloat16GemmHandle, nullptr, 0) != CUBLAS_STATUS_SUCCESS) {
                return cudaErrorInvalidResourceHandle;
            }
            gBFloat16GemmStream = stream;
        }
        size_t inputElements = static_cast<size_t>(rows) * inFeatures;
        BFloat16Workspace& workspace = gBFloat16Workspace;
        if (workspace.capacity < inputElements) {
            if (workspace.data) {
                cudaError_t status = cudaStreamSynchronize(stream);
                if (status != cudaSuccess) return status;
                cudaFree(workspace.data);
                workspace.data = nullptr;
                workspace.capacity = 0;
            }
            cudaError_t status = cudaMalloc(&workspace.data, inputElements * sizeof(__nv_bfloat16));
            if (status != cudaSuccess) return status;
            workspace.capacity = inputElements;
        }
        __nv_bfloat16* inputBF16 = workspace.data;
        int blocks = (static_cast<int>(inputElements) + 255) / 256;
        garnet_fp32_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
            input, inputBF16, static_cast<int>(inputElements));
        cudaError_t status = cudaGetLastError();
        if (status == cudaSuccess) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t gemmStatus = cublasGemmEx(
                gBFloat16GemmHandle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                outFeatures,
                rows,
                inFeatures,
                &alpha,
                weight,
                CUDA_R_16BF,
                inFeatures,
                inputBF16,
                CUDA_R_16BF,
                inFeatures,
                &beta,
                output,
                CUDA_R_32F,
                outFeatures,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (gemmStatus != CUBLAS_STATUS_SUCCESS) status = cudaErrorLaunchFailure;
        }
        return status;
    }
}

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

extern "C" cudaError_t runLinearTransposeBF16WeightFP32(
    const float* input,
    const __nv_bfloat16* weight,
    float* output,
    int rows,
    int inFeatures,
    int outFeatures,
    cudaStream_t stream) {
    if (!input || !weight || !output || rows <= 0 || inFeatures <= 0 || outFeatures <= 0) {
        return cudaErrorInvalidValue;
    }
    return RunCublasLinearTransposeBF16WeightFP32(
        input, weight, output, rows, inFeatures, outFeatures, stream);
}

extern "C" cudaError_t runLinearBiasTransposeBF16WeightFP32(
    const float* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    float* output,
    int rows,
    int inFeatures,
    int outFeatures,
    cudaStream_t stream) {
    if (!input || !weight || !bias || !output || rows <= 0 || inFeatures <= 0 || outFeatures <= 0) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = RunCublasLinearTransposeBF16WeightFP32(
        input, weight, output, rows, inFeatures, outFeatures, stream);
    if (status != cudaSuccess) return status;
    int count = rows * outFeatures;
    garnet_add_bf16_bias_fp32_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        output, bias, rows, outFeatures);
    return cudaGetLastError();
}

__global__ void garnet_pack_qkv_head_norm_bf16_kernel(
    const float* q,
    const float* k,
    const float* v,
    const __nv_bfloat16* qNorm,
    const __nv_bfloat16* kNorm,
    float* output,
    int tokens,
    int qOut,
    int kOut,
    int headDim,
    float epsilon) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int packed = qOut + 2 * kOut;
    int count = tokens * packed;
    if (index >= count) return;
    int token = index / packed;
    int feature = index % packed;
    if (feature >= qOut + kOut) {
        output[index] = v[token * kOut + feature - qOut - kOut];
        return;
    }
    const float* source = feature < qOut ? q : k;
    const __nv_bfloat16* norm = feature < qOut ? qNorm : kNorm;
    int local = feature < qOut ? feature : feature - qOut;
    int width = feature < qOut ? qOut : kOut;
    int headStart = (local / headDim) * headDim;
    float squareSum = 0.0f;
    for (int dim = 0; dim < headDim; ++dim) {
        float value = source[token * width + headStart + dim];
        squareSum += value * value;
    }
    float invRms = rsqrtf(squareSum / static_cast<float>(headDim) + epsilon);
    output[index] = source[token * width + local] * invRms * __bfloat162float(norm[local % headDim]);
}

extern "C" cudaError_t runQKVHeadNormBF16WeightFP32(
    const float* input,
    const __nv_bfloat16* qWeight,
    const __nv_bfloat16* kWeight,
    const __nv_bfloat16* vWeight,
    const __nv_bfloat16* qNormWeight,
    const __nv_bfloat16* kNormWeight,
    float* packedOutput,
    int tokens,
    int hidden,
    int qOut,
    int kOut,
    int headDim,
    float epsilon,
    cudaStream_t stream) {
    if (!input || !qWeight || !kWeight || !vWeight || !qNormWeight || !kNormWeight ||
        !packedOutput || tokens <= 0 || hidden <= 0 || qOut <= 0 || kOut <= 0 || headDim <= 0) {
        return cudaErrorInvalidValue;
    }
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    size_t qBytes = static_cast<size_t>(tokens) * qOut * sizeof(float);
    size_t kvBytes = static_cast<size_t>(tokens) * kOut * sizeof(float);
    cudaError_t status = cudaMallocAsync(&q, qBytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(&k, kvBytes, stream);
    if (status == cudaSuccess) status = cudaMallocAsync(&v, kvBytes, stream);
    if (status == cudaSuccess) status = runLinearTransposeBF16WeightFP32(input, qWeight, q, tokens, hidden, qOut, stream);
    if (status == cudaSuccess) status = runLinearTransposeBF16WeightFP32(input, kWeight, k, tokens, hidden, kOut, stream);
    if (status == cudaSuccess) status = runLinearTransposeBF16WeightFP32(input, vWeight, v, tokens, hidden, kOut, stream);
    if (status == cudaSuccess) {
        int count = tokens * (qOut + 2 * kOut);
        int blocks = (count + 255) / 256;
        garnet_pack_qkv_head_norm_bf16_kernel<<<blocks, 256, 0, stream>>>(
            q, k, v, qNormWeight, kNormWeight, packedOutput,
            tokens, qOut, kOut, headDim, epsilon);
        status = cudaGetLastError();
    }
    if (q) cudaFreeAsync(q, stream);
    if (k) cudaFreeAsync(k, stream);
    if (v) cudaFreeAsync(v, stream);
    return status;
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
