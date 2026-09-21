// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

// init_ops_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half, __float2half, __half2float
#include <cuda_bf16.h>    // For __nv_bfloat16, __bfloat162float, __float2bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2 if needed
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <type_traits>
#include "cuda_lib.h"
//--------------------------------------------------------
// Basic initialization kernels (zeros, ones, full)
//--------------------------------------------------------

template <typename T>
__global__ void init_kernel_zeros(T* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<T>(0);
    }
}

template <typename T>
__global__ void init_kernel_ones(T* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = static_cast<T>(1);
    }
}

template <typename T>
__global__ void init_kernel_full(T* output, int numel, T value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = value;
    }
}

//--------------------------------------------------------
// Uniform initialization kernel
//--------------------------------------------------------

template <typename T>
__global__ void init_kernel_uniform(T* output, int numel, T low, T high, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        unsigned int localSeed = seed ^ idx;
        localSeed = 1664525u * localSeed + 1013904223u;
        float r = (float)localSeed / (float)UINT_MAX;

        if constexpr (std::is_same<T, __half>::value) {
            float f_low = __half2float(low);
            float f_high = __half2float(high);
            float result = f_low + r * (f_high - f_low);
            output[idx] = __float2half(result);
        }
        else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            float f_low = __bfloat162float(low);
            float f_high = __bfloat162float(high);
            float result = f_low + r * (f_high - f_low);
            output[idx] = __float2bfloat16(result);
        }
        else {
            output[idx] = low + r * (high - low);
        }
    }
}

//--------------------------------------------------------
// Randn kernel using Box-Muller transform (mean=0, std=1)
//--------------------------------------------------------

template <typename T>
__global__ void init_kernel_randn(T* output, int numel, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        unsigned int localSeed = seed ^ (idx * 2);
        localSeed = 1664525u * localSeed + 1013904223u;
        float u1 = ((float)localSeed + 1.0f) / (float)UINT_MAX;
        localSeed = 1664525u * localSeed + 1013904223u;
        float u2 = ((float)localSeed + 1.0f) / (float)UINT_MAX;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;
        float z = r * cosf(theta);
        if constexpr (std::is_same<T, __half>::value) {
            output[idx] = __float2half(z);
        }
        else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            output[idx] = __float2bfloat16(z);
        }
        else {
            output[idx] = static_cast<T>(z);
        }
    }
}

//--------------------------------------------------------
// Normal initialization kernel with specified mean and std
//--------------------------------------------------------

template <typename T>
__global__ void init_kernel_normal(T* output, int numel, float mean, float std, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        unsigned int localSeed = seed ^ (idx * 2);
        localSeed = 1664525u * localSeed + 1013904223u;
        float u1 = ((float)localSeed + 1.0f) / (float)UINT_MAX;
        localSeed = 1664525u * localSeed + 1013904223u;
        float u2 = ((float)localSeed + 1.0f) / (float)UINT_MAX;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;
        float z = r * cosf(theta);
        float result = mean + std * z;
        if constexpr (std::is_same<T, __half>::value) {
            output[idx] = __float2half(result);
        }
        else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            output[idx] = __float2bfloat16(result);
        }
        else {
            output[idx] = static_cast<T>(result);
        }
    }
}

//--------------------------------------------------------
// Truncated Normal initialization kernel
//--------------------------------------------------------

template <typename T>
__global__ void init_kernel_trunc_normal(T* output, int numel, float mean, float std, float a, float b, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        unsigned int localSeed = seed ^ (idx * 2);
        float value;
        bool accepted = false;
        for (int iter = 0; iter < 10; ++iter) {
            localSeed = 1664525u * localSeed + 1013904223u;
            float u1 = ((float)localSeed + 1.0f) / (float)UINT_MAX;
            localSeed = 1664525u * localSeed + 1013904223u;
            float u2 = ((float)localSeed + 1.0f) / (float)UINT_MAX;
            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * 3.14159265358979323846f * u2;
            float z = r * cosf(theta);
            value = mean + std * z;
            if (value >= a && value <= b) {
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            if (value < a) value = a;
            if (value > b) value = b;
        }
        if constexpr (std::is_same<T, __half>::value) {
            output[idx] = __float2half(value);
        }
        else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            output[idx] = __float2bfloat16(value);
        }
        else {
            output[idx] = static_cast<T>(value);
        }
    }
}

//--------------------------------------------------------
// Extern "C" wrappers for host calls (non-template functions only)
//--------------------------------------------------------

extern "C" {

    // Zeros
    void runInitZerosFloat(float* output, int numel) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_zeros<float> << <blocks, threadsPerBlock >> > (output, numel);
        cudaDeviceSynchronize();
    }
    void runInitZerosFP16(__half* output, int numel) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_zeros<__half> << <blocks, threadsPerBlock >> > (output, numel);
        cudaDeviceSynchronize();
    }
    void runInitZerosBF16(__nv_bfloat16* output, int numel) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_zeros<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel);
        cudaDeviceSynchronize();
    }

    // Ones
    void runInitOnesFloat(float* output, int numel) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_ones<float> << <blocks, threadsPerBlock >> > (output, numel);
        cudaDeviceSynchronize();
    }
    void runInitOnesFP16(__half* output, int numel) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_ones<__half> << <blocks, threadsPerBlock >> > (output, numel);
        cudaDeviceSynchronize();
    }
    void runInitOnesBF16(__nv_bfloat16* output, int numel) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_ones<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel);
        cudaDeviceSynchronize();
    }

    // Full (constant value)
    void runInitFullFloat(float* output, int numel, float value) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_full<float> << <blocks, threadsPerBlock >> > (output, numel, value);
        cudaDeviceSynchronize();
    }
    void runInitFullFP16(__half* output, int numel, __half value) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_full<__half> << <blocks, threadsPerBlock >> > (output, numel, value);
        cudaDeviceSynchronize();
    }
    void runInitFullBF16(__nv_bfloat16* output, int numel, __nv_bfloat16 value) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_full<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel, value);
        cudaDeviceSynchronize();
    }

    // Rand (uniform [0,1))
    void runInitRandFloat(float* output, int numel, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_uniform<float> << <blocks, threadsPerBlock >> > (output, numel, 0.0f, 1.0f, seed);
        cudaDeviceSynchronize();
    }
    void runInitRandFP16(__half* output, int numel, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_uniform<__half> << <blocks, threadsPerBlock >> > (output, numel, __float2half(0.0f), __float2half(1.0f), seed);
        cudaDeviceSynchronize();
    }
    void runInitRandBF16(__nv_bfloat16* output, int numel, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_uniform<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel, __float2bfloat16(0.0f), __float2bfloat16(1.0f), seed);
        cudaDeviceSynchronize();
    }

    // Randn (normal distribution, mean=0, std=1)
    void runInitRandnFloat(float* output, int numel, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_randn<float> << <blocks, threadsPerBlock >> > (output, numel, seed);
        cudaDeviceSynchronize();
    }
    void runInitRandnFP16(__half* output, int numel, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_randn<__half> << <blocks, threadsPerBlock >> > (output, numel, seed);
        cudaDeviceSynchronize();
    }
    void runInitRandnBF16(__nv_bfloat16* output, int numel, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_randn<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel, seed);
        cudaDeviceSynchronize();
    }

    // Uniform distribution in [low, high)
    void runInitUniformFloat(float* output, int numel, float low, float high, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_uniform<float> << <blocks, threadsPerBlock >> > (output, numel, low, high, seed);
        cudaDeviceSynchronize();
    }
    void runInitUniformFP16(__half* output, int numel, __half low, __half high, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_uniform<__half> << <blocks, threadsPerBlock >> > (output, numel, low, high, seed);
        cudaDeviceSynchronize();
    }
    void runInitUniformBF16(__nv_bfloat16* output, int numel, __nv_bfloat16 low, __nv_bfloat16 high, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_uniform<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel, low, high, seed);
        cudaDeviceSynchronize();
    }

    // Normal distribution with specified mean and std
    void runInitNormalFloat(float* output, int numel, float mean, float std, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_normal<float> << <blocks, threadsPerBlock >> > (output, numel, mean, std, seed);
        cudaDeviceSynchronize();
    }
    void runInitNormalFP16(__half* output, int numel, float mean, float std, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_normal<__half> << <blocks, threadsPerBlock >> > (output, numel, mean, std, seed);
        cudaDeviceSynchronize();
    }
    void runInitNormalBF16(__nv_bfloat16* output, int numel, float mean, float std, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_normal<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel, mean, std, seed);
        cudaDeviceSynchronize();
    }

    // Truncated Normal distribution
    void runInitTruncNormalFloat(float* output, int numel, float mean, float std, float a, float b, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_trunc_normal<float> << <blocks, threadsPerBlock >> > (output, numel, mean, std, a, b, seed);
        cudaDeviceSynchronize();
    }
    void runInitTruncNormalFP16(__half* output, int numel, float mean, float std, float a, float b, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_trunc_normal<__half> << <blocks, threadsPerBlock >> > (output, numel, mean, std, a, b, seed);
        cudaDeviceSynchronize();
    }
    void runInitTruncNormalBF16(__nv_bfloat16* output, int numel, float mean, float std, float a, float b, unsigned int seed) {
        int threadsPerBlock = 256;
        int blocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
        init_kernel_trunc_normal<__nv_bfloat16> << <blocks, threadsPerBlock >> > (output, numel, mean, std, a, b, seed);
        cudaDeviceSynchronize();
    }
}
