#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
//#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include "cuda_lib.h"

namespace cg = cooperative_groups;

// ==============================================
// 通用工具函数和常量定义
// ==============================================

constexpr int BLOCK_SIZE = 256;
constexpr int TILE_DIM = 32;

// 计算线性索引
__device__ int computeLinearIndex(int* dims, int dimCount, int* indices) {
    int index = 0;
    int stride = 1;
    for (int i = dimCount - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= dims[i];
    }
    return index;
}
/*
// ==============================================
// 矩阵乘法(GEMM)实现
// ==============================================

template <typename T, int TILE_SIZE = 32>
__global__ void gemm_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    // 线程块平铺
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 线程平铺
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 每个线程块计算的C矩阵块
    const int cRow = by * TILE_SIZE;
    const int cCol = bx * TILE_SIZE;

    // 共享内存声明
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    T cVal = (T)0;

    // 循环遍历平铺
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载A和B的平铺到共享内存
        int aRow = cRow + ty;
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        int bCol = cCol + tx;

        if (aRow < M && aCol < K) {
            As[ty][tx] = A[aRow * K + aCol];
        }
        else {
            As[ty][tx] = (T)0;
        }

        if (bRow < K && bCol < N) {
            Bs[ty][tx] = B[bRow * N + bCol];
        }
        else {
            Bs[ty][tx] = (T)0;
        }

        __syncthreads();

        // 计算部分结果
        for (int k = 0; k < TILE_SIZE; ++k) {
            cVal =  T(float(cVal) + float(As[ty][k]) * float(Bs[k][tx]));
        }

        __syncthreads();
    }

    // 存储结果
    int cIdx = (cRow + ty) * N + (cCol + tx);
    if ((cRow + ty) < M && (cCol + tx) < N) {
        C[cIdx] = cVal;
    }
}

void runGemmFP32(float* A, float* B, float* C, int m, int k, int n) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM);
    gemm_kernel<float> << <grid, block >> > (A, B, C, m, n, k);
    cudaDeviceSynchronize();
}

void runGemmFP16(__half* A, __half* B, __half* C, int m, int k, int n) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM);
    gemm_kernel<__half> << <grid, block >> > (A, B, C, m, n, k);
    cudaDeviceSynchronize();
}

void runGemmBF16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int m, int k, int n) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM);
    gemm_kernel<__nv_bfloat16> << <grid, block >> > (A, B, C, m, n, k);
    cudaDeviceSynchronize();
}

void runGemmFP8E4M3(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* C, int m, int k, int n) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM);
    gemm_kernel<__nv_fp8_e4m3> << <grid, block >> > (A, B, C, m, n, k);
    cudaDeviceSynchronize();
}

void runGemmFP8E5M2(__nv_fp8_e5m2* A, __nv_fp8_e5m2* B, __nv_fp8_e5m2* C, int m, int k, int n) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM);
    gemm_kernel<__nv_fp8_e5m2> << <grid, block >> > (A, B, C, m, n, k);
    cudaDeviceSynchronize();
}
*/

// ==============================================
// 逐元素乘法实现
// ==============================================

template <typename T>
__global__ void elementwise_multiply_kernel(const T* A, const T* B, T* C, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] = T(float( A[idx]) * float( B[0]));
    }
}

void runSingleElementTensorMultiplyFP32(float* multi, float* single, float* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_multiply_kernel<float> << <gridSize, BLOCK_SIZE >> > (multi, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMultiplyFP16(__half* multi, __half* single, __half* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_multiply_kernel<__half> << <gridSize, BLOCK_SIZE >> > (multi, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMultiplyBF16(__nv_bfloat16* multi, __nv_bfloat16* single, __nv_bfloat16* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_multiply_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (multi, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMultiplyFP8E4M3(__nv_fp8_e4m3* multi, __nv_fp8_e4m3* single, __nv_fp8_e4m3* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_multiply_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (multi, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMultiplyFP8E5M2(__nv_fp8_e5m2* multi, __nv_fp8_e5m2* single, __nv_fp8_e5m2* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    elementwise_multiply_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (multi, single, result, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 标量乘法实现
// ==============================================

template <typename T>
__global__ void scalar_multiply_kernel(const T* input, T* result, float scalar, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] =T( float( input[idx]) * float(static_cast<T>(scalar)));
    }
}

void runScalarMultiplyFP32(float* input, float* result, float scalar, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_multiply_kernel<float> << <gridSize, BLOCK_SIZE >> > (input, result, scalar, count);
    cudaDeviceSynchronize();
}

void runScalarMultiplyFP16(__half* input, __half* result, float scalar, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_multiply_kernel<__half> << <gridSize, BLOCK_SIZE >> > (input, result, scalar, count);
    cudaDeviceSynchronize();
}

void runScalarMultiplyBF16(__nv_bfloat16* input, __nv_bfloat16* result, float scalar, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_multiply_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (input, result, scalar, count);
    cudaDeviceSynchronize();
}

void runScalarMultiplyFP8E4M3(__nv_fp8_e4m3* input, __nv_fp8_e4m3* result, float scalar, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_multiply_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (input, result, scalar, count);
    cudaDeviceSynchronize();
}

void runScalarMultiplyFP8E5M2(__nv_fp8_e5m2* input, __nv_fp8_e5m2* result, float scalar, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_multiply_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (input, result, scalar, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 加法实现
// ==============================================

template <typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] =T(float( A[idx]) +float( B[idx]));
    }
}

void runAddFP64(double* A, double* B, double* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<double> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runAddFP32(float* A, float* B, float* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<float> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runAddFP16(__half* A, __half* B, __half* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<__half> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runAddBF16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runAddFP8E4M3(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runAddFP8E5M2(__nv_fp8_e5m2* A, __nv_fp8_e5m2* B, __nv_fp8_e5m2* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 单元素张量加法实现
// ==============================================

template <typename T>
__global__ void single_element_add_kernel(const T* tensor, T single, T* result, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = T(float(tensor[idx]) + float(single));
    }
}

void runSingleElementTensorAddFP64(double* tensor, double single, double* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_add_kernel<double> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorAddFP32(float* tensor, float single, float* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_add_kernel<float> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorAddFP16(__half* tensor, __half single, __half* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_add_kernel<__half> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorAddBF16(__nv_bfloat16* tensor, __nv_bfloat16 single, __nv_bfloat16* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_add_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorAddFP8E4M3(__nv_fp8_e4m3* tensor, __nv_fp8_e4m3 single, __nv_fp8_e4m3* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_add_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorAddFP8E5M2(__nv_fp8_e5m2* tensor, __nv_fp8_e5m2 single, __nv_fp8_e5m2* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_add_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 标量加法实现
// ==============================================

void runScalarAddFP32(float* input, float scalar, float* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_multiply_kernel<float> << <gridSize, BLOCK_SIZE >> > (input, result, scalar, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 减法实现
// ==============================================

template <typename T>
__global__ void minus_kernel(const T* A, const T* B, T* C, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        C[idx] =T(float( A[idx]) - float(B[idx]));
    }
}

void runMinusFP64(double* A, double* B, double* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    minus_kernel<double> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runMinusFP32(float* A, float* B, float* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    minus_kernel<float> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runMinusFP16(__half* A, __half* B, __half* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    minus_kernel<__half> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runMinusBF16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    minus_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runMinusFP8E4M3(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    minus_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

void runMinusFP8E5M2(__nv_fp8_e5m2* A, __nv_fp8_e5m2* B, __nv_fp8_e5m2* C, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    minus_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (A, B, C, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 单元素张量减法实现
// ==============================================

template <typename T>
__global__ void single_element_minus_kernel(const T* tensor, T single, T* result, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = T(float(tensor[idx]) - float(single));
    }
}

void runSingleElementTensorMinusFP64(double* tensor, double single, double* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_minus_kernel<double> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMinusFP32(float* tensor, float single, float* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_minus_kernel<float> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMinusFP16(__half* tensor, __half single, __half* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_minus_kernel<__half> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMinusBF16(__nv_bfloat16* tensor, __nv_bfloat16 single, __nv_bfloat16* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_minus_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMinusFP8E4M3(__nv_fp8_e4m3* tensor, __nv_fp8_e4m3 single, __nv_fp8_e4m3* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_minus_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

void runSingleElementTensorMinusFP8E5M2(__nv_fp8_e5m2* tensor, __nv_fp8_e5m2 single, __nv_fp8_e5m2* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    single_element_minus_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (tensor, single, result, count);
    cudaDeviceSynchronize();
}

// ==============================================
// 标量减法实现
// ==============================================

template <typename T>
__global__ void scalar_minus_kernel(const T* input, T scalar, T* result, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = input[idx] - scalar;
    }
}

void runScalarMinusFP32(float* input, float scalar, float* result, int count) {
    int gridSize = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scalar_minus_kernel<float> << <gridSize, BLOCK_SIZE >> > (input, scalar, result, count);
    cudaDeviceSynchronize();
}
/*
// ==============================================
// 矩阵乘法(Matmul)实现
// ==============================================

void runMatmulFP32(float* A, float* B, float* C, int m, int n, int k) {
    runGemmFP32(A, B, C, m, k, n);
}

void runMatmulFP16(__half* A, __half* B, __half* C, int m, int n, int k) {
    runGemmFP16(A, B, C, m, k, n);
}

void runMatmulBF16(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int m, int n, int k) {
    runGemmBF16(A, B, C, m, k, n);
}

void runMatmulFP8E4M3(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, __nv_fp8_e4m3* C, int m, int n, int k) {
    runGemmFP8E4M3(A, B, C, m, k, n);
}

void runMatmulFP8E5M2(__nv_fp8_e5m2* A, __nv_fp8_e5m2* B, __nv_fp8_e5m2* C, int m, int n, int k) {
    runGemmFP8E5M2(A, B, C, m, k, n);
}
*/
// ==============================================
// 转置(Permute)实现
// ==============================================

template <typename T>
__global__ void permute_kernel(const T* input, T* output, int* permOrder, int* dimSizes, int dimCount, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    // 计算原始坐标
    int temp = idx;
    int originalCoords[8]; // 支持最多8维张量
    for (int i = dimCount - 1; i >= 0; --i) {
        originalCoords[i] = temp % dimSizes[i];
        temp /= dimSizes[i];
    }

    // 计算转置后坐标
    int transposedCoords[8];
    for (int i = 0; i < dimCount; ++i) {
        transposedCoords[i] = originalCoords[permOrder[i]];
    }

    // 计算转置后线性索引
    int transposedIdx = 0;
    int stride = 1;
    for (int i = dimCount - 1; i >= 0; --i) {
        transposedIdx += transposedCoords[i] * stride;
        stride *= dimSizes[permOrder[i]];
    }

    output[transposedIdx] = input[idx];
}

void runPermuteFP32(float* input, float* output, int* permOrder, int* dimSizes, int dimCount) {
    int totalElements = 1;
    for (int i = 0; i < dimCount; ++i) {
        totalElements *= dimSizes[i];
    }

    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_kernel<float> << <gridSize, BLOCK_SIZE >> > (input, output, permOrder, dimSizes, dimCount, totalElements);
    cudaDeviceSynchronize();
}

void runPermuteFP16(__half* input, __half* output, int* permOrder, int* dimSizes, int dimCount) {
    int totalElements = 1;
    for (int i = 0; i < dimCount; ++i) {
        totalElements *= dimSizes[i];
    }

    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_kernel<__half> << <gridSize, BLOCK_SIZE >> > (input, output, permOrder, dimSizes, dimCount, totalElements);
    cudaDeviceSynchronize();
}

void runPermuteBF16(__nv_bfloat16* input, __nv_bfloat16* output, int* permOrder, int* dimSizes, int dimCount) {
    int totalElements = 1;
    for (int i = 0; i < dimCount; ++i) {
        totalElements *= dimSizes[i];
    }

    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_kernel<__nv_bfloat16> << <gridSize, BLOCK_SIZE >> > (input, output, permOrder, dimSizes, dimCount, totalElements);
    cudaDeviceSynchronize();
}

void runPermuteFP8E4M3(__nv_fp8_e4m3* input, __nv_fp8_e4m3* output, int* permOrder, int* dimSizes, int dimCount) {
    int totalElements = 1;
    for (int i = 0; i < dimCount; ++i) {
        totalElements *= dimSizes[i];
    }

    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_kernel<__nv_fp8_e4m3> << <gridSize, BLOCK_SIZE >> > (input, output, permOrder, dimSizes, dimCount, totalElements);
    cudaDeviceSynchronize();
}

void runPermuteFP8E5M2(__nv_fp8_e5m2* input, __nv_fp8_e5m2* output, int* permOrder, int* dimSizes, int dimCount) {
    int totalElements = 1;
    for (int i = 0; i < dimCount; ++i) {
        totalElements *= dimSizes[i];
    }

    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_kernel<__nv_fp8_e5m2> << <gridSize, BLOCK_SIZE >> > (input, output, permOrder, dimSizes, dimCount, totalElements);
    cudaDeviceSynchronize();
}

// ==============================================
// Gather操作实现
// ==============================================

template <typename T>
__global__ void gather_kernel(const T* data, const int* indices, T* output,
    int* dataDims, int dataDimCount,
    int* indicesDims, int indicesDimCount, int dim) {

    // 计算每个索引对应的元素数量
    int elementsPerIndex = 1;
    for (int i = dim + 1; i < dataDimCount; ++i) {
        elementsPerIndex *= dataDims[i];
    }

    // 计算总索引数
    int totalIndices = 1;
    for (int i = 0; i < indicesDimCount; ++i) {
        totalIndices *= indicesDims[i];
    }

    // 每个线程处理一个输出元素
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int element = blockIdx.y * blockDim.y + threadIdx.y;

    if (index < totalIndices && element < elementsPerIndex) {
        // 获取要收集的索引
        int gatherIdx = indices[index];

        // 计算输入数据偏移
        int dataOffset = gatherIdx * elementsPerIndex + element;

        // 计算输出位置
        int outputOffset = index * elementsPerIndex + element;

        // 执行收集操作
        output[outputOffset] = data[dataOffset];
    }
}

void runGatherFP32(float* data, float* indices, float* output,
    int* dataDims, int dataDimCount,
    int* indicesDims, int indicesDimCount, int dim) {
    // 计算总索引数和每个索引对应的元素数
    int totalIndices = 1;
    for (int i = 0; i < indicesDimCount; ++i) {
        totalIndices *= indicesDims[i];
    }

    int elementsPerIndex = 1;
    for (int i = dim + 1; i < dataDimCount; ++i) {
        elementsPerIndex *= dataDims[i];
    }

    // 配置内核启动参数
    dim3 block(16, 16);
    dim3 grid((totalIndices + block.x - 1) / block.x,
        (elementsPerIndex + block.y - 1) / block.y);

    gather_kernel<float> << <grid, block >> > (
        data, (int*)indices, output,
        dataDims, dataDimCount,
        indicesDims, indicesDimCount, dim);

    cudaDeviceSynchronize();
}

void runGatherFP16(__half* data, __half* indices, __half* output,
    int* dataDims, int dataDimCount,
    int* indicesDims, int indicesDimCount, int dim) {
    int totalIndices = 1;
    for (int i = 0; i < indicesDimCount; ++i) {
        totalIndices *= indicesDims[i];
    }

    int elementsPerIndex = 1;
    for (int i = dim + 1; i < dataDimCount; ++i) {
        elementsPerIndex *= dataDims[i];
    }

    dim3 block(16, 16);
    dim3 grid((totalIndices + block.x - 1) / block.x,
        (elementsPerIndex + block.y - 1) / block.y);

    gather_kernel<__half> << <grid, block >> > (
        data, (int*)indices, output,
        dataDims, dataDimCount,
        indicesDims, indicesDimCount, dim);

    cudaDeviceSynchronize();
}

void runGatherBF16(__nv_bfloat16* data, int* indices, __nv_bfloat16* output,
    int* dataDims, int dataDimCount,
    int* indicesDims, int indicesDimCount, int dim) {
    int totalIndices = 1;
    for (int i = 0; i < indicesDimCount; ++i) {
        totalIndices *= indicesDims[i];
    }

    int elementsPerIndex = 1;
    for (int i = dim + 1; i < dataDimCount; ++i) {
        elementsPerIndex *= dataDims[i];
    }

    dim3 block(16, 16);
    dim3 grid((totalIndices + block.x - 1) / block.x,
        (elementsPerIndex + block.y - 1) / block.y);

    gather_kernel<__nv_bfloat16> << <grid, block >> > (
        data, indices, output,
        dataDims, dataDimCount,
        indicesDims, indicesDimCount, dim);

    cudaDeviceSynchronize();
}

void runGatherFP8E4M3(__nv_fp8_e4m3* data, __nv_fp8_e4m3* indices, __nv_fp8_e4m3* output,
    int* dataDims, int dataDimCount,
    int* indicesDims, int indicesDimCount, int dim) {
    int totalIndices = 1;
    for (int i = 0; i < indicesDimCount; ++i) {
        totalIndices *= indicesDims[i];
    }

    int elementsPerIndex = 1;
    for (int i = dim + 1; i < dataDimCount; ++i) {
        elementsPerIndex *= dataDims[i];
    }

    dim3 block(16, 16);
    dim3 grid((totalIndices + block.x - 1) / block.x,
        (elementsPerIndex + block.y - 1) / block.y);

    gather_kernel<__nv_fp8_e4m3> << <grid, block >> > (
        data, (int*)indices, output,
        dataDims, dataDimCount,
        indicesDims, indicesDimCount, dim);

    cudaDeviceSynchronize();
}

void runGatherFP8E5M2(__nv_fp8_e5m2* data, __nv_fp8_e5m2* indices, __nv_fp8_e5m2* output,
    int* dataDims, int dataDimCount,
    int* indicesDims, int indicesDimCount, int dim) {
    int totalIndices = 1;
    for (int i = 0; i < indicesDimCount; ++i) {
        totalIndices *= indicesDims[i];
    }

    int elementsPerIndex = 1;
    for (int i = dim + 1; i < dataDimCount; ++i) {
        elementsPerIndex *= dataDims[i];
    }

    dim3 block(16, 16);
    dim3 grid((totalIndices + block.x - 1) / block.x,
        (elementsPerIndex + block.y - 1) / block.y);

    gather_kernel<__nv_fp8_e5m2> << <grid, block >> > (
        data, (int*)indices, output,
        dataDims, dataDimCount,
        indicesDims, indicesDimCount, dim);

    cudaDeviceSynchronize();
}

// ==============================================
// 精度转换实现
// ==============================================

__global__ void convert_fp32_to_fp16_kernel(float* src, __half* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        dest[idx] = __float2half(src[idx]);
    }
}

void runConvertFP32ToFP16(float* src, __half* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp32_to_fp16_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_fp32_to_bf16_kernel(float* src, __nv_bfloat16* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        dest[idx] = __float2bfloat16(src[idx]);
    }
}

void runConvertFP32ToBF16(float* src, __nv_bfloat16* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp32_to_bf16_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_fp32_to_fp8e4m3_kernel(float* src, __nv_fp8_e4m3* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        //dest[idx] = __float2nv_fp8_e4m3(src[idx]);
        dest[idx] = (__nv_fp8_e4m3)__nv_cvt_float_to_fp8(src[idx], __NV_SATFINITE, __NV_E4M3);
    }
}

void runConvertFP32ToFP8E4M3(float* src, __nv_fp8_e4m3* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp32_to_fp8e4m3_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_fp32_to_fp8e5m2_kernel(float* src, __nv_fp8_e5m2* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        //dest[idx] = __float2nv_fp8_e5m2(src[idx]);
        dest[idx] = (__nv_fp8_e5m2)__nv_cvt_float_to_fp8(src[idx], __NV_SATFINITE, __NV_E5M2);
    }
}

void runConvertFP32ToFP8E5M2(float* src, __nv_fp8_e5m2* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp32_to_fp8e5m2_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_fp16_to_fp32_kernel(__half* src, float* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        dest[idx] = __half2float(src[idx]);
    }
}

void runConvertFP16ToFP32(__half* src, float* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp16_to_fp32_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_bf16_to_fp32_kernel(__nv_bfloat16* src, float* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        dest[idx] = __bfloat162float(src[idx]);
    }
}

void runConvertBF16ToFP32(__nv_bfloat16* src, float* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_bf16_to_fp32_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_fp8e4m3_to_fp32_kernel(__nv_fp8_e4m3* src, float* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        //dest[idx] = __nv_fp8e4m32float(src[idx]);
#if defined(__CUDA_FP8_TYPES_EXIST__)
        dest[idx] = float(__nv_fp8_e4m3(src[idx]));  // 显式构造后转换
#else
        return 0.0f;
#endif
    }
}

void runConvertFP8E4M3ToFP32(__nv_fp8_e4m3* src, float* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp8e4m3_to_fp32_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

__global__ void convert_fp8e5m2_to_fp32_kernel(__nv_fp8_e5m2* src, float* dest, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
       // dest[idx] = __nv_fp8e5m22float(src[idx]);        
#if defined(__CUDA_FP8_TYPES_EXIST__)
        dest[idx] = float(__nv_fp8_e5m2(src[idx]));  // 显式构造后转换
#else
        return 0.0f;
#endif
    }
}

void runConvertFP8E5M2ToFP32(__nv_fp8_e5m2* src, float* dest, int totalElements) {
    int gridSize = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_fp8e5m2_to_fp32_kernel << <gridSize, BLOCK_SIZE >> > (src, dest, totalElements);
    cudaDeviceSynchronize();
}

// ------------------------
// Memory Management
// ------------------------
void runZeroInitializeFP32(float* data, int count, cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(data, 0, count * sizeof(float), stream);
    assert(err == cudaSuccess);
}

void runZeroInitializeFP16(__half* data, int count, cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(data, 0, count * sizeof(__half), stream);
    assert(err == cudaSuccess);
}

void runZeroInitializeBF16(bfloat16* data, int count, cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(data, 0, count * sizeof(bfloat16), stream);
    assert(err == cudaSuccess);
}

//#ifdef _WIN32
//static class Cleanup {
//public:
//    ~Cleanup() {
//        CublasHandle::destroy();
//    }
//} 
//#else
//__attribute__((destructor))
//static void cleanup() {
//    CublasHandle::destroy();
//}
//#endif