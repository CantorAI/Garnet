// gather_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2
#include <stdio.h>
#include "cuda_lib.h"

// Templated 2D gather kernel.
// Input matrix is in row-major order with shape (N, M):
//   N = vocabulary size (number of rows)
//   M = embedding dimension (number of columns)
// Each block in grid.x processes one output row (gather index),
// and grid.y covers the columns in tiles.
// Each thread computes: col = blockIdx.y * blockDim.x + threadIdx.x.
template <typename T>
__global__ void gather_kernel_2d(const T* input, const int* indices, T* output, int M) {
    // Each block.x corresponds to one token (one row to gather)
    int gatherIdx = blockIdx.x;
    // Compute the column index from both blockIdx.y and threadIdx.x.
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < M) {
        // We assume indices are valid.
        int row = indices[gatherIdx];
        output[gatherIdx * M + col] = input[row * M + col];
    }
}

// Extern "C" wrappers for various floating-point types.
extern "C" {

    void runGatherKernelFloat(const float* input, const int* indices, float* output, int M, int /*N*/, int numIndices) {
        int threadsPerBlock = 256;
        // grid.x = numIndices (one block per gather index)
        // grid.y = ceil(M / threadsPerBlock) to cover all columns.
        dim3 grid(numIndices, (M + threadsPerBlock - 1) / threadsPerBlock);
        dim3 block(threadsPerBlock);
        gather_kernel_2d<float> << <grid, block >> > (input, indices, output, M);
        cudaDeviceSynchronize();
    }

    void runGatherKernelFP16(const __half* input, const int* indices, __half* output, int M, int /*N*/, int numIndices) {
        int threadsPerBlock = 256;
        dim3 grid(numIndices, (M + threadsPerBlock - 1) / threadsPerBlock);
        dim3 block(threadsPerBlock);
        gather_kernel_2d<__half> << <grid, block >> > (input, indices, output, M);
        cudaDeviceSynchronize();
    }

    void runGatherKernelBF16(const __nv_bfloat16* input, const int* indices, __nv_bfloat16* output, int M, int /*N*/, int numIndices) {
        int threadsPerBlock = 256;
        dim3 grid(numIndices, (M + threadsPerBlock - 1) / threadsPerBlock);
        dim3 block(threadsPerBlock);
        gather_kernel_2d<__nv_bfloat16> << <grid, block >> > (input, indices, output, M);
        cudaDeviceSynchronize();
    }

    void runGatherKernelFP8E4M3(const __nv_fp8_e4m3* input, const int* indices, __nv_fp8_e4m3* output, int M, int /*N*/, int numIndices) {
        int threadsPerBlock = 256;
        dim3 grid(numIndices, (M + threadsPerBlock - 1) / threadsPerBlock);
        dim3 block(threadsPerBlock);
        gather_kernel_2d<__nv_fp8_e4m3> << <grid, block >> > (input, indices, output, M);
        cudaDeviceSynchronize();
    }

    void runGatherKernelFP8E5M2(const __nv_fp8_e5m2* input, const int* indices, __nv_fp8_e5m2* output, int M, int /*N*/, int numIndices) {
        int threadsPerBlock = 256;
        dim3 grid(numIndices, (M + threadsPerBlock - 1) / threadsPerBlock);
        dim3 block(threadsPerBlock);
        gather_kernel_2d<__nv_fp8_e5m2> << <grid, block >> > (input, indices, output, M);
        cudaDeviceSynchronize();
    }
}
