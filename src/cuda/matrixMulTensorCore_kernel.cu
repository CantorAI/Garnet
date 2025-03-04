// matrixMulTensorCore_kernel.cu
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

// Include headers for each type.
#include <cuda_fp16.h>    // For FP16
#include <cuda_bf16.h>    // For BF16
#include <cuda_fp8.h>     // For FP8 (assumed to be provided by your CUDA installation)

// Use the WMMA API.
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// -----------------------------------------------------------------
// Templated WMMA kernel for lower-precision types (FP16, BF16, FP8)
// Each block (one warp) computes one 16x16 tile of C.
template <typename DataType>
__global__ void wmmaGemmKernelT(const DataType* A, const DataType* B, float* C, int M, int N, int K) {
    // Compute tile indices.
    int tileRow = blockIdx.y;  // Tile row index in C.
    int tileCol = blockIdx.x;  // Tile column index in C.
    int row = tileRow * WMMA_M;
    int col = tileCol * WMMA_N;

    // Declare the accumulator fragment and initialize to zero.
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    fill_fragment(cFrag, 0.0f);

    // Loop over the K dimension tiles.
    for (int i = 0; i < K; i += WMMA_K) {
        // Pointers to the sub-tiles of A and B.
        const DataType* A_tile = A + row * K + i;
        const DataType* B_tile = B + i * N + col;

        // Declare fragments for the A and B sub-tiles.
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, DataType, row_major> aFrag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, DataType, col_major> bFrag;

        // Load the sub-tiles from global memory.
        load_matrix_sync(aFrag, A_tile, K);
        load_matrix_sync(bFrag, B_tile, N);

        // Multiply and accumulate: cFrag += aFrag * bFrag.
        mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // Write the resulting tile back to global memory.
    float* C_tile = C + row * N + col;
    store_matrix_sync(C_tile, cFrag, N, mem_row_major);
}

// -----------------------------------------------------------------
// Conventional tiled kernel for float (32-bit) matrix multiplication.
// This kernel uses shared memory tiling and does not use tensor cores.
__global__ void gemmKernelFloat(const float* A, const float* B, float* C, int M, int N, int K) {
    // Tile size (assumed 16x16).
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles of K dimension.
    for (int t = 0; t < K / TILE_SIZE; t++) {
        // Load A and B tiles into shared memory.
        As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Multiply the two tiles.
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

// -----------------------------------------------------------------
// Extern "C" wrappers for each type.

// FP16 version.
extern "C" void runWmmaGemmKernel_fp16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K) {
    dim3 gridDim(N / WMMA_N, M / WMMA_M);
    dim3 blockDim(32, 1, 1); // One warp per block.
    wmmaGemmKernelT<__half> << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// BF16 version.
extern "C" void runWmmaGemmKernel_bf16(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C, int M, int N, int K) {
    dim3 gridDim(N / WMMA_N, M / WMMA_M);
    dim3 blockDim(32, 1, 1);
    wmmaGemmKernelT<__nv_bfloat16> << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// FP8 version: E4M3.
extern "C" void runWmmaGemmKernel_fp8_e4m3(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B, float* d_C, int M, int N, int K) {
    dim3 gridDim(N / WMMA_N, M / WMMA_M);
    dim3 blockDim(32, 1, 1);
    wmmaGemmKernelT<__nv_fp8_e4m3> << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// FP8 version: E5M2.
extern "C" void runWmmaGemmKernel_fp8_e5m2(const __nv_fp8_e5m2* d_A, const __nv_fp8_e5m2* d_B, float* d_C, int M, int N, int K) {
    dim3 gridDim(N / WMMA_N, M / WMMA_M);
    dim3 blockDim(32, 1, 1);
    wmmaGemmKernelT<__nv_fp8_e5m2> << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}

// Float (32-bit) version using conventional tiled kernel.
extern "C" void runGemmKernel_fp32(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // We assume M, N, and K are multiples of 16.
    dim3 gridDim(N / 16, M / 16);
    dim3 blockDim(16, 16);
    gemmKernelFloat << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
}
