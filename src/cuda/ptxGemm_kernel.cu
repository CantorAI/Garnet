// matrixMulTensorCore_kernel.cu
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

using namespace nvcuda;

// Templated WMMA-based GEMM kernel that handles non-16 boundaries.
template <typename DataType>
__global__ void wmmaGemmKernelT(const DataType* A, const DataType* B, float* C,
    int M, int N, int K) {
    // Each block computes a 16x16 tile.
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = block_row * 16;
    int col = block_col * 16;

    // For FP8 types, WMMA fragments are not available.
    if constexpr (std::is_same_v<DataType, __nv_fp8_e4m3> ||
        std::is_same_v<DataType, __nv_fp8_e5m2>) {
        // Fallback: each thread computes one element of the 16x16 tile.
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            int r = i / 16;
            int c = i % 16;
            int globalRow = row + r;
            int globalCol = col + c;
            if (globalRow < M && globalCol < N) {
                float sum = 0.0f;
                // Loop over the K dimension.
                for (int k = 0; k < K; ++k) {
                    DataType a = A[globalRow * K + k];
                    DataType b = B[k * N + globalCol];
                    sum += static_cast<float>(a) * static_cast<float>(b);
                }
                C[globalRow * N + globalCol] = sum;
            }
        }
    }
    else {
        // WMMA path for supported types (e.g. __half, __nv_bfloat16)
        // Declare the accumulator fragment.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
        wmma::fill_fragment(cFrag, 0.0f);

        // Shared memory tiles for loading from global memory.
        __shared__ DataType tileA[16][16];
        __shared__ DataType tileB[16][16];

        // Loop over the K dimension in tiles of 16.
        for (int t = 0; t < K; t += 16) {
            // Load tile of A from global memory into shared memory with boundary checks.
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                int r = i / 16;
                int c = i % 16;
                int globalRow = row + r;
                int globalCol = t + c;
                if (globalRow < M && globalCol < K)
                    tileA[r][c] = A[globalRow * K + globalCol];
                else
                    tileA[r][c] = DataType(0);
            }
            // Load tile of B from global memory into shared memory with boundary checks.
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                int r = i / 16;
                int c = i % 16;
                int globalRow = t + r;
                int globalCol = col + c;
                if (globalRow < K && globalCol < N)
                    tileB[r][c] = B[globalRow * N + globalCol];
                else
                    tileB[r][c] = DataType(0);
            }
            __syncthreads();

            // Load WMMA fragments from the shared memory tiles.
            wmma::fragment<wmma::matrix_a, 16, 16, 16, DataType, wmma::row_major> aFrag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, DataType, wmma::col_major> bFrag;
            wmma::load_matrix_sync(aFrag, &tileA[0][0], 16);
            wmma::load_matrix_sync(bFrag, &tileB[0][0], 16);

            // Multiply and accumulate using WMMA.
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
            __syncthreads();
        }

        // Store the computed tile (from the WMMA accumulator) to shared memory.
        __shared__ float sharedC[16][16];
        wmma::store_matrix_sync(reinterpret_cast<float*>(sharedC), cFrag, 16, wmma::mem_row_major);
        __syncthreads();

        // Copy the tile from shared memory to global memory with boundary checks.
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            int r = i / 16;
            int c = i % 16;
            int globalRow = row + r;
            int globalCol = col + c;
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = sharedC[r][c];
        }
    }
}

// Helper function to launch the kernel.
template <typename DataType>
void launchWmmaGemmKernel(const DataType* A, const DataType* B, float* C,
    int M, int N, int K) {
    // Compute grid dimensions to cover the entire output.
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);
    // Use one warp (32 threads) per block.
    dim3 blockDim(32, 1, 1);
    wmmaGemmKernelT<DataType> << <gridDim, blockDim >> > (A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// Extern "C" interface for launching the kernels.
extern "C" {

    void runGemmFP16(const __half* A, const __half* B, float* C,
        int M, int N, int K) {
        launchWmmaGemmKernel(A, B, C, M, N, K);
    }

    void runGemmBF16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
        int M, int N, int K) {
        launchWmmaGemmKernel(A, B, C, M, N, K);
    }

    void runGemmFP8E4M3(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* C,
        int M, int N, int K) {
        launchWmmaGemmKernel(A, B, C, M, N, K);
    }

    void runGemmFP8E5M2(const __nv_fp8_e5m2* A, const __nv_fp8_e5m2* B, float* C,
        int M, int N, int K) {
        launchWmmaGemmKernel(A, B, C, M, N, K);
    }

} // extern "C"
