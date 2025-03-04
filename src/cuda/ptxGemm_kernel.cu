// ptxGemm_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2

// IMPORTANT: Compile with -arch=sm_80 or higher

extern "C" {

    //------------------------------------------------------------------------------
    // FP16 Kernel
    // Assumes matrices A (MxK) and B (KxN) are stored in row–major order and M, N, K are multiples of 16.
    // Each block computes one 16×16 tile of C. For simplicity, this kernel computes only a dummy value
    // (the top–left element of the tile) using one inline PTX MMA call.
    __global__ void gemm_kernel_fp16(const __half* A, const __half* B, float* C,
        int M, int N, int K) {
        int block_row = blockIdx.y; // tile row index
        int block_col = blockIdx.x; // tile column index
        int row = block_row * 16;
        int col = block_col * 16;

        // Allocate shared memory for one tile of A and one tile of B.
        __shared__ __half tileA[16][16];
        __shared__ __half tileB[16][16];

        float accum = 0.0f;

        // Loop over K dimension tiles.
        for (int t = 0; t < K; t += 16) {
            int tid = threadIdx.x;
            // Each block loads a 16x16 sub-tile from global memory.
            for (int i = tid; i < 256; i += 32) {
                int r = i / 16;
                int c = i % 16;
                tileA[r][c] = A[(row + r) * K + (t + c)];
                tileB[r][c] = B[(t + r) * N + (col + c)];
            }
            __syncthreads();

            // For demonstration, pack the first two __half values from each tile into a 32–bit word.
            unsigned int a0 = *((unsigned int*)&tileA[0][0]); // packs tileA[0][0] and tileA[0][1]
            unsigned int b0 = *((unsigned int*)&tileB[0][0]); // packs tileB[0][0] and tileB[0][1]

            // For this simple demo, we do not load the remaining registers.
            unsigned int a1 = 0, a2 = 0, a3 = 0;
            unsigned int b1 = 0, b2 = 0, b3 = 0;
            // Provide separate accumulator input registers (all zero).
            unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            unsigned int c0, c1, c2, c3;
            asm volatile(
                "mma.sync.aligned.m16n16k16.row.col.f32.f16 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, "
                "{%12, %13, %14, %15};\n"
                : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                "r"(acc0), "r"(acc1), "r"(acc2), "r"(acc3)
                );
            // Interpret c0 as a float value (dummy result) and accumulate.
            float frag_val = *((float*)&c0);
            accum += frag_val;
            __syncthreads();
        }
        // Write the dummy accumulated result to C at position (row, col)
        C[row * N + col] = accum;
    }

    //------------------------------------------------------------------------------
    // BF16 Kernel
    __global__ void gemm_kernel_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
        int M, int N, int K) {
        int block_row = blockIdx.y;
        int block_col = blockIdx.x;
        int row = block_row * 16;
        int col = block_col * 16;

        __shared__ __nv_bfloat16 tileA[16][16];
        __shared__ __nv_bfloat16 tileB[16][16];

        float accum = 0.0f;
        for (int t = 0; t < K; t += 16) {
            int tid = threadIdx.x;
            for (int i = tid; i < 256; i += 32) {
                int r = i / 16;
                int c = i % 16;
                tileA[r][c] = A[(row + r) * K + (t + c)];
                tileB[r][c] = B[(t + r) * N + (col + c)];
            }
            __syncthreads();

            unsigned int a0 = *((unsigned int*)&tileA[0][0]);
            unsigned int b0 = *((unsigned int*)&tileB[0][0]);
            unsigned int a1 = 0, a2 = 0, a3 = 0;
            unsigned int b1 = 0, b2 = 0, b3 = 0;
            unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            unsigned int c0, c1, c2, c3;
            asm volatile(
                "mma.sync.aligned.m16n16k16.row.col.f32.bf16 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, "
                "{%12, %13, %14, %15};\n"
                : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                "r"(acc0), "r"(acc1), "r"(acc2), "r"(acc3)
                );
            float frag_val = *((float*)&c0);
            accum += frag_val;
            __syncthreads();
        }
        C[row * N + col] = accum;
    }

    //------------------------------------------------------------------------------
    // FP8 E4M3 Kernel
    __global__ void gemm_kernel_fp8_e4m3(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* C,
        int M, int N, int K) {
        int block_row = blockIdx.y;
        int block_col = blockIdx.x;
        int row = block_row * 16;
        int col = block_col * 16;

        __shared__ __nv_fp8_e4m3 tileA[16][16];
        __shared__ __nv_fp8_e4m3 tileB[16][16];

        float accum = 0.0f;
        for (int t = 0; t < K; t += 16) {
            int tid = threadIdx.x;
            for (int i = tid; i < 256; i += 32) {
                int r = i / 16;
                int c = i % 16;
                tileA[r][c] = A[(row + r) * K + (t + c)];
                tileB[r][c] = B[(t + r) * N + (col + c)];
            }
            __syncthreads();

            unsigned int a0 = *((unsigned int*)&tileA[0][0]);
            unsigned int b0 = *((unsigned int*)&tileB[0][0]);
            unsigned int a1 = 0, a2 = 0, a3 = 0;
            unsigned int b1 = 0, b2 = 0, b3 = 0;
            unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            unsigned int c0, c1, c2, c3;
            asm volatile(
                "mma.sync.aligned.m16n16k16.row.col.f32.fp8e4m3 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, "
                "{%12, %13, %14, %15};\n"
                : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                "r"(acc0), "r"(acc1), "r"(acc2), "r"(acc3)
                );
            float frag_val = *((float*)&c0);
            accum += frag_val;
            __syncthreads();
        }
        C[row * N + col] = accum;
    }

    //------------------------------------------------------------------------------
    // FP8 E5M2 Kernel
    __global__ void gemm_kernel_fp8_e5m2(const __nv_fp8_e5m2* A, const __nv_fp8_e5m2* B, float* C,
        int M, int N, int K) {
        int block_row = blockIdx.y;
        int block_col = blockIdx.x;
        int row = block_row * 16;
        int col = block_col * 16;

        __shared__ __nv_fp8_e5m2 tileA[16][16];
        __shared__ __nv_fp8_e5m2 tileB[16][16];

        float accum = 0.0f;
        for (int t = 0; t < K; t += 16) {
            int tid = threadIdx.x;
            for (int i = tid; i < 256; i += 32) {
                int r = i / 16;
                int c = i % 16;
                tileA[r][c] = A[(row + r) * K + (t + c)];
                tileB[r][c] = B[(t + r) * N + (col + c)];
            }
            __syncthreads();

            unsigned int a0 = *((unsigned int*)&tileA[0][0]);
            unsigned int b0 = *((unsigned int*)&tileB[0][0]);
            unsigned int a1 = 0, a2 = 0, a3 = 0;
            unsigned int b1 = 0, b2 = 0, b3 = 0;
            unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            unsigned int c0, c1, c2, c3;
            asm volatile(
                "mma.sync.aligned.m16n16k16.row.col.f32.fp8e5m2 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, "
                "{%12, %13, %14, %15};\n"
                : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                "r"(acc0), "r"(acc1), "r"(acc2), "r"(acc3)
                );
            float frag_val = *((float*)&c0);
            accum += frag_val;
            __syncthreads();
        }
        C[row * N + col] = accum;
    }

    //------------------------------------------------------------------------------
    // FP32 Kernel
    // (Tensor–core FP32 instructions are available on supported hardware.)
    __global__ void gemm_kernel_fp32(const float* A, const float* B, float* C,
        int M, int N, int K) {
        int block_row = blockIdx.y;
        int block_col = blockIdx.x;
        int row = block_row * 16;
        int col = block_col * 16;

        __shared__ float tileA[16][16];
        __shared__ float tileB[16][16];

        float accum = 0.0f;
        for (int t = 0; t < K; t += 16) {
            int tid = threadIdx.x;
            for (int i = tid; i < 256; i += 32) {
                int r = i / 16;
                int c = i % 16;
                tileA[r][c] = A[(row + r) * K + (t + c)];
                tileB[r][c] = B[(t + r) * N + (col + c)];
            }
            __syncthreads();

            unsigned int a0 = *((unsigned int*)&tileA[0][0]);
            unsigned int b0 = *((unsigned int*)&tileB[0][0]);
            unsigned int a1 = 0, a2 = 0, a3 = 0;
            unsigned int b1 = 0, b2 = 0, b3 = 0;
            unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            unsigned int c0, c1, c2, c3;
            asm volatile(
                "mma.sync.aligned.m16n16k16.row.col.f32.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9, %10, %11}, "
                "{%12, %13, %14, %15};\n"
                : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1), "r"(b2), "r"(b3),
                "r"(acc0), "r"(acc1), "r"(acc2), "r"(acc3)
                );
            float frag_val = *((float*)&c0);
            accum += frag_val;
            __syncthreads();
        }
        C[row * N + col] = accum;
    }

    //------------------------------------------------------------------------------
    // Wrapper functions: each launches the corresponding kernel over a grid covering the full matrix.
    // Assumes that M, N, and K are multiples of 16.
    void runGemmFP16(const __half* d_A, const __half* d_B, float* d_C,
        int M, int N, int K) {
        dim3 gridDim(N / 16, M / 16);
        dim3 blockDim(32, 1, 1);  // one warp per block
        gemm_kernel_fp16 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    void runGemmBF16(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C,
        int M, int N, int K) {
        dim3 gridDim(N / 16, M / 16);
        dim3 blockDim(32, 1, 1);
        gemm_kernel_bf16 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    void runGemmFP8E4M3(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B, float* d_C,
        int M, int N, int K) {
        dim3 gridDim(N / 16, M / 16);
        dim3 blockDim(32, 1, 1);
        gemm_kernel_fp8_e4m3 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    void runGemmFP8E5M2(const __nv_fp8_e5m2* d_A, const __nv_fp8_e5m2* d_B, float* d_C,
        int M, int N, int K) {
        dim3 gridDim(N / 16, M / 16);
        dim3 blockDim(32, 1, 1);
        gemm_kernel_fp8_e5m2 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

    void runGemmFP32(const float* d_A, const float* d_B, float* d_C,
        int M, int N, int K) {
        dim3 gridDim(N / 16, M / 16);
        dim3 blockDim(32, 1, 1);
        gemm_kernel_fp32 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

} // extern "C"
