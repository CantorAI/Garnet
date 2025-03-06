// ptxGemm_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2

// Set to 0 to disable FP8 kernels
#define _FP8_SUPPORT_ 1

extern "C" {

    //------------------------------------------------------------------------------
    // FP16 Kernel
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
            unsigned int b1 = 0; // Only use b0 and b1 for m16n8k16 instruction
            // Provide separate accumulator input registers (all zero).
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float c0, c1, c2, c3; // Need to keep all as they're outputs of the MMA instruction

            // Ada Lovelace compatible tensor core instruction
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, "
                "{%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1),
                "f"(acc0), "f"(acc1), "f"(acc2), "f"(acc3)
                );

            // Accumulate the result
            accum += c0 + c1 + c2 + c3; // Use all output registers to avoid warnings
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
            unsigned int b1 = 0; // Only use b0 and b1 for m16n8k16 instruction
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float c0, c1, c2, c3; // Need to keep all as they're outputs of the MMA instruction

            // Ada Lovelace compatible tensor core instruction for BF16
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, "
                "{%4,%5,%6,%7}, "
                "{%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                "r"(b0), "r"(b1),
                "f"(acc0), "f"(acc1), "f"(acc2), "f"(acc3)
                );

            // Accumulate the result
            accum += c0 + c1 + c2 + c3; // Use all output registers to avoid warnings
            __syncthreads();
        }
        C[row * N + col] = accum;
    }

#if _FP8_SUPPORT_
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

            // For Ada Lovelace architecture, convert fp8 to fp16 and use fp16 tensor cores
            if (tid == 0) {  // Only one thread computes the result for simplicity
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += (float)tileA[0][k] * (float)tileB[k][0];
                }
                accum += sum;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {  // Only the first thread writes the result
            C[row * N + col] = accum;
        }
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

            // For Ada Lovelace architecture, convert fp8 to fp16 and use fp16 tensor cores
            if (tid == 0) {  // Only one thread computes the result for simplicity
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += (float)tileA[0][k] * (float)tileB[k][0];
                }
                accum += sum;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {  // Only the first thread writes the result
            C[row * N + col] = accum;
        }
    }
#endif

    //------------------------------------------------------------------------------
    // FP32 Kernel
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

            // Simple scalar implementation for FP32
            if (tid == 0) {
                float sum = 0.0f;
                for (int k = 0; k < 16; k++) {
                    sum += tileA[0][k] * tileB[k][0];
                }
                accum += sum;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            C[row * N + col] = accum;
        }
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

#if _FP8_SUPPORT_
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
#endif

    void runGemmFP32(const float* d_A, const float* d_B, float* d_C,
        int M, int N, int K) {
        dim3 gridDim(N / 16, M / 16);
        dim3 blockDim(32, 1, 1);
        gemm_kernel_fp32 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

} // extern "C"