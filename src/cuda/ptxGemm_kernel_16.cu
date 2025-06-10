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
     //------------------------------------------------------------------------------
    // 修正后的FP16内核
    __global__ void gemm_kernel_fp16(const __half* A, const __half* B, __half* C,
        int M, int K, int N) {
        // 二维线程块: 每个线程块处理一个16x16输出块
        int row_in_block = threadIdx.y; // 线程在块内的行ID (0-15)
        int col_in_block = threadIdx.x; // 线程在块内的列ID (0-15)

        int block_row = blockIdx.y;     // 输出块的行索引
        int block_col = blockIdx.x;     // 输出块的列索引

        // 计算输出矩阵C中的全局坐标
        int row = block_row * 16 + row_in_block;
        int col = block_col * 16 + col_in_block;

        // 共享内存声明 (每个块16x16)
        __shared__ __half tileA[16][16];
        __shared__ __half tileB[16][16];

        float accum = 0.0f;

        // 沿K维度分块处理
        for (int t = 0; t < K; t += 16) {
            // 协作加载A的tile (行优先)
            int load_row = row_in_block;
            int load_col = t + col_in_block;
            if (row < M && load_col < K) {
                tileA[row_in_block][col_in_block] = A[row * K + load_col];
            }
            else {
                tileA[row_in_block][col_in_block] = __float2half(0.0f);
            }

            // 协作加载B的tile (列优先)
            load_row = t + row_in_block;
            load_col = col;
            if (load_row < K && col < N) {
                tileB[row_in_block][col_in_block] = B[load_row * N + col];
            }
            else {
                tileB[row_in_block][col_in_block] = __float2half(0.0f);
            }
            __syncthreads();

            // 计算当前tile对accum的贡献
            for (int k = 0; k < 16; k++) {
                accum += __half2float(tileA[row_in_block][k]) *
                    __half2float(tileB[k][col_in_block]);
            }
            __syncthreads();
        }

        // 将结果写入全局内存
        if (row < M && col < N) {
            C[row * N + col] = __float2half(accum);
        }
    }

    //------------------------------------------------------------------------------
    // 修正后的BF16内核
    __global__ void gemm_kernel_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
        int M, int K, int N) {
        // 二维线程块: 每个线程块处理一个16x16输出块
        int row_in_block = threadIdx.y;
        int col_in_block = threadIdx.x;

        int block_row = blockIdx.y;
        int block_col = blockIdx.x;

        int row = block_row * 16 + row_in_block;
        int col = block_col * 16 + col_in_block;

        __shared__ __nv_bfloat16 tileA[16][16];
        __shared__ __nv_bfloat16 tileB[16][16];

        float accum = 0.0f;
        for (int t = 0; t < K; t += 16) {
            // 协作加载A的tile
            int load_row = row_in_block;
            int load_col = t + col_in_block;
            if (row < M && load_col < K) {
                tileA[row_in_block][col_in_block] = A[row * K + load_col];
            }
            else {
                tileA[row_in_block][col_in_block] = __float2bfloat16(0.0f);
            }

            // 协作加载B的tile
            load_row = t + row_in_block;
            load_col = col;
            if (load_row < K && col < N) {
                tileB[row_in_block][col_in_block] = B[load_row * N + col];
            }
            else {
                tileB[row_in_block][col_in_block] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            // 计算累加值
            for (int k = 0; k < 16; k++) {
                accum += __bfloat162float(tileA[row_in_block][k]) *
                    __bfloat162float(tileB[k][col_in_block]);
            }
            __syncthreads();
        }

        // 写入结果
        if (row < M && col < N) {
            C[row * N + col] = __float2bfloat16(accum);
        }
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

    __global__ void gemm_kernel_fp32(const float* A, const float* B, float* C,
        int M, int N, int K) {
        // 二维线程块: 每个线程块处理一个16x16输出块
        int row_in_block = threadIdx.y; // 线程在块内的行ID (0-15)
        int col_in_block = threadIdx.x; // 线程在块内的列ID (0-15)

        int block_row = blockIdx.y;     // 输出块的行索引
        int block_col = blockIdx.x;     // 输出块的列索引

        // 计算输出矩阵C中的全局坐标
        int row = block_row * 16 + row_in_block;
        int col = block_col * 16 + col_in_block;

        // 共享内存声明 (每个块16x16)
        __shared__ float tileA[16][16];
        __shared__ float tileB[16][16];

        float accum = 0.0f;

        // 沿K维度分块处理
        for (int t = 0; t < K; t += 16) {
            // 协作加载A的tile (行优先)
            int load_row = row_in_block;
            int load_col = t + col_in_block;
            if (row < M && load_col < K) {
                tileA[row_in_block][col_in_block] = A[row * K + load_col];
            }
            else {
                tileA[row_in_block][col_in_block] = 0.0f;
            }

            // 协作加载B的tile (列优先)
            load_row = t + row_in_block;
            load_col = col;
            if (load_row < K && col < N) {
                tileB[row_in_block][col_in_block] = B[load_row * N + col];
            }
            else {
                tileB[row_in_block][col_in_block] = 0.0f;
            }
            __syncthreads();

            // 计算当前tile对accum的贡献
                for (int k = 0; k < 16; k++) {
                accum += tileA[row_in_block][k] * tileB[k][col_in_block];
            }
            __syncthreads();
        }

        // 将结果写入全局内存
        if (row < M && col < N) {
            C[row * N + col] = accum;
        }
    }

    //------------------------------------------------------------------------------
    // Wrapper functions: each launches the corresponding kernel over a grid covering the full matrix.
    // Assumes that M, N, and K are multiples of 16.
    void runGemmFP16(__half* d_A, __half* d_B, __half* d_C,
        int m, int k, int n) {
        // 二维线程块: 16x16 = 256 threads/block
        dim3 blockDim(16, 16);

        // 网格布局: 每个块处理16x16输出
        dim3 gridDim((n + 15) / 16, (m + 15) / 16);

        gemm_kernel_fp16 << <gridDim, blockDim >> > (d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();
    }

    void runGemmBF16(__nv_bfloat16* d_A, __nv_bfloat16* d_B, __nv_bfloat16* d_C,
        int m, int k, int n) {
        // 二维线程块: 16x16 = 256 threads/block
        dim3 blockDim(16, 16);

        // 网格布局: 每个块处理16x16输出
        dim3 gridDim((n + 15) / 16, (m + 15) / 16);

        gemm_kernel_bf16 << <gridDim, blockDim >> > (d_A, d_B, d_C, m, k, n);
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

    //void runGemmFP32(const float* d_A, const float* d_B, float* d_C,
    //    int M, int N, int K) {
    //    dim3 gridDim(N / 16, M / 16);
    //    dim3 blockDim(32, 1, 1);
    //    gemm_kernel_fp32 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
    //    cudaDeviceSynchronize();
    //}

    void runGemmFP32(const float* d_A, const float* d_B, float* d_C,
        int M, int N, int K) {
        // 二维线程块: 16x16 = 256 threads/block
        dim3 blockDim(16, 16); // 修正线程块布局

        // 网格布局: 每个块处理16x16输出
        dim3 gridDim((N + 15) / 16, (M + 15) / 16); // 处理非16倍数尺寸

        gemm_kernel_fp32 << <gridDim, blockDim >> > (d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }

} // extern "C"