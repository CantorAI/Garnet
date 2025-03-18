// cuda_templates/cuda_function_declarations.h
#ifndef CUDA_FUNCTION_DECLARATIONS_H
#define CUDA_FUNCTION_DECLARATIONS_H

// This file contains all CUDA function declarations used by GarnetTensor
// Edit this file to add or modify CUDA function declarations

extern "C" {
    // Matrix multiplication operations
    void runGemmFP32(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
    void runGemmFP16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K);
    void runGemmBF16(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C, int M, int N, int K);
    void runGemmFP8E4M3(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B, float* d_C, int M, int N, int K);
    void runGemmFP8E5M2(const __nv_fp8_e5m2* d_A, const __nv_fp8_e5m2* d_B, float* d_C, int M, int N, int K);

    // Gather operations
    void runGatherKernelFloat(const float* input, const int* indices, float* output, int M, int N, int numIndices);
    void runGatherKernelFP16(const __half* input, const int* indices, __half* output, int M, int N, int numIndices);
    void runGatherKernelBF16(const __nv_bfloat16* input, const int* indices, __nv_bfloat16* output, int M, int N, int numIndices);
    void runGatherKernelFP8E4M3(const __nv_fp8_e4m3* input, const int* indices, __nv_fp8_e4m3* output, int M, int N, int numIndices);
    void runGatherKernelFP8E5M2(const __nv_fp8_e5m2* input, const int* indices, __nv_fp8_e5m2* output, int M, int N, int numIndices);

    // Scalar operations
    void runScalarMultiplyFP32(const float* input, float* output, float scalar, long long length);
    void runScalarMultiplyFP16(const __half* input, __half* output, float scalar, long long length);
    void runScalarMultiplyBF16(const __nv_bfloat16* input, __nv_bfloat16* output, float scalar, long long length);
    void runScalarMultiplyFP8E4M3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, float scalar, long long length);
    void runScalarMultiplyFP8E5M2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, float scalar, long long length);

    // Element-wise operations
    void runElementwiseAddFP32(const float* input1, const float* input2, float* output, long long length);
    void runElementwiseAddFP16(const __half* input1, const __half* input2, __half* output, long long length);
    void runElementwiseAddBF16(const __nv_bfloat16* input1, const __nv_bfloat16* input2, __nv_bfloat16* output, long long length);
    void runElementwiseAddFP8E4M3(const __nv_fp8_e4m3* input1, const __nv_fp8_e4m3* input2, __nv_fp8_e4m3* output, long long length);
    void runElementwiseAddFP8E5M2(const __nv_fp8_e5m2* input1, const __nv_fp8_e5m2* input2, __nv_fp8_e5m2* output, long long length);

    void runElementwiseSubtractFP32(const float* input1, const float* input2, float* output, long long length);
    void runElementwiseSubtractFP16(const __half* input1, const __half* input2, __half* output, long long length);
    void runElementwiseSubtractBF16(const __nv_bfloat16* input1, const __nv_bfloat16* input2, __nv_bfloat16* output, long long length);
    void runElementwiseSubtractFP8E4M3(const __nv_fp8_e4m3* input1, const __nv_fp8_e4m3* input2, __nv_fp8_e4m3* output, long long length);
    void runElementwiseSubtractFP8E5M2(const __nv_fp8_e5m2* input1, const __nv_fp8_e5m2* input2, __nv_fp8_e5m2* output, long long length);

    // Permute operations
    void runPermuteFP32(const float* input, float* output, const int* dims, const int* perm, int rank);
    void runPermuteFP16(const __half* input, __half* output, const int* dims, const int* perm, int rank);
    void runPermuteBF16(const __nv_bfloat16* input, __nv_bfloat16* output, const int* dims, const int* perm, int rank);
    void runPermuteFP8E4M3(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, const int* dims, const int* perm, int rank);
    void runPermuteFP8E5M2(const __nv_fp8_e5m2* input, __nv_fp8_e5m2* output, const int* dims, const int* perm, int rank);

    // Type conversion operations
    void runConvertFP32ToFP16(const float* input, __half* output, long long length);
    void runConvertFP32ToBF16(const float* input, __nv_bfloat16* output, long long length);
    void runConvertFP32ToFP8E4M3(const float* input, __nv_fp8_e4m3* output, long long length);
    void runConvertFP32ToFP8E5M2(const float* input, __nv_fp8_e5m2* output, long long length);
    void runConvertFP16ToFP32(const __half* input, float* output, long long length);
    void runConvertBF16ToFP32(const __nv_bfloat16* input, float* output, long long length);
    void runConvertFP8E4M3ToFP32(const __nv_fp8_e4m3* input, float* output, long long length);
    void runConvertFP8E5M2ToFP32(const __nv_fp8_e5m2* input, float* output, long long length);
}

#endif // CUDA_FUNCTION_DECLARATIONS_H