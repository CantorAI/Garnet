#ifndef _GARNET_CUDA_H_
#define _GARNET_CUDA_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h> // Assumes definitions for fp8_e4m3 and fp8_e5m2 types

using bfloat16 = __nv_bfloat16;
typedef __nv_fp8_e4m3 fp8_e4m3;
typedef __nv_fp8_e5m2 fp8_e5m2;

#ifdef __cplusplus
extern "C" {
#endif

    // ------------------------
    // Multiply Kernels
    // ------------------------
    void runGemmFP32(float* A, float* B, float* C, int m, int k, int n);
    void runGemmFP16(__half* A, __half* B, __half* C, int m, int k, int n);
    void runGemmBF16(bfloat16* A, bfloat16* B, bfloat16* C, int m, int k, int n);
    void runGemmFP8E4M3(fp8_e4m3* A, fp8_e4m3* B, fp8_e4m3* C, int m, int k, int n);
    void runGemmFP8E5M2(fp8_e5m2* A, fp8_e5m2* B, fp8_e5m2* C, int m, int k, int n);

    void runSingleElementTensorMultiplyFP32(float* multi, float* single, float* result, int count);
    void runSingleElementTensorMultiplyFP16(__half* multi, __half* single, __half* result, int count);
    void runSingleElementTensorMultiplyBF16(bfloat16* multi, bfloat16* single, bfloat16* result, int count);
    void runSingleElementTensorMultiplyFP8E4M3(fp8_e4m3* multi, fp8_e4m3* single, fp8_e4m3* result, int count);
    void runSingleElementTensorMultiplyFP8E5M2(fp8_e5m2* multi, fp8_e5m2* single, fp8_e5m2* result, int count);

    void runScalarMultiplyFP32(float* input, float* result, float scalar, int count);
    void runScalarMultiplyFP16(__half* input, __half* result, float scalar, int count);
    void runScalarMultiplyBF16(bfloat16* input, bfloat16* result, float scalar, int count);
    void runScalarMultiplyFP8E4M3(fp8_e4m3* input, fp8_e4m3* result, float scalar, int count);
    void runScalarMultiplyFP8E5M2(fp8_e5m2* input, fp8_e5m2* result, float scalar, int count);

    // ------------------------
    // Add Kernels
    // ------------------------
    void runAddFP64(double* A, double* B, double* C, int count);
    void runAddFP32(float* A, float* B, float* C, int count);
    void runAddFP16(__half* A, __half* B, __half* C, int count);
    void runAddBF16(bfloat16* A, bfloat16* B, bfloat16* C, int count);
    void runAddFP8E4M3(fp8_e4m3* A, fp8_e4m3* B, fp8_e4m3* C, int count);
    void runAddFP8E5M2(fp8_e5m2* A, fp8_e5m2* B, fp8_e5m2* C, int count);

    void runSingleElementTensorAddFP64(double* tensor, double single, double* result, int count);
    void runSingleElementTensorAddFP32(float* tensor, float single, float* result, int count);
    void runSingleElementTensorAddFP16(__half* tensor, __half single, __half* result, int count);
    void runSingleElementTensorAddBF16(bfloat16* tensor, bfloat16 single, bfloat16* result, int count);
    void runSingleElementTensorAddFP8E4M3(fp8_e4m3* tensor, fp8_e4m3 single, fp8_e4m3* result, int count);
    void runSingleElementTensorAddFP8E5M2(fp8_e5m2* tensor, fp8_e5m2 single, fp8_e5m2* result, int count);

    void runScalarAddFP32(float* input, float scalar, float* result, int count);

    // ------------------------
    // Minus Kernels
    // ------------------------
    void runMinusFP64(double* A, double* B, double* C, int count);
    void runMinusFP32(float* A, float* B, float* C, int count);
    void runMinusFP16(__half* A, __half* B, __half* C, int count);
    void runMinusBF16(bfloat16* A, bfloat16* B, bfloat16* C, int count);
    void runMinusFP8E4M3(fp8_e4m3* A, fp8_e4m3* B, fp8_e4m3* C, int count);
    void runMinusFP8E5M2(fp8_e5m2* A, fp8_e5m2* B, fp8_e5m2* C, int count);

    void runSingleElementTensorMinusFP64(double* tensor, double single, double* result, int count);
    void runSingleElementTensorMinusFP32(float* tensor, float single, float* result, int count);
    void runSingleElementTensorMinusFP16(__half* tensor, __half single, __half* result, int count);
    void runSingleElementTensorMinusBF16(bfloat16* tensor, bfloat16 single, bfloat16* result, int count);
    void runSingleElementTensorMinusFP8E4M3(fp8_e4m3* tensor, fp8_e4m3 single, fp8_e4m3* result, int count);
    void runSingleElementTensorMinusFP8E5M2(fp8_e5m2* tensor, fp8_e5m2 single, fp8_e5m2* result, int count);

    void runElementwiseTensorMultiplyFP8E4M3(fp8_e4m3* tensor, fp8_e4m3 single, fp8_e4m3* result, int count);
    void runElementwiseTensorMultiplyFP8E5M2(fp8_e5m2* tensor, fp8_e5m2 single, fp8_e5m2* result, int count);

    void runScalarMinusFP32(float* input, float scalar, float* result, int count);

    // ------------------------
    // Matmul Kernels
    // ------------------------
    void runMatmulFP32(float* A, float* B, float* C, int m, int n, int k);
    void runMatmulFP16(__half* A, __half* B, __half* C, int m, int n, int k);
    void runMatmulBF16(bfloat16* A, bfloat16* B, bfloat16* C, int m, int n, int k);
    void runMatmulFP8E4M3(fp8_e4m3* A, fp8_e4m3* B, fp8_e4m3* C, int m, int n, int k);
    void runMatmulFP8E5M2(fp8_e5m2* A, fp8_e5m2* B, fp8_e5m2* C, int m, int n, int k);

    // ------------------------
    // Permute Kernels
    // ------------------------
    void runPermuteFP32(float* input, float* output, int* permOrder, int* dimSizes, int dimCount);
    void runPermuteFP16(__half* input, __half* output, int* permOrder, int* dimSizes, int dimCount);
    void runPermuteBF16(bfloat16* input, bfloat16* output, int* permOrder, int* dimSizes, int dimCount);
    void runPermuteFP8E4M3(fp8_e4m3* input, fp8_e4m3* output, int* permOrder, int* dimSizes, int dimCount);
    void runPermuteFP8E5M2(fp8_e5m2* input, fp8_e5m2* output, int* permOrder, int* dimSizes, int dimCount);

    // ------------------------
    // Gather Kernels
    // ------------------------
    void runGatherFP32(float* data, float* indices, float* output,
        int* dataDims, int dataDimCount,
        int* indicesDims, int indicesDimCount, int dim);
    void runGatherFP16(__half* data, __half* indices, __half* output,
        int* dataDims, int dataDimCount,
        int* indicesDims, int indicesDimCount, int dim);
    void runGatherBF16(bfloat16* data, bfloat16* indices, bfloat16* output,
        int* dataDims, int dataDimCount,
        int* indicesDims, int indicesDimCount, int dim);
    void runGatherFP8E4M3(fp8_e4m3* data, fp8_e4m3* indices, fp8_e4m3* output,
        int* dataDims, int dataDimCount,
        int* indicesDims, int indicesDimCount, int dim);
    void runGatherFP8E5M2(fp8_e5m2* data, fp8_e5m2* indices, fp8_e5m2* output,
        int* dataDims, int dataDimCount,
        int* indicesDims, int indicesDimCount, int dim);

    // ------------------------
    // Convert Kernels
    // ------------------------
    void runConvertFP32ToFP16(float* src, __half* dest, int totalElements);
    void runConvertFP32ToBF16(float* src, bfloat16* dest, int totalElements);
    void runConvertFP32ToFP8E4M3(float* src, fp8_e4m3* dest, int totalElements);
    void runConvertFP32ToFP8E5M2(float* src, fp8_e5m2* dest, int totalElements);

    void runConvertFP16ToFP32(__half* src, float* dest, int totalElements);
    void runConvertBF16ToFP32(bfloat16* src, float* dest, int totalElements);
    void runConvertFP8E4M3ToFP32(fp8_e4m3* src, float* dest, int totalElements);
    void runConvertFP8E5M2ToFP32(fp8_e5m2* src, float* dest, int totalElements);

    // ------------------------
    // Memory Management
    // ------------------------
    void runZeroInitializeFP32(float* data, int count, cudaStream_t stream = 0);
    void runZeroInitializeFP16(__half* data, int count, cudaStream_t stream = 0);
    void runZeroInitializeBF16(bfloat16* data, int count, cudaStream_t stream = 0);

    // Exact QKV self-attention for Qwen3-VL vision attention.
    // Input qkv is [tokens, 3 * heads * head_dim], output is [tokens, heads * head_dim].
    // This path avoids materializing [heads, tokens, tokens] scores for original-resolution images.
    cudaError_t runVisionAttentionFP32(
        const float* qkv,
        float* output,
        int tokens,
        int heads,
        int headDim,
        cudaStream_t stream = 0);

    // Row-major linear layer: output[M, N] = input[M, K] * weight[N, K]^T + bias[N].
    cudaError_t runLinearBiasTransposeFP32(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int rows,
        int inFeatures,
        int outFeatures,
        cudaStream_t stream = 0);

    // Row-major linear layer: output[M, N] = input[M, K] * weight[N, K]^T.
    cudaError_t runLinearTransposeFP32(
        const float* input,
        const float* weight,
        float* output,
        int rows,
        int inFeatures,
        int outFeatures,
        cudaStream_t stream = 0);

    cudaError_t runSiluMulFP32(
        const float* gate,
        const float* up,
        float* output,
        int count,
        cudaStream_t stream = 0);

    //void cleanup();

#ifdef __cplusplus
}
#endif

#endif // _GARNET_CUDA_H_
