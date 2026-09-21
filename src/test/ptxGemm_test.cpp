// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // For __half and __float2half
#include <cuda_bf16.h>    // For __nv_bfloat16
#include <cuda_fp8.h>     // For __nv_fp8_e4m3 and __nv_fp8_e5m2

// Forward declarations for the CUDA functions from ptxGemm_kernel.cu
extern "C" {
    void runGemmFP16(const __half* d_A, const __half* d_B, float* d_C, int M, int N, int K);
    void runGemmBF16(const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C, int M, int N, int K);
    void runGemmFP8E4M3(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B, float* d_C, int M, int N, int K);
    void runGemmFP8E5M2(const __nv_fp8_e5m2* d_A, const __nv_fp8_e5m2* d_B, float* d_C, int M, int N, int K);
    void runGemmFP32(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
}

// Helper function to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Reference CPU implementation to verify results
void cpuGemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
    int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Template for testing all GEMM precision types
template<typename T_input>
void testGemm(const std::string& testName,
    void (*gemmFunc)(const T_input*, const T_input*, float*, int, int, int),
    int M, int N, int K) {
    std::cout << "Testing " << testName << " GEMM..." << std::endl;

    // Generate random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_float(M * K);
    std::vector<float> B_float(K * N);
    std::vector<float> C_cpu(M * N, 0);

    for (int i = 0; i < M * K; i++) A_float[i] = dist(gen);
    for (int i = 0; i < K * N; i++) B_float[i] = dist(gen);

    // Create input matrices in desired precision
    std::vector<T_input> A(M * K);
    std::vector<T_input> B(K * N);

    // Convert from float to target precision
    for (int i = 0; i < M * K; i++) A[i] = static_cast<T_input>(A_float[i]);
    for (int i = 0; i < K * N; i++) B[i] = static_cast<T_input>(B_float[i]);

    // CPU reference calculation
    cpuGemm(A_float, B_float, C_cpu, M, N, K);

    // Allocate GPU memory
    T_input* d_A, * d_B;
    float* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(T_input)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(T_input)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(T_input), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(T_input), cudaMemcpyHostToDevice));

    // Run GPU GEMM
    auto start = std::chrono::high_resolution_clock::now();
    gemmFunc(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Copy results back
    std::vector<float> C_gpu(M * N);
    CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results (only top-left element since our kernels only compute that)
    float max_diff = std::abs(C_gpu[0] - C_cpu[0]);
    float relative_diff = (C_cpu[0] != 0) ? max_diff / std::abs(C_cpu[0]) : max_diff;

    std::cout << "  Top-left element: CPU=" << C_cpu[0] << ", GPU=" << C_gpu[0]
        << ", Relative diff=" << relative_diff << std::endl;

    // Report timing
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "  Execution time: " << duration << " us" << std::endl;

    // Free GPU memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// Specialization for FP16
template<>
void testGemm<__half>(const std::string& testName,
    void (*gemmFunc)(const __half*, const __half*, float*, int, int, int),
    int M, int N, int K) {
    std::cout << "Testing " << testName << " GEMM..." << std::endl;

    std::vector<float> A_float(M * K);
    std::vector<float> B_float(K * N);
    std::vector<float> C_cpu(M * N, 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; i++) A_float[i] = dist(gen);
    for (int i = 0; i < K * N; i++) B_float[i] = dist(gen);

    // CPU reference
    cpuGemm(A_float, B_float, C_cpu, M, N, K);

    // Convert to FP16
    std::vector<__half> A(M * K);
    std::vector<__half> B(K * N);
    for (int i = 0; i < M * K; i++) A[i] = __float2half(A_float[i]);
    for (int i = 0; i < K * N; i++) B[i] = __float2half(B_float[i]);

    // GPU allocation and execution
    __half* d_A, * d_B;
    float* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();
    gemmFunc(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<float> C_gpu(M * N);
    CHECK_CUDA(cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = std::abs(C_gpu[0] - C_cpu[0]);
    float relative_diff = (C_cpu[0] != 0) ? max_diff / std::abs(C_cpu[0]) : max_diff;

    std::cout << "  Top-left element: CPU=" << C_cpu[0] << ", GPU=" << C_gpu[0]
        << ", Relative diff=" << relative_diff << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "  Execution time: " << duration << " us" << std::endl;

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

int ptxGemm_test() {
    // Matrix dimensions (must be multiples of 16)
    const int M = 64;
    const int N = 48;
    const int K = 1;

    // Print device information
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;

    // Test all precision formats
    //testGemm<float>("FP32", runGemmFP32, M, N, K);
    testGemm<__half>("FP16", runGemmFP16, M, N, K);
    testGemm<__nv_bfloat16>("BF16", runGemmBF16, M, N, K);
    testGemm<__nv_fp8_e4m3>("FP8_E4M3", runGemmFP8E4M3, M, N, K);
    testGemm<__nv_fp8_e5m2>("FP8_E5M2", runGemmFP8E5M2, M, N, K);

    std::cout << "All tests completed" << std::endl;
    return 0;
}
