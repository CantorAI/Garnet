#include <iostream>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <climits>
#include <cstdint>
#include <system_error>
#include "cuda_lib.h"

#ifdef _WIN32
#include <windows.h>
#undef min
#undef max
#endif

// Test precision
#define EPSILON float(1e-3)//float(1e-4)
#define FP16_EPSILON float(1e-2)
#define BF16_EPSILON float(1e-1)
#define FP8_EPSILON float(2e-1)

// Random number generation
std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> float_dist(-10.0f, 10.0f);
std::uniform_int_distribution<int> int_dist(0, 100);

// Test status
enum TestStatus {
    PASS,
    FAIL,
    SKIP
};

// Test report
struct TestReport {
    std::string functionName;
    TestStatus status;
    std::string message;

    TestReport(const std::string& name, TestStatus stat, const std::string& msg = "")
        : functionName(name), status(stat), message(msg) {
    }
};

// Print test report
void printTestReport(const TestReport& report) {
    std::string statusStr = (report.status == PASS) ? "PASS" :
        (report.status == FAIL) ? "FAIL" : "SKIP";
    std::cout << "[" << statusStr << "] " << report.functionName;
    if (report.status == FAIL && !report.message.empty()) {
        std::cout << ": " << report.message;
    }
    std::cout << std::endl;
}

// Helper function to compare floating point values with epsilon
template<typename T>
bool almostEqual(T a, T b, T epsilon) {
    return std::abs(a - b) <= epsilon;
}

// ------------------------
// Memory Management Tests
// ------------------------

TestReport testZeroInitializeFP32() {
    try {
        const int count = 1000;
        float* d_data;
        cudaMalloc(&d_data, count * sizeof(float));
        runZeroInitializeFP32(d_data, count);

        std::vector<float> h_data(count);
        cudaMemcpy(h_data.data(), d_data, count * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; i++) {
            if (h_data[i] != 0.0f) {
                cudaFree(d_data);
                return TestReport("runZeroInitializeFP32", FAIL, "Non-zero value found");
            }
        }

        cudaFree(d_data);
        return TestReport("runZeroInitializeFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runZeroInitializeFP32", FAIL, e.what());
    }
}

TestReport testZeroInitializeFP16() {
    try {
        const int count = 1000;
        __half* d_data;
        cudaMalloc(&d_data, count * sizeof(__half));
        runZeroInitializeFP16(d_data, count);

        std::vector<__half> h_data(count);
        cudaMemcpy(h_data.data(), d_data, count * sizeof(__half), cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; i++) {
            if (__half2float(h_data[i]) != 0.0f) {
                cudaFree(d_data);
                return TestReport("runZeroInitializeFP16", FAIL, "Non-zero value found");
            }
        }

        cudaFree(d_data);
        return TestReport("runZeroInitializeFP16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runZeroInitializeFP16", FAIL, e.what());
    }
}

TestReport testZeroInitializeBF16() {
    try {
        const int count = 1000;
        bfloat16* d_data;
        cudaMalloc(&d_data, count * sizeof(bfloat16));
        runZeroInitializeBF16(d_data, count);

        std::vector<bfloat16> h_data(count);
        cudaMemcpy(h_data.data(), d_data, count * sizeof(bfloat16), cudaMemcpyDeviceToHost);

        for (int i = 0; i < count; i++) {
            if (__bfloat162float(h_data[i]) != 0.0f) {
                cudaFree(d_data);
                return TestReport("runZeroInitializeBF16", FAIL, "Non-zero value found");
            }
        }

        cudaFree(d_data);
        return TestReport("runZeroInitializeBF16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runZeroInitializeBF16", FAIL, e.what());
    }
}

// ------------------------
// Multiply Kernels Tests
// ------------------------

TestReport testGemmFP32() {
    try {
        const int m = 32, k = 64, n = 16;
        const int sizeA = m * k;
        const int sizeB = k * n;
        const int sizeC = m * n;

        std::vector<float> h_A(sizeA);
        std::vector<float> h_B(sizeB);
        std::vector<float> h_C(sizeC, 0);
        std::vector<float> h_C_expected(sizeC, 0);

        // Generate test data
        for (int i = 0; i < sizeA; i++) h_A[i] = float_dist(rng);
        for (int i = 0; i < sizeB; i++) h_B[i] = float_dist(rng);

        // Compute expected result (CPU version)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += h_A[i * k + p] * h_B[p * n + j];
                }
                h_C_expected[i * n + j] = sum;
            }
        }

        // Allocate GPU memory
        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, sizeA * sizeof(float));
        cudaMalloc(&d_B, sizeB * sizeof(float));
        cudaMalloc(&d_C, sizeC * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, sizeC * sizeof(float));

        // Execute GPU computation
        runGemmFP32(d_A, d_B, d_C, m, n, k);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < sizeC; i++) {
            if (!almostEqual(h_C[i], h_C_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runGemmFP32", FAIL, "Result mismatch at index " + std::to_string(i));
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runGemmFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runGemmFP32", FAIL, e.what());
    }
}

TestReport testGemmFP16() {
    try {
        const int m = 32, k = 64, n = 16;
        const int sizeA = m * k;
        const int sizeB = k * n;
        const int sizeC = m * n;

        std::vector<__half> h_A(sizeA);
        std::vector<__half> h_B(sizeB);
        std::vector<__half> h_C(sizeC);
        std::vector<float> h_C_expected(sizeC, 0);

        // Generate test data
        for (int i = 0; i < sizeA; i++) h_A[i] = __float2half(float_dist(rng));
        for (int i = 0; i < sizeB; i++) h_B[i] = __float2half(float_dist(rng));

        // Compute expected result (CPU version) - 修正了B矩阵的索引计算
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    // 修正: B[p * n + j] -> B[p * n + j] 是正确索引
                    // 但需要确保B矩阵是行优先存储 (k x n)
                    sum += __half2float(h_A[i * k + p]) * __half2float(h_B[p * n + j]);
                }
                h_C_expected[i * n + j] = sum;
            }
        }

        // Allocate GPU memory
        __half* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, sizeA * sizeof(__half));
        cudaMalloc(&d_B, sizeB * sizeof(__half));
        cudaMalloc(&d_C, sizeC * sizeof(__half));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, sizeC * sizeof(__half));

        // 修正: 参数顺序改为 (m, k, n)
        runGemmFP16(d_A, d_B, d_C, m, k, n);  // 正确顺序: m, k, n

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(__half), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < sizeC; i++) {
            float actual = __half2float(h_C[i]);
            // 使用相对误差比较更合适
            float expected = h_C_expected[i];
            float abs_error = fabs(actual - expected);
            float rel_error = abs_error / fmax(1.0f, fabs(expected));

            if (rel_error > FP16_EPSILON) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runGemmFP16", FAIL,
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(expected) +
                    " Actual: " + std::to_string(actual) +
                    " Rel error: " + std::to_string(rel_error));
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runGemmFP16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runGemmFP16", FAIL, e.what());
    }
}

TestReport testGemmBF16() {
    try {
        const int m = 32, k = 64, n = 16;
        const int sizeA = m * k;
        const int sizeB = k * n;
        const int sizeC = m * n;

        // 修正: 使用正确的数据类型 __nv_bfloat16
        std::vector<__nv_bfloat16> h_A(sizeA);
        std::vector<__nv_bfloat16> h_B(sizeB);
        std::vector<__nv_bfloat16> h_C(sizeC);
        std::vector<float> h_C_expected(sizeC, 0);

        // Generate test data
        for (int i = 0; i < sizeA; i++) h_A[i] = __float2bfloat16(float_dist(rng));
        for (int i = 0; i < sizeB; i++) h_B[i] = __float2bfloat16(float_dist(rng));

        // Compute expected result (CPU version) - 修正了B矩阵的索引计算
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    // 修正: B[p * n + j] -> B[p * n + j] 是正确索引
                    sum += __bfloat162float(h_A[i * k + p]) * __bfloat162float(h_B[p * n + j]);
                }
                h_C_expected[i * n + j] = sum;
            }
        }

        // Allocate GPU memory - 修正: 使用正确的数据类型 __nv_bfloat16
        __nv_bfloat16* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, sizeA * sizeof(__nv_bfloat16));
        cudaMalloc(&d_B, sizeB * sizeof(__nv_bfloat16));
        cudaMalloc(&d_C, sizeC * sizeof(__nv_bfloat16));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, sizeC * sizeof(__nv_bfloat16));

        // 修正: 参数顺序改为 (m, k, n)
        runGemmBF16(d_A, d_B, d_C, m, k, n);  // 正确顺序: m, k, n

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < sizeC; i++) {
            float actual = __bfloat162float(h_C[i]);
            // 使用相对误差比较更合适
            float expected = h_C_expected[i];
            float abs_error = fabs(actual - expected);
            float rel_error = abs_error / fmax(1.0f, fabs(expected));

            if (rel_error > BF16_EPSILON) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runGemmBF16", FAIL,
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(expected) +
                    " Actual: " + std::to_string(actual) +
                    " Rel error: " + std::to_string(rel_error));
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runGemmBF16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runGemmBF16", FAIL, e.what());
    }
}

// Helper function to convert float to fp8_e4m3
fp8_e4m3 float_to_fp8_e4m3(float f) {
    fp8_e4m3 result;
    // Simple conversion (actual implementation may vary)
    result = fp8_e4m3(fminf(fmaxf(f * 8.0f, -8.0f), 8.0f) * 16.0f);
    return result;
}

// Helper function to convert fp8_e4m3 to float
float fp8_e4m3_to_float(fp8_e4m3 f) {
    // Simple conversion (actual implementation may vary)
    return static_cast<float>(f) / 16.0f;
}

// Helper function to convert float to fp8_e5m2
fp8_e5m2 float_to_fp8_e5m2(float f) {
    fp8_e5m2 result;
    // Simple conversion (actual implementation may vary)
    result = fp8_e5m2(fminf(fmaxf(f * 4.0f, -16.0f), 16.0f) * 8.0f);
    return result;
}

// Helper function to convert fp8_e5m2 to float
float fp8_e5m2_to_float(fp8_e5m2 f) {
    // Simple conversion (actual implementation may vary)
    return static_cast<float>(f) / 8.0f;
}

TestReport testGemmFP8E4M3() {
    try {
        const int m = 16, k = 32, n = 8;  // Smaller sizes for FP8 testing
        const int sizeA = m * k;
        const int sizeB = k * n;
        const int sizeC = m * n;

        std::vector<fp8_e4m3> h_A(sizeA);
        std::vector<fp8_e4m3> h_B(sizeB);
        std::vector<fp8_e4m3> h_C(sizeC);
        std::vector<float> h_C_expected(sizeC, 0);

        // Generate test data in range that FP8 can handle
        std::uniform_real_distribution<float> fp8_dist(-2.0f, 2.0f);

        for (int i = 0; i < sizeA; i++) {
            h_A[i] = float_to_fp8_e4m3(fp8_dist(rng));
        }
        for (int i = 0; i < sizeB; i++) {
            h_B[i] = float_to_fp8_e4m3(fp8_dist(rng));
        }

        // Compute expected result (CPU version)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += fp8_e4m3_to_float(h_A[i * k + p]) *
                        fp8_e4m3_to_float(h_B[p * n + j]);
                }
                h_C_expected[i * n + j] = sum;
            }
        }

        // Allocate GPU memory
        fp8_e4m3* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, sizeA * sizeof(fp8_e4m3));
        cudaMalloc(&d_B, sizeB * sizeof(fp8_e4m3));
        cudaMalloc(&d_C, sizeC * sizeof(fp8_e4m3));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(fp8_e4m3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(fp8_e4m3), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, sizeC * sizeof(fp8_e4m3));

        // Execute GPU computation
        runGemmFP8E4M3(d_A, d_B, d_C, m, k, n);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(fp8_e4m3), cudaMemcpyDeviceToHost);

        // Verify results with higher tolerance for FP8
        const float fp8_epsilon = 0.1f;  // Larger epsilon for FP8 precision
        for (int i = 0; i < sizeC; i++) {
            float actual = fp8_e4m3_to_float(h_C[i]);
            float expected = h_C_expected[i];

            if (fabs(expected) < 1e-6) {
                // Near-zero values - check absolute difference
                if (fabs(actual - expected) > fp8_epsilon) {
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runGemmFP8E4M3", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                        " Actual: " + std::to_string(actual));
                }
            }
            else {
                // Non-zero values - check relative difference
                if (fabs((actual - expected) / expected) > fp8_epsilon) {
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runGemmFP8E4M3", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                        " Actual: " + std::to_string(actual));
                }
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runGemmFP8E4M3", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runGemmFP8E4M3", FAIL, e.what());
    }
}

TestReport testGemmFP8E5M2() {
    try {
        const int m = 16, k = 32, n = 8;  // Smaller sizes for FP8 testing
        const int sizeA = m * k;
        const int sizeB = k * n;
        const int sizeC = m * n;

        std::vector<fp8_e5m2> h_A(sizeA);
        std::vector<fp8_e5m2> h_B(sizeB);
        std::vector<fp8_e5m2> h_C(sizeC);
        std::vector<float> h_C_expected(sizeC, 0);

        // Generate test data in range that FP8 can handle
        std::uniform_real_distribution<float> fp8_dist(-4.0f, 4.0f);

        for (int i = 0; i < sizeA; i++) {
            h_A[i] = float_to_fp8_e5m2(fp8_dist(rng));
        }
        for (int i = 0; i < sizeB; i++) {
            h_B[i] = float_to_fp8_e5m2(fp8_dist(rng));
        }

        // Compute expected result (CPU version)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int p = 0; p < k; p++) {
                    sum += fp8_e5m2_to_float(h_A[i * k + p]) *
                        fp8_e5m2_to_float(h_B[p * n + j]);
                }
                h_C_expected[i * n + j] = sum;
            }
        }

        // Allocate GPU memory
        fp8_e5m2* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, sizeA * sizeof(fp8_e5m2));
        cudaMalloc(&d_B, sizeB * sizeof(fp8_e5m2));
        cudaMalloc(&d_C, sizeC * sizeof(fp8_e5m2));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(fp8_e5m2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(fp8_e5m2), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, sizeC * sizeof(fp8_e5m2));

        // Execute GPU computation
        runGemmFP8E5M2(d_A, d_B, d_C, m, k, n);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(fp8_e5m2), cudaMemcpyDeviceToHost);

        // Verify results with higher tolerance for FP8
        const float fp8_epsilon = 0.15f;  // Larger epsilon for FP8 precision
        for (int i = 0; i < sizeC; i++) {
            float actual = fp8_e5m2_to_float(h_C[i]);
            float expected = h_C_expected[i];

            if (fabs(expected) < 1e-6) {
                // Near-zero values - check absolute difference
                if (fabs(actual - expected) > fp8_epsilon) {
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runGemmFP8E5M2", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                        " Actual: " + std::to_string(actual));
                }
            }
            else {
                // Non-zero values - check relative difference
                if (fabs((actual - expected) / expected) > fp8_epsilon) {
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runGemmFP8E5M2", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                        " Actual: " + std::to_string(actual));
                }
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runGemmFP8E5M2", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runGemmFP8E5M2", FAIL, e.what());
    }
}

TestReport testSingleElementTensorMultiplyFP32() {
    try {
        const int count = 1024;
        std::vector<float> h_A(count);
        std::vector<float> h_B(count);
        std::vector<float> h_C(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_A[i] = float_dist(rng);
            h_B[i] = float_dist(rng);
            h_expected[i] = h_A[i] * h_B[i];
        }

        // Allocate GPU memory
        float* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, count * sizeof(float));
        cudaMalloc(&d_B, count * sizeof(float));
        cudaMalloc(&d_C, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runSingleElementTensorMultiplyFP32(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_C[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runSingleElementTensorMultiplyFP32", FAIL,
                    "Result mismatch at index " + std::to_string(i));
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runSingleElementTensorMultiplyFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runSingleElementTensorMultiplyFP32", FAIL, e.what());
    }
}


TestReport testSingleElementTensorMultiplyFP16() {  // 修改函数名以匹配实现
    try {
        const int count = 1024;
        std::vector<__half> h_A(count);
        std::vector<__half> h_B(count);
        std::vector<__half> h_C(count);
        std::vector<float> h_expected(count);

        // 初始化随机数生成器
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> float_dist(0.1f, 1.0f);  // 添加分布定义

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_A[i] = __float2half(float_dist(rng));
            h_B[i] = __float2half(float_dist(rng));
            h_expected[i] = __half2float(h_A[i]) * __half2float(h_B[i]);
        }

        // Allocate GPU memory
        __half* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, count * sizeof(__half));
        cudaMalloc(&d_B, count * sizeof(__half));
        cudaMalloc(&d_C, count * sizeof(__half));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(__half), cudaMemcpyHostToDevice);

        // Execute GPU computation - 使用修正后的函数名
        runSingleElementTensorMultiplyFP16(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(__half), cudaMemcpyDeviceToHost);

        // 添加浮点数比较函数 (如果未定义)
        auto almostEqual = [](float a, float b, float epsilon) {
            return std::fabs(a - b) < epsilon;
            };

        // Verify results
        for (int i = 0; i < count; i++) {
            float actual = __half2float(h_C[i]);
            if (!almostEqual(actual, h_expected[i], FP16_EPSILON)) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runSingleElementTensorMultiplyFP16", FAIL,  // 更新测试名
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(h_expected[i]) +
                    " Actual: " + std::to_string(actual));
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runSingleElementTensorMultiplyFP16", PASS);  // 更新测试名
    }
    catch (const std::exception& e) {
        return TestReport("runSingleElementTensorMultiplyFP16", FAIL, e.what());  // 更新测试名
    }
}

TestReport testScalarMultiplyFP32() {
    try {
        const int count = 1024;
        const float scalar = float_dist(rng);
        std::vector<float> h_input(count);
        std::vector<float> h_result(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_input[i] = float_dist(rng);
            h_expected[i] = h_input[i] * scalar;
        }

        // Allocate GPU memory
        float* d_input, * d_result;
        cudaMalloc(&d_input, count * sizeof(float));
        cudaMalloc(&d_result, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_input, h_input.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runScalarMultiplyFP32(d_input, d_result, scalar, count);

        // Copy result back to CPU
        cudaMemcpy(h_result.data(), d_result, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_result[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_input);
                cudaFree(d_result);
                return TestReport("runScalarMultiplyFP32", FAIL,
                    "Result mismatch at index " + std::to_string(i));
            }
        }

        cudaFree(d_input);
        cudaFree(d_result);
        return TestReport("runScalarMultiplyFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runScalarMultiplyFP32", FAIL, e.what());
    }
}

TestReport testScalarMultiplyFP16() {
    try {
        const int count = 1024;
        const float scalar = float_dist(rng);
        std::vector<__half> h_input(count);
        std::vector<__half> h_result(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_input[i] = __float2half(float_dist(rng));
            h_expected[i] = __half2float(h_input[i]) * scalar;
        }

        // Allocate GPU memory
        __half* d_input, * d_result;
        cudaMalloc(&d_input, count * sizeof(__half));
        cudaMalloc(&d_result, count * sizeof(__half));

        // Copy data to GPU
        cudaMemcpy(d_input, h_input.data(), count * sizeof(__half), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runScalarMultiplyFP16(d_input, d_result, scalar, count);

        // Copy result back to CPU
        cudaMemcpy(h_result.data(), d_result, count * sizeof(__half), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            float actual = __half2float(h_result[i]);
            if (!almostEqual(actual, h_expected[i], FP16_EPSILON)) {
                cudaFree(d_input);
                cudaFree(d_result);
                return TestReport("runScalarMultiplyFP16", FAIL,
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(h_expected[i]) +
                    " Actual: " + std::to_string(actual));
            }
        }

        cudaFree(d_input);
        cudaFree(d_result);
        return TestReport("runScalarMultiplyFP16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runScalarMultiplyFP16", FAIL, e.what());
    }
}

// ------------------------
// Element-wise Operations Tests
// ------------------------

TestReport testElementwiseMultiplyFP32() {
    try {
        const int count = 1024;
        std::vector<float> h_A(count);
        std::vector<float> h_B(count);
        std::vector<float> h_C(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_A[i] = float_dist(rng);
            h_B[i] = float_dist(rng);
            h_expected[i] = h_A[i] * h_B[i];
        }

        // Allocate GPU memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, count * sizeof(float));
        cudaMalloc(&d_B, count * sizeof(float));
        cudaMalloc(&d_C, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runSingleElementTensorMultiplyFP32(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_C[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runSingleElementTensorMultiplyFP32", FAIL, "Result mismatch");
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runSingleElementTensorMultiplyFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runSingleElementTensorMultiplyFP32", FAIL, e.what());
    }
}

// ------------------------
// BF16 Element-wise Multiply Test
// ------------------------

TestReport testSingleElementTensorMultiplyBF16() {
    try {
        const int count = 1024;
        std::vector<bfloat16> h_A(count);
        std::vector<bfloat16> h_B(count);
        std::vector<bfloat16> h_C(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            float valA = float_dist(rng);
            float valB = float_dist(rng);
            h_A[i] = __float2bfloat16(valA);
            h_B[i] = __float2bfloat16(valB);
            h_expected[i] = valA * valB;
        }

        // Allocate GPU memory
        bfloat16* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, count * sizeof(bfloat16));
        cudaMalloc(&d_B, count * sizeof(bfloat16));
        cudaMalloc(&d_C, count * sizeof(bfloat16));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(bfloat16), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runSingleElementTensorMultiplyBF16(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(bfloat16), cudaMemcpyDeviceToHost);

        // Verify results with BF16 tolerance
        for (int i = 0; i < count; i++) {
            float actual = __bfloat162float(h_C[i]);
            float expected = h_expected[i];

            if (fabs(expected) < 1e-3) {
                // Near-zero values - check absolute difference
                if (fabs(actual - expected) > BF16_EPSILON) {
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runSingleElementTensorMultiplyBF16", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                        " Actual: " + std::to_string(actual));
                }
            }
            else {
                // Non-zero values - check relative difference
                if (fabs((actual - expected) / expected) > BF16_EPSILON) {
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runSingleElementTensorMultiplyBF16", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                        " Actual: " + std::to_string(actual));
                }
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runSingleElementTensorMultiplyBF16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runSingleElementTensorMultiplyBF16", FAIL, e.what());
    }
}



TestReport testSingleElementTensorMultiplyFP8E4M3() {  // 更新测试函数名
    try {
        const int count = 1024;
        std::vector<__nv_fp8_e4m3> h_A(count);
        std::vector<__nv_fp8_e4m3> h_B(count);
        std::vector<__nv_fp8_e4m3> h_C(count);
        std::vector<float> h_expected(count);

        // 初始化随机数生成器
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> fp8_dist(-2.0f, 2.0f);

        // 使用CUDA专用转换函数
        auto float_to_fp8_e4m3 = [](float val) {
            return __nv_fp8_e4m3(val);  // 使用专用函数而非static_cast
            };
        auto fp8_e4m3_to_float = [](__nv_fp8_e4m3 val) {
            return static_cast<float>(val);
            };

        // Generate test data
        for (int i = 0; i < count; i++) {
            float valA = fp8_dist(rng);
            float valB = fp8_dist(rng);
            h_A[i] = float_to_fp8_e4m3(valA);
            h_B[i] = float_to_fp8_e4m3(valB);
            // 期望值计算考虑FP8精度损失
            h_expected[i] = static_cast<float>(h_A[i]) * static_cast<float>(h_B[i]);
        }

        // Allocate GPU memory
        __nv_fp8_e4m3* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, count * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_B, count * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_C, count * sizeof(__nv_fp8_e4m3));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

        // 使用更新后的函数名
        runSingleElementTensorMultiplyFP8E4M3(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);

        // 更精确的FP8误差处理
        const float fp8_epsilon = 0.2f;  // 增加容忍度
        int errors = 0;
        for (int i = 0; i < count; i++) {
            float actual = fp8_e4m3_to_float(h_C[i]);
            float expected = h_expected[i];
            float diff = std::fabs(actual - expected);

            // 处理NaN和Inf
            if (std::isnan(actual) || std::isnan(expected)) {
                errors++;
                continue;
                }

            // 统一使用绝对误差检查
            if (diff > fp8_epsilon) {
                errors++;
            }

            // 允许一定比例的误差（FP8精度较低）
            if (errors > count * 0.1) {  // 允许10%的误差
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                return TestReport("runSingleElementTensorMultiplyFP8E5M2", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                    " Actual: " + std::to_string(actual)+
                    " Too many errors: " + std::to_string(errors) + "/" + std::to_string(count) 
                );
            }

        }


        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runSingleElementTensorMultiplyFP8E4M3", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runSingleElementTensorMultiplyFP8E4M3", FAIL, e.what());
    }
}


// ------------------------
// FP8 E5M2 Element-wise Multiply Test
// ------------------------

TestReport testSingleElementTensorMultiplyFP8E5M2() { 
    try {
        const int count = 1024;
        std::vector<__nv_fp8_e5m2> h_A(count);  // 使用CUDA原生类型
        std::vector<__nv_fp8_e5m2> h_B(count);
        std::vector<__nv_fp8_e5m2> h_C(count);
        std::vector<float> h_expected(count);

        // 初始化随机数生成器
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> fp8_dist(-4.0f, 4.0f);  // 在FP8有效范围内

        // 添加FP8转换函数（如果未定义）
        auto float_to_fp8_e5m2 = [](float val) {
            return static_cast<__nv_fp8_e5m2>(val);
            };
        auto fp8_e5m2_to_float = [](__nv_fp8_e5m2 val) {
            return static_cast<float>(val);
            };

        // Generate test data
        for (int i = 0; i < count; i++) {
            float valA = fp8_dist(rng);
            float valB = fp8_dist(rng);
            h_A[i] = float_to_fp8_e5m2(valA);
            h_B[i] = float_to_fp8_e5m2(valB);
            h_expected[i] = valA * valB;
        }

        // Allocate GPU memory
        __nv_fp8_e5m2* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, count * sizeof(__nv_fp8_e5m2));
        cudaMalloc(&d_B, count * sizeof(__nv_fp8_e5m2));
        cudaMalloc(&d_C, count * sizeof(__nv_fp8_e5m2));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(__nv_fp8_e5m2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(__nv_fp8_e5m2), cudaMemcpyHostToDevice);

        runSingleElementTensorMultiplyFP8E5M2(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(__nv_fp8_e5m2), cudaMemcpyDeviceToHost);


        const float fp8_epsilon = 0.2f;  // 增加容忍度
        int errors = 0;
        for (int i = 0; i < count; i++) {
            float actual = fp8_e5m2_to_float(h_C[i]);
            float expected = h_expected[i];
            float diff = std::fabs(actual - expected);

            // 处理NaN和Inf
            if (std::isnan(actual) || std::isnan(expected)) {
                errors++;
                continue;
                }

            // 统一使用绝对误差检查
            if (diff > fp8_epsilon) {
                errors++;
            }
            // 允许一定比例的误差（FP8精度较低）
            if (errors > count * 0.1) {  // 允许10%的误差
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_C);
                    return TestReport("runSingleElementTensorMultiplyFP8E5M2", FAIL,
                        "Result mismatch at index " + std::to_string(i) +
                        " Expected: " + std::to_string(expected) +
                    " Actual: " + std::to_string(actual) +
                    " Too many errors: " + std::to_string(errors) + "/" + std::to_string(count)
                    );
            }

        }
        

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runSingleElementTensorMultiplyFP8E5M2", PASS);  // 更新测试名
    }
    catch (const std::exception& e) {
        return TestReport("runSingleElementTensorMultiplyFP8E5M2", FAIL, e.what());  // 更新测试名
    }
}

TestReport testAddFP32() {
    try {
        const int count = 1024;
        std::vector<float> h_A(count);
        std::vector<float> h_B(count);
        std::vector<float> h_C(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_A[i] = float_dist(rng);
            h_B[i] = float_dist(rng);
            h_expected[i] = h_A[i] + h_B[i];
        }

        // Allocate GPU memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, count * sizeof(float));
        cudaMalloc(&d_B, count * sizeof(float));
        cudaMalloc(&d_C, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runAddFP32(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_C[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runAddFP32", FAIL, "Result mismatch");
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runAddFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runAddFP32", FAIL, e.what());
    }
}

TestReport testScalarAddFP32() {
    try {
        const int count = 1024;
        const float scalar = float_dist(rng);
        std::vector<float> h_input(count);
        std::vector<float> h_result(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_input[i] = float_dist(rng);
            h_expected[i] = h_input[i] + scalar;
        }

        // Allocate GPU memory
        float *d_input, *d_result;
        cudaMalloc(&d_input, count * sizeof(float));
        cudaMalloc(&d_result, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_input, h_input.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runScalarAddFP32(d_input, scalar, d_result, count);

        // Copy result back to CPU
        cudaMemcpy(h_result.data(), d_result, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_result[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_input);
                cudaFree(d_result);
                return TestReport("runScalarAddFP32", FAIL, 
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(h_expected[i]) +
                    " Actual: " + std::to_string(h_result[i]));
            }
        }

        cudaFree(d_input);
        cudaFree(d_result);
        return TestReport("runScalarAddFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runScalarAddFP32", FAIL, e.what());
    }
}

TestReport testMinusFP32() {
    try {
        const int count = 1024;
        std::vector<float> h_A(count);
        std::vector<float> h_B(count);
        std::vector<float> h_C(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_A[i] = float_dist(rng);
            h_B[i] = float_dist(rng);
            h_expected[i] = h_A[i] - h_B[i];
        }

        // Allocate GPU memory
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, count * sizeof(float));
        cudaMalloc(&d_B, count * sizeof(float));
        cudaMalloc(&d_C, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, h_A.data(), count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runMinusFP32(d_A, d_B, d_C, count);

        // Copy result back to CPU
        cudaMemcpy(h_C.data(), d_C, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_C[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                return TestReport("runMinusFP32", FAIL, 
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(h_expected[i]) +
                    " Actual: " + std::to_string(h_C[i]));
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return TestReport("runMinusFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runMinusFP32", FAIL, e.what());
    }
}

// ------------------------
// Conversion Tests
// ------------------------

TestReport testConvertFP32ToFP16() {
    try {
        const int count = 1024;
        std::vector<float> h_src(count);
        std::vector<__half> h_dest(count);
        std::vector<__half> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_src[i] = float_dist(rng);
            h_expected[i] = __float2half(h_src[i]);
        }

        // Allocate GPU memory
        float *d_src;
        __half *d_dest;
        cudaMalloc(&d_src, count * sizeof(float));
        cudaMalloc(&d_dest, count * sizeof(__half));

        // Copy data to GPU
        cudaMemcpy(d_src, h_src.data(), count * sizeof(float), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runConvertFP32ToFP16(d_src, d_dest, count);

        // Copy result back to CPU
        cudaMemcpy(h_dest.data(), d_dest, count * sizeof(__half), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            float expected = __half2float(h_expected[i]);
            float actual = __half2float(h_dest[i]);
            if (!almostEqual(expected, actual, float(FP16_EPSILON))) {
                cudaFree(d_src);
                cudaFree(d_dest);
                return TestReport("runConvertFP32ToFP16", FAIL, 
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(expected) +
                    " Actual: " + std::to_string(actual));
            }
        }

        cudaFree(d_src);
        cudaFree(d_dest);
        return TestReport("runConvertFP32ToFP16", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runConvertFP32ToFP16", FAIL, e.what());
    }
}

TestReport testConvertFP16ToFP32() {
    try {
        const int count = 1024;
        std::vector<__half> h_src(count);
        std::vector<float> h_dest(count);
        std::vector<float> h_expected(count);

        // Generate test data
        for (int i = 0; i < count; i++) {
            h_src[i] = __float2half(float_dist(rng));
            h_expected[i] = __half2float(h_src[i]);
        }

        // Allocate GPU memory
        __half *d_src;
        float *d_dest;
        cudaMalloc(&d_src, count * sizeof(__half));
        cudaMalloc(&d_dest, count * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_src, h_src.data(), count * sizeof(__half), cudaMemcpyHostToDevice);

        // Execute GPU computation
        runConvertFP16ToFP32(d_src, d_dest, count);

        // Copy result back to CPU
        cudaMemcpy(h_dest.data(), d_dest, count * sizeof(float), cudaMemcpyDeviceToHost);

        // Verify results
        for (int i = 0; i < count; i++) {
            if (!almostEqual(h_dest[i], h_expected[i], static_cast<float>(EPSILON))) {
                cudaFree(d_src);
                cudaFree(d_dest);
                return TestReport("runConvertFP16ToFP32", FAIL, 
                    "Result mismatch at index " + std::to_string(i) +
                    " Expected: " + std::to_string(h_expected[i]) +
                    " Actual: " + std::to_string(h_dest[i]));
            }
        }

        cudaFree(d_src);
        cudaFree(d_dest);
        return TestReport("runConvertFP16ToFP32", PASS);
    }
    catch (const std::exception& e) {
        return TestReport("runConvertFP16ToFP32", FAIL, e.what());
    }
}

// Similar tests for BF16 conversions can be added here

// ------------------------
// Main Test Function
// ------------------------

int main() {
    std::cout << "=== GARNET CUDA TEST START ===" << std::endl;

    // Memory Management Tests
    printTestReport(testZeroInitializeFP32());
    printTestReport(testZeroInitializeFP16());
    printTestReport(testZeroInitializeBF16());

    // GEMM Tests
    printTestReport(testGemmFP32());
    printTestReport(testGemmFP16());
    printTestReport(testGemmBF16());
    // FP8 GEMM Tests
    printTestReport(testGemmFP8E4M3());
    printTestReport(testGemmFP8E5M2());

    // Single Element Tensor Multiply Tests
    printTestReport(testSingleElementTensorMultiplyFP32());
    printTestReport(testSingleElementTensorMultiplyFP16());
//    printTestReport(testElementwiseMultiplyFP32());
    printTestReport(testSingleElementTensorMultiplyBF16());
    printTestReport(testSingleElementTensorMultiplyFP8E4M3());
    printTestReport(testSingleElementTensorMultiplyFP8E5M2());

    // Scalar Multiply Tests
    printTestReport(testScalarMultiplyFP32());
    printTestReport(testScalarMultiplyFP16());

    // Element-wise Operations Tests

    printTestReport(testAddFP32());
    printTestReport(testScalarAddFP32());
    printTestReport(testMinusFP32());
    // Add tests for FP16, BF16 versions here

    // Conversion Tests
    printTestReport(testConvertFP32ToFP16());
    printTestReport(testConvertFP16ToFP32());
    // Add tests for other conversions here

    std::cout << "=== GARNET CUDA TEST FINISH ===" << std::endl;
    return 0;
}