// SPDX-License-Identifier: Apache-2.0
// Untimed real-checkpoint control. Preserve both outputs even on mismatch.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "modelopt_cublaslt.h"
#include "modelopt_fp8.cuh"

// Observe the exact native helper's selected algorithm without changing calls.
static cublasStatus_t ObserveMatmul(cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation, const void* alpha, const void* a,
    cublasLtMatrixLayout_t ad, const void* b, cublasLtMatrixLayout_t bd,
    const void* beta, const void* c, cublasLtMatrixLayout_t cd,
    void* d, cublasLtMatrixLayout_t dd, const cublasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceBytes, cudaStream_t stream) {
    const auto status = cublasLtMatmul(handle, operation, alpha, a, ad, b, bd,
        beta, c, cd, d, dd, algo, workspace, workspaceBytes, stream);
    int id = -1; size_t written = 0;
    const auto query = cublasLtMatmulAlgoConfigGetAttribute(algo,
        CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written);
    printf("ALGORITHM status=%d id_query=%d id=%d workspace=%zu opaque=",
        int(status), int(query), id, workspaceBytes);
    const auto* bytes = reinterpret_cast<const unsigned char*>(algo);
    for (size_t i = 0; i < sizeof(*algo); ++i) printf("%02x", bytes[i]);
    puts("");
    return status;
}
#define cublasLtMatmul ObserveMatmul
#include "modelopt_cublaslt.cpp"
#undef cublasLtMatmul

static void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static std::vector<unsigned char> Read(const std::string& name, size_t size) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(size))
        throw std::runtime_error("Input missing or wrong size: " + name);
    f.seekg(0); std::vector<unsigned char> data(size);
    if (!f.read(reinterpret_cast<char*>(data.data()), size))
        throw std::runtime_error("Input read failed: " + name);
    return data;
}
static void Write(const std::string& name, const std::vector<unsigned char>& data) {
    std::ofstream f(name, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(data.data()), data.size()))
        throw std::runtime_error("Output write failed: " + name);
}
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { Check(cudaMalloc(&p, bytes)); }
    ~Device() { if (p) cudaFree(p); }
};
__global__ void Cast(const float* input, __nv_bfloat16* output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(input[i]);
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: control FIXTURE OUTPUT_EXISTING_DIR");
        const std::string fixture = std::string(argv[1]) + "/";
        const std::string output = std::string(argv[2]) + "/";
        constexpr int n = 5120, k = 6144;
        const auto weights = Read(fixture + "weight-n5120-k6144.e4m3", n * k);
        const auto ws = Read(fixture + "weight-scale.f32", 4);
        const auto xs = Read(fixture + "input-scale.f32", 4);
        float weightScale, inputScale;
        std::memcpy(&weightScale, ws.data(), 4); std::memcpy(&inputScale, xs.data(), 4);
        cublasLtHandle_t handle = nullptr;
        if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLASLt creation failed");
        printf("CUBLASLT_VERSION %zu\n", cublasLtGetVersion());
        Device w(weights.size()), workspace(4ULL << 20);
        Check(cudaMemcpy(w.p, weights.data(), weights.size(), cudaMemcpyHostToDevice));
        size_t totalChanged = 0;
        for (int m : {16, 32}) {
            const auto input = Read(fixture + "input-m" + std::to_string(m) + "-k6144.f32", m * k * 4);
            const auto expected = Read(fixture + "expected-m" + std::to_string(m) + "-n5120.bf16", m * n * 2);
            Device x(input.size()), bf16(m * k * 2), packed(m * k), result(m * n * 2);
            Check(cudaMemcpy(x.p, input.data(), input.size(), cudaMemcpyHostToDevice));
            Cast<<<(m * k + 255) / 256, 256>>>(static_cast<const float*>(x.p),
                static_cast<__nv_bfloat16*>(bf16.p), m * k);
            Check(cudaGetLastError());
            Check(Garnet::QuantizeModelOptFP8Activation(bf16.p,
                static_cast<unsigned char*>(packed.p), inputScale, m, k, nullptr));
            std::vector<unsigned char> cast(m * k * 2), quantized(m * k);
            Check(cudaMemcpy(cast.data(), bf16.p, cast.size(), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(quantized.data(), packed.p, quantized.size(), cudaMemcpyDeviceToHost));
            Write(output + "cast-m" + std::to_string(m) + ".bf16", cast);
            Write(output + "quantized-m" + std::to_string(m) + ".e4m3", quantized);
            for (int repeat = 0; repeat < 2; ++repeat) {
                printf("CONTROL m=%d repeat=%d\n", m, repeat);
                const auto status = Garnet::RunModelOptFP8Matmul(handle, w.p, packed.p,
                    result.p, m, n, k, weightScale * inputScale, workspace.p, 4ULL << 20, nullptr);
                if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("Native helper matmul failed");
                Check(cudaDeviceSynchronize());
                std::vector<unsigned char> actual(m * n * 2);
                Check(cudaMemcpy(actual.data(), result.p, actual.size(), cudaMemcpyDeviceToHost));
                Write(output + "actual-m" + std::to_string(m) + "-r" + std::to_string(repeat) + ".bf16", actual);
                size_t changed = 0;
                for (size_t i = 0; i < actual.size(); i += 2)
                    if (actual[i] != expected[i] || actual[i + 1] != expected[i + 1]) ++changed;
                totalChanged += changed;
                printf("EXACT_CONTROL m=%d repeat=%d changed_words=%zu total_words=%d\n", m, repeat, changed, m * n);
            }
        }
        cublasLtDestroy(handle);
        puts(totalChanged ? "REAL_PROJECTION_CONTROL_FAILED" : "REAL_PROJECTION_CONTROL_PASSED");
        return totalChanged ? 1 : 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR %s\n", e.what()); return 2;
    }
}
