#include "../../src/image/qwen_vl/qwen_vl_image_preprocessor.h"
#include "../../src/image/qwen_vl/qwen_vl_vision_metadata.h"
#include "../../src/image/cuda/jpeg_decode_nvjpeg.h"
#include "../../src/tensor/tensor_helper.h"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <thread>
#include <chrono>

namespace {
using namespace Garnet;
using namespace Garnet::Image;
using namespace Garnet::Image::QwenVL;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct OutputMode {
    std::string previous;
    bool existed;
    explicit OutputMode(bool cpu) {
        const char* value = std::getenv("GARNET_TRT_SYNC_CPU_OUTPUTS");
        existed = value != nullptr;
        if (value) previous = value;
        Set(cpu ? "1" : "0");
    }
    static void Set(const char* value) {
#ifdef _WIN32
        if (_putenv_s("GARNET_TRT_SYNC_CPU_OUTPUTS", value ? value : ""))
            throw std::runtime_error("cannot set test output mode");
#else
        if (value ? setenv("GARNET_TRT_SYNC_CPU_OUTPUTS", value, 1) : unsetenv("GARNET_TRT_SYNC_CPU_OUTPUTS"))
            throw std::runtime_error("cannot set test output mode");
#endif
    }
    ~OutputMode() { try { Set(existed ? previous.c_str() : nullptr); } catch (...) {} }
};

std::vector<float> Pixels(int height, int width, int channels) {
    std::vector<float> result(static_cast<size_t>(height) * width * channels);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c)
                result[(y * width + x) * channels + c] = static_cast<float>((y * 13 + x * 7 + c * 61) % 256);
    return result;
}

// Enumerate patches spatially, independently of the CUDA flattened index math.
std::vector<float> Reference(const std::vector<float>& pixels, int h, int w,
    const QwenVLImagePreprocessConfig& config) {
    const int channels = ChannelCount(config.pixelFormat);
    const bool reverse = config.pixelFormat == PixelFormat::BGR_FLOAT32_HWC ||
        config.pixelFormat == PixelFormat::BGRA_FLOAT32_HWC;
    std::vector<float> expected;
    for (int by = 0; by < h / config.patchSize; by += config.mergeSize)
        for (int bx = 0; bx < w / config.patchSize; bx += config.mergeSize)
            for (int my = 0; my < config.mergeSize; ++my)
                for (int mx = 0; mx < config.mergeSize; ++mx)
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < config.temporalPatchSize; ++t)
                            for (int py = 0; py < config.patchSize; ++py)
                                for (int px = 0; px < config.patchSize; ++px) {
                                    const int y = (by + my) * config.patchSize + py;
                                    const int x = (bx + mx) * config.patchSize + px;
                                    float value = pixels[(y * w + x) * channels + (reverse ? 2 - c : c)];
                                    expected.push_back((value / config.inputScale - config.mean[c]) / config.std[c]);
                                }
    return expected;
}

void Compare(const X::Value& value, const std::vector<float>& expected) {
    auto cpu = TensorHelper::CopyToCPU(X::Tensor(value));
    auto info = cpu.Info();
    Require(info.dtype == X3_TENSOR_FLOAT32 && info.byte_size == expected.size() * sizeof(float), "pixel dtype/size");
    const auto* actual = static_cast<const float*>(info.data);
    for (size_t i = 0; i < expected.size(); ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-6f)
            throw std::runtime_error("pixel transform mismatch at " + std::to_string(i));
}

void Metadata(const PreprocessResult& result, int gh, int gw, int merge) {
    auto grid = TensorHelper::CopyToCPU(X::Tensor(result.imageGridTHW));
    Require(grid.Info().dtype == X3_TENSOR_INT64 && grid.Info().byte_size == 24, "image grid storage");
    auto* gridData = static_cast<const int64_t*>(grid.Info().data);
    Require(gridData[0] == 1 && gridData[1] == gh && gridData[2] == gw, "image grid values");
    auto positions = TensorHelper::CopyToCPU(X::Tensor(result.visionPositionIds));
    auto seq = TensorHelper::CopyToCPU(X::Tensor(result.visionCuSeqlens));
    auto indices = TensorHelper::CopyToCPU(X::Tensor(result.bilinearIndices));
    auto weights = TensorHelper::CopyToCPU(X::Tensor(result.bilinearWeights));
    Require(positions.Info().dtype == X3_TENSOR_INT64 && positions.Info().byte_size == uint64_t(gh * gw) * 16, "position storage");
    Require(seq.Info().dtype == X3_TENSOR_INT32 && seq.Info().byte_size == 8, "sequence storage");
    Require(indices.Info().dtype == X3_TENSOR_INT64 && indices.Info().byte_size == uint64_t(gh * gw) * 32, "index storage");
    Require(weights.Info().dtype == X3_TENSOR_BFLOAT16 && weights.Info().byte_size == uint64_t(gh * gw) * 8, "weight storage");
    auto* seqlens = static_cast<const int32_t*>(seq.Info().data);
    Require(seqlens[0] == 0 && seqlens[1] == gh * gw, "sequence values");
    auto* pos = static_cast<const int64_t*>(positions.Info().data);
    auto* idx = static_cast<const int64_t*>(indices.Info().data);
    auto* weight = static_cast<const __nv_bfloat16*>(weights.Info().data);
    int n = 0;
    for (int by = 0; by < gh; by += merge)
        for (int bx = 0; bx < gw; bx += merge)
            for (int y = 0; y < merge; ++y)
                for (int x = 0; x < merge; ++x, ++n) {
                    Require(pos[n * 2] == by + y && pos[n * 2 + 1] == bx + x, "position ordering");
                    float sum = 0;
                    for (int k = 0; k < 4; ++k) {
                        Require(idx[n * 4 + k] >= 0 && idx[n * 4 + k] < 2304, "bilinear index bounds");
                        float w = __bfloat162float(weight[n * 4 + k]);
                        Require(w >= 0 && w <= 1, "bilinear weight bounds");
                        sum += w;
                    }
                    Require(std::abs(sum - 1) < 0.01f, "bilinear weights sum");
                }
}

void RawSources(X::Runtime& runtime) {
    constexpr int h = 32, w = 64;
    for (auto format : {PixelFormat::RGB_FLOAT32_HWC, PixelFormat::BGR_FLOAT32_HWC,
        PixelFormat::RGBA_FLOAT32_HWC, PixelFormat::BGRA_FLOAT32_HWC}) {
        QwenVLImagePreprocessConfig config;
        config.pixelFormat = format;
        config.mean[1] = 0.3f;
        config.std[2] = 0.7f;
        int channels = ChannelCount(format);
        auto pixels = Pixels(h, w, channels);
        auto expected = Reference(pixels, h, w, config);
        for (bool gpuSource : {false, true}) for (bool cpuOutput : {false, true}) {
            OutputMode mode(cpuOutput);
            auto source = X::Tensor::Create(runtime.host(), X3_TENSOR_FLOAT32, {h, w, channels}, pixels.data(), pixels.size() * sizeof(float));
            if (gpuSource) source = TensorHelper::CopyToGPU(source);
            auto before = source.Info();
            auto result = PreprocessRawImageTensor(source, h, w, config);
            Require(source.Info().data == before.data && source.Info().device_type == before.device_type, "input storage mutated");
            Compare(source, pixels);
            auto output = X::Tensor(result.pixelValues);
            auto info = output.Info();
            Require(info.rank == 2 && info.shape[0] == 8 && info.shape[1] == 1536, "output shape");
            Require(info.device_type == (cpuOutput ? 0 : TensorHelper::CudaDevice), "output residency");
            Compare(output, expected);
            Metadata(result, 2, 4, 2);
            auto retained = output.View({8, 1536}, {1536 * 4, 4});
            source = X::Tensor();
            output = X::Tensor();
            result = {};
            Compare(retained, expected);
        }
    }
    std::cout << "garnet-image-cpu-gpu-pixel-parity-passed\n";
}

void Ownership(X::Runtime& runtime) {
    auto pixels = Pixels(32, 32, 3);
    void* allocation = nullptr;
    Check(cudaMalloc(&allocation, pixels.size() * sizeof(float) + 16));
    int releases = 0;
    auto owner = std::shared_ptr<void>(allocation, [&releases](void* p) { cudaFree(p); ++releases; });
    Check(cudaMemcpy(static_cast<char*>(allocation) + 16, pixels.data(), pixels.size() * sizeof(float), cudaMemcpyHostToDevice));
    int64_t shape[] = {static_cast<int64_t>(pixels.size() + 4)}, strides[] = {4};
    X3TensorInfo info{};
    info.size = sizeof(info); info.rank = 1; info.shape = shape; info.strides = strides;
    info.dtype = X3_TENSOR_FLOAT32; info.data = allocation; info.byte_size = pixels.size() * sizeof(float) + 16;
    info.device_type = TensorHelper::CudaDevice;
    Check(cudaGetDevice(&info.device_id));
    auto parent = TensorHelper::WrapBorrowedGPU(runtime.host(), info, owner);
    auto view = parent.View({32, 32, 3}, {32 * 3 * 4, 3 * 4, 4}, 16);
    owner.reset(); parent = X::Tensor();
    Require(releases == 0, "input view lost ownership");
    OutputMode mode(false);
    QwenVLImagePreprocessConfig config;
    auto result = PreprocessRawImageTensor(view, 32, 32, config);
    view = X::Tensor();
    Require(releases == 1, "preprocessing leaked or retained input allocation");
    Compare(result.pixelValues, Reference(pixels, 32, 32, config));
    std::cout << "garnet-image-offset-view-ownership-passed\n";
}

template<class F> void Rejected(F call, const char* message) {
    bool rejected = false;
    try { call(); } catch (const std::invalid_argument&) { rejected = true; }
    Require(rejected, message);
}
void InvalidInputs(X::Runtime& runtime) {
    QwenVLImagePreprocessConfig config;
    auto source = X::Tensor::Create(runtime.host(), X3_TENSOR_FLOAT32, {32, 32, 3});
    auto transposed = source.View({32, 32, 3}, {12, 32 * 12, 4});
    Rejected([&] { PreprocessRawImageTensor(transposed, 32, 32, config); }, "noncontiguous CPU accepted");
    auto gpu = TensorHelper::CopyToGPU(source);
    auto gpuTranspose = gpu.View({32, 32, 3}, {12, 32 * 12, 4});
    Rejected([&] { PreprocessRawImageTensor(gpuTranspose, 32, 32, config); }, "noncontiguous GPU accepted");
    Rejected([&] { PreprocessRawImageTensor(source, 16, 32, config); }, "wrong element count accepted");
    Rejected([&] { PreprocessRawImageTensor(source, 31, 32, config); }, "unaligned size accepted");
    Rejected([&] { PreprocessRawImageTensor(source, 0, 32, config); }, "zero height accepted");
    config.patchSize = 0;
    Rejected([&] { PreprocessRawImageTensor(source, 32, 32, config); }, "zero patch accepted");
    config.patchSize = 16; config.mergeSize = 3;
    Rejected([&] { PreprocessRawImageTensor(source, 32, 32, config); }, "invalid merge accepted");
    config.mergeSize = 2; config.std[0] = 0;
    Rejected([&] { PreprocessRawImageTensor(source, 32, 32, config); }, "zero std accepted");
    config.std[0] = 0.5f; config.inputScale = std::numeric_limits<float>::quiet_NaN();
    Rejected([&] { PreprocessRawImageTensor(source, 32, 32, config); }, "NaN scale accepted");
    config.inputScale = 255;
    auto wrongType = X::Tensor::Create(runtime.host(), X3_TENSOR_INT32, {32, 32, 3});
    Rejected([&] { PreprocessRawImageTensor(wrongType, 32, 32, config); }, "integer pixels accepted");
    auto symbolic = X::Tensor::Input(runtime.host(), "image", X3_TENSOR_FLOAT32, {32, 32, 3});
    Rejected([&] { PreprocessRawImageTensor(symbolic, 32, 32, config); }, "symbolic image accepted");
    std::cout << "garnet-image-invalid-inputs-passed\n";
}

void Jpeg(X::Runtime& runtime, const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    Require(bool(file), "JPEG fixture missing");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
    Require(!file.bad() && !bytes.empty(), "JPEG fixture read failed");
    OutputMode mode(false);
    auto fromFile = PreprocessJpegFileToTensor(runtime.host(), path.string(), 1024, 4096);
    auto fromBytes = PreprocessJpegBytesToTensor(runtime.host(), bytes.data(), bytes.size(), 1024, 4096);
    Require(fromFile.sourceHeight > 0 && fromFile.sourceWidth > 0, "JPEG source dimensions");
    Require(fromFile.resizedHeight == fromBytes.resizedHeight && fromFile.resizedWidth == fromBytes.resizedWidth, "JPEG dimensions differ");
    auto cpu = TensorHelper::CopyToCPU(X::Tensor(fromFile.pixelValues));
    const float* data = static_cast<const float*>(cpu.Info().data);
    std::vector<float> expected(data, data + cpu.Info().byte_size / sizeof(float));
    Compare(fromBytes.pixelValues, expected);
    for (float value : expected) Require(std::isfinite(value) && std::abs(value) < 2.0f, "invalid JPEG normalized pixel");
    Metadata(fromBytes, fromBytes.resizedHeight / 16, fromBytes.resizedWidth / 16, 2);
    {
        OutputMode cpuMode(true);
        auto cpuResult = PreprocessJpegBytesToTensor(runtime.host(), bytes.data(), bytes.size(), 1024, 4096);
        Require(X::Tensor(cpuResult.pixelValues).Info().device_type == 0, "JPEG CPU output mode");
        Compare(cpuResult.pixelValues, expected);
    }
    std::cout << "garnet-image-jpeg-file-bytes-parity-passed\n";
}

void DecoderDevices(const std::filesystem::path& path, int deviceCount) {
    std::ifstream file(path, std::ios::binary);
    Require(bool(file), "decoder fixture missing");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
    Require(!file.bad() && !bytes.empty(), "decoder fixture read failed");
    for (int device = 0; device < deviceCount; ++device) {
        Check(cudaSetDevice(device));
        cudaStream_t first = nullptr, second = nullptr;
        Check(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking));
        Check(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking));
        Garnet::Image::Cuda::GpuImageRGB8 a, b;
        std::string error;
        Check(Garnet::Image::Cuda::DecodeJpegToDeviceRGB8(bytes.data(), bytes.size(), first, &a, &error));
        // Reuse state on a different stream before explicitly waiting for the first.
        Check(Garnet::Image::Cuda::DecodeJpegToDeviceRGB8(bytes.data(), bytes.size(), second, &b, &error));
        cudaPointerAttributes attributes{};
        Check(cudaPointerGetAttributes(&attributes, b.data));
        Require(attributes.device == device && b.deviceId == device, "JPEG decoder allocation is on wrong device");
        Check(cudaStreamSynchronize(first));
        Check(cudaStreamSynchronize(second));
        const size_t size = static_cast<size_t>(a.pitchBytes) * a.height;
        Require(a.width == b.width && a.height == b.height, "decoder stream dimensions differ");
        std::vector<unsigned char> hostA(size), hostB(size);
        Check(cudaMemcpy(hostA.data(), a.data, size, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(hostB.data(), b.data, size, cudaMemcpyDeviceToHost));
        Require(hostA == hostB, "decoder state reused before completion");
        Check(cudaStreamDestroy(first));
        Check(cudaStreamDestroy(second));
        Check(cudaSetDevice((device + 1) % deviceCount));
        Garnet::Image::Cuda::FreeDecodedImage(&a);
        Garnet::Image::Cuda::FreeDecodedImage(&b);
        Require(!a.data && !b.data, "cross-device image release failed");
        int current = -1;
        Check(cudaGetDevice(&current));
        Require(current == (device + 1) % deviceCount, "image release changed current device");
    }
    Check(cudaSetDevice(0));
    Garnet::Image::Cuda::ShutdownThreadNvJpegDecoder();
    std::cout << "garnet-image-decoder-stream-device-passed: devices=" << deviceCount << '\n';
}

void CUDART_CB DelayProducer(void*) {
    // No CUDA calls are permitted from a stream host callback.
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
}

void CrossThreadReadiness(X::Runtime& runtime) {
    OutputMode mode(false);
    std::vector<float> initial(32 * 32 * 3, 255.0f);
    auto source = TensorHelper::CreateGPU(runtime.host(), X3_TENSOR_FLOAT32, {32, 32, 3}, initial.data());
    cudaStream_t producer = nullptr;
    Check(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    {
        auto write = TensorHelper::AcquireGPU(source, X3_TENSOR_WRITE, producer);
        Check(cudaLaunchHostFunc(producer, DelayProducer, nullptr));
        Check(cudaMemsetAsync(source.Info().data, 0, source.Info().byte_size, producer));
        write.Finish();
    }
    PreprocessResult result;
    std::exception_ptr failure;
    std::thread consumer([&] {
        try {
            Check(cudaSetDevice(0));
            QwenVLImagePreprocessConfig config;
            result = PreprocessRawImageTensor(source, 32, 32, config);
        } catch (...) { failure = std::current_exception(); }
    });
    consumer.join();
    Check(cudaStreamDestroy(producer));
    if (failure) std::rethrow_exception(failure);
    source = X::Tensor();
    Compare(result.pixelValues, std::vector<float>(4 * 1536, -1.0f));
    Metadata(result, 2, 2, 2);

    // A thread returns asynchronously-produced metadata, then exits. Its
    // output storage must carry the completion event to this CPU consumer.
    std::thread metadataProducer([&] {
        try {
            Check(cudaSetDevice(0));
            Check(cudaLaunchHostFunc(cudaStreamPerThread, DelayProducer, nullptr));
            auto metadata = Garnet::Image::QwenVL::BuildVisionMetadataTensors(runtime.host(), 1, 2, 2, 2);
            result.bilinearIndices = metadata.bilinearIndices;
            result.bilinearWeights = metadata.bilinearWeights;
            result.visionPositionIds = metadata.positionIds;
            result.visionCuSeqlens = metadata.cuSeqlens;
        } catch (...) { failure = std::current_exception(); }
    });
    metadataProducer.join();
    if (failure) std::rethrow_exception(failure);
    Metadata(result, 2, 2, 2);
    std::cout << "garnet-image-cross-thread-readiness-passed\n";
}
}

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "usage: garnet_image_tests <JPEG fixture>");
        int devices = 0;
        Check(cudaGetDeviceCount(&devices));
        Require(devices > 0, "CUDA device required for image execution tests");
        Check(cudaSetDevice(0));
        X::Runtime runtime;
        RawSources(runtime);
        Ownership(runtime);
        InvalidInputs(runtime);
        Jpeg(runtime, argv[1]);
        DecoderDevices(argv[1], devices);
        CrossThreadReadiness(runtime);
        Check(cudaDeviceSynchronize());
        std::cout << "garnet-xlang3-image-tests-passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "garnet-image-tests: " << error.what() << '\n';
        return 1;
    }
}
