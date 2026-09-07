#include "graph_capture.h"
#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "../../src/model/compiled_graph_capture.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

static void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class CaptureProbe {
public:
    X::Value Capture(const X::Value& graph) {
        Garnet::TensorGraphCapture capture(graph);
        auto result = X::Value::List(Host());
        for (const auto& op : capture.Operations()) {
            auto record = X::Value::Dict(Host());
            Require(record.SetItem("id", op.id), "set id");
            Require(record.SetItem("name", op.name), "set name");
            Require(record.SetItem("provider", op.provider), "set provider");
            Require(record.SetItem("operand_count", op.operandCount), "set arity");
            Require(record.SetItem("ordered", (op.flags & X3_TENSOR_ORDERED) != 0), "set order");
            Require(record.SetItem("attributes", op.attributes), "set attributes");
            Require(record.SetItem("regions", op.regions), "set regions");
            auto inputs = X::Value::List(Host());
            for (const auto& input : op.inputs) Require(inputs.Append(input), "append input");
            Require(record.SetItem("inputs", inputs), "set inputs");
            Require(result.Append(record), "append operation");
        }
        return result;
    }
    X::Value Partitions(const X::Value& graph) {
        Garnet::TensorGraphCapture capture(graph);
        Garnet::FusionPartitionOptions options;
        options.maxAtomicRegionsPerPartition = 1;
        std::string error;
        if (!Garnet::CaptureFusionGraph(capture, options, error)) throw X::Error(error);
        auto result = X::Value::Dict(Host());
        auto operations = X::Value::List(Host());
        for (const auto& operation : Garnet::GetCapturedTensorOperations()) {
            auto record = X::Value::Dict(Host());
            record.SetItem("partition", operation.candidatePartition);
            record.SetItem("region", operation.regionId);
            operations.Append(record);
        }
        auto regions = X::Value::List(Host());
        for (const auto& region : Garnet::GetCapturedFusionRegions()) {
            auto record = X::Value::Dict(Host());
            record.SetItem("parent", region.parentId);
            record.SetItem("operations", region.operationCount);
            record.SetItem("inclusive_operations", region.inclusiveOperationCount);
            record.SetItem("input_count", region.inputTensorIds.size());
            record.SetItem("output_count", region.outputTensorIds.size());
            regions.Append(record);
        }
        result.SetItem("operations", operations);
        result.SetItem("regions", regions);
        return result;
    }
    X::Value Outputs(const X::Value& graph) {
        return Garnet::TensorGraphCapture(graph).Outputs();
    }
    BEGIN_PACKAGE(CaptureProbe)
        APISET().AddClass<0, Garnet::GarnetTensor>("tensor");
        APISET().AddFunc<1>("capture", &CaptureProbe::Capture);
        APISET().AddFunc<1>("partitions", &CaptureProbe::Partitions);
        APISET().AddFunc<1>("outputs", &CaptureProbe::Outputs);
    END_PACKAGE
};

static void Storage(X::Runtime& runtime) {
    Require(!X::Tensor::IsTensor(runtime.Dict()) && !X::Tensor::IsTensor(X::Value(1)) &&
        !X::Tensor::IsTensor(X::Value()), "non-tensor type queries");
    int failedWrapCleanups = 0;
    float scalar = 1;
    int64_t invalidShape[] = {-1};
    X3TensorInfo invalid{};
    invalid.size = sizeof(invalid); invalid.dtype = X3_TENSOR_FLOAT32;
    invalid.rank = 1; invalid.shape = invalidShape;
    invalid.data = &scalar; invalid.byte_size = sizeof(scalar);
    bool rejected = false;
    try {
        X::Tensor::Wrap(runtime.host(), invalid, &failedWrapCleanups,
            [](void* pointer) { ++*static_cast<int*>(pointer); });
    } catch (const std::exception&) { rejected = true; }
    Require(rejected && failedWrapCleanups == 0,
        "failed wrap must leave ownership with caller");
    for (auto dtype : {X3_TENSOR_FLOAT16, X3_TENSOR_BFLOAT16}) {
        int cleanups = 0;
        uint16_t bits[] = {0x3c00, 0x4000, 0x4200, 0x4400};
        int64_t shape[] = {2, 2};
        X3TensorInfo info{};
        info.size = sizeof(info); info.dtype = dtype;
        info.rank = 2; info.shape = shape;
        info.data = bits; info.byte_size = sizeof(bits); info.readonly = 1;
        std::unique_ptr<Garnet::TensorGraphCapture> captured;
        {
            auto tensor = X::Tensor::Wrap(runtime.host(), info, &cleanups,
                [](void* pointer) { ++*static_cast<int*>(pointer); });
            Require(X::Tensor::IsTensor(tensor), "native tensor type query");
            Require(tensor.Info().data == bits, "wrap must not copy tensor storage");
            auto view = tensor.View({2}, {4}, 2);
            Require(view.Info().data == bits + 1 && view.Info().strides[0] == 4,
                "low precision strided view");
            auto unary = Garnet::RegisterTensorOperator(runtime.host(), 1);
            auto expr = view * unary("rms_norm");
            auto graph = X::TensorGraph(expr);
            Require(X::TensorGraph::IsGraph(graph) && !X::Tensor::IsTensor(graph) &&
                !X::TensorGraph::IsGraph(tensor), "graph and tensor types are distinct");
            captured = std::make_unique<Garnet::TensorGraphCapture>(graph);
            Require(captured->Operations().size() == 2, "capture constant and operation");
            Require(X::Tensor(captured->Operations()[0].output).Info().data == bits + 1,
                "graph capture must not copy tensor payload");
            auto input = X::Tensor::Input(runtime.host(), "weights", dtype, {2, 2});
            Require(input.Info().symbolic && input.Info().dtype == dtype,
                "low precision symbolic weights");
            auto cpuGraph = X::TensorGraph(tensor + tensor);
            X3Value result = x3_value_invalid();
            auto bindings = runtime.Dict();
            Require(x3_tensor_graph_run(runtime.get(), cpuGraph.raw(), bindings.raw(), &result)
                == X3_STATUS_ERROR, "unsupported low precision CPU arithmetic must fail");
            x3_value_release(result);
        }
        Require(cleanups == 0, "capture retains storage after source release");
        captured.reset();
        Require(cleanups == 1, "storage released exactly once");
        auto copied = X::Tensor::Create(runtime.host(), dtype, {2, 2}, bits, sizeof(bits));
        Require(copied.Info().byte_size == sizeof(bits) &&
            std::memcmp(copied.Info().data, bits, sizeof(bits)) == 0, "preserve packed weight bits");
    }
}

static void DeviceStorage(X::Runtime& runtime) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        std::cout << "garnet-cuda-storage-skipped: no CUDA device\n";
        return;
    }
    float data[] = {1, 2, 3, 4, 5, 6};
    auto gpu = Garnet::TensorHelper::CreateGPU(runtime.host(), X3_TENSOR_FLOAT32,
        {2, 3}, data);
    auto gpuView = gpu.View({3, 2}, {4, 12});
    Garnet::TensorHelper::ReleaseGPUMemory(gpu);
    auto cpu = Garnet::TensorHelper::CopyToCPU(gpuView);
    Require(cpu.Info().shape[0] == 3 && cpu.Info().strides[1] == 12,
        "device transfer preserves strided layout");
    Require(std::memcmp(cpu.Info().data, data, sizeof(data)) == 0,
        "retained GPU view survives parent release");
    auto back = Garnet::TensorHelper::CopyToGPU(cpu);
    auto roundtrip = Garnet::TensorHelper::CopyToCPU(back);
    Require(std::memcmp(roundtrip.Info().data, data, sizeof(data)) == 0,
        "CPU to GPU roundtrip");
    int releases = 0;
    void* memory = nullptr;
    Require(cudaMalloc(&memory, sizeof(data)) == cudaSuccess, "allocate borrowed buffer");
    std::shared_ptr<void> owner(memory, [&releases](void* pointer) {
        cudaFree(pointer); ++releases;
    });
    auto descriptor = gpuView.Info();
    descriptor.data = memory;
    descriptor.byte_size = sizeof(data);
    {
        auto borrowed = Garnet::TensorHelper::WrapBorrowedGPU(runtime.host(), descriptor, owner);
        owner.reset();
        auto retained = borrowed.View({2}, {12});
        borrowed = X::Tensor();
        Require(releases == 0, "borrowed buffer retained by view");
    }
    Require(releases == 1, "borrowed buffer released exactly once");
    auto empty = Garnet::TensorHelper::CreateGPU(runtime.host(), X3_TENSOR_BFLOAT16, {0, 4});
    Require(empty.Info().byte_size == 0, "empty GPU tensor");
    void* rejectedAllocation = nullptr;
    Require(cudaMalloc(&rejectedAllocation, sizeof(float)) == cudaSuccess,
        "allocate rejected wrap buffer");
    auto invalid = empty.Info();
    int64_t invalidShape[] = {-1};
    invalid.rank = 1; invalid.shape = invalidShape; invalid.strides = nullptr;
    invalid.data = rejectedAllocation; invalid.byte_size = sizeof(float);
    bool rejected = false;
    try {
        Garnet::TensorHelper::WrapGPU(runtime.host(), invalid, rejectedAllocation,
            empty.Info().device_id);
    } catch (const std::exception&) { rejected = true; }
    cudaPointerAttributes attributes{};
    const auto query = cudaPointerGetAttributes(&attributes, rejectedAllocation);
    const auto released = cudaFree(rejectedAllocation);
    Require(rejected && query == cudaSuccess && released == cudaSuccess,
        "failed CUDA wrap leaves allocation owned by caller");
    for (auto dtype : {X3_TENSOR_FLOAT8_E4M3FN, X3_TENSOR_FLOAT8_E4M3FNUZ,
        X3_TENSOR_FLOAT8_E5M2, X3_TENSOR_FLOAT8_E5M2FNUZ}) {
        const uint8_t packed[] = {0, 1, 0x7f, 0xff};
        auto device = Garnet::TensorHelper::CreateGPU(runtime.host(), dtype, {4}, packed);
        auto host = Garnet::TensorHelper::CopyToCPU(device);
        Require(host.Info().dtype == dtype && host.Info().byte_size == sizeof(packed) &&
            std::memcmp(host.Info().data, packed, sizeof(packed)) == 0,
            "packed FP8 device roundtrip");
    }
    std::cout << "garnet-cuda-storage-passed\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected graph test script path");
        CaptureProbe probe;
        X::Runtime runtime;
        auto package = runtime.RegisterPackage("garnet_capture_test", probe);
        Storage(runtime);
        DeviceStorage(runtime);
        X3Value result = x3_value_invalid();
        const auto status = x3_runtime_eval_file(runtime.get(), argv[1], &result);
        x3_value_release(result);
        runtime.check(status);
        std::cout << "garnet-xlang3-tensor-capture-passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
