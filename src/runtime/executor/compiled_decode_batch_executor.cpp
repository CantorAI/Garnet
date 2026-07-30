#include "compiled_decode_batch_executor.h"

#include "garnet_tensor.h"
#include "tensor_helper.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>

namespace Garnet
{
    namespace
    {
        X::Value MakeOwnedDeviceTensor(
            X::TensorDataType dataType,
            const std::vector<int>& dimensions)
        {
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
            for (const int dimension : dimensions) shape.push_back(dimension);
            tensor->SetDataType(dataType);
            tensor->SetShape(shape);
            void* memory = nullptr;
            if (cudaMalloc(&memory, static_cast<size_t>(tensor->GetDataSize())) !=
                    cudaSuccess ||
                TensorHelper::AttachGPUMemory(tensor, memory) !=
                    TensorOpStatus::Success) {
                if (memory) cudaFree(memory);
                return X::Value();
            }
            return X::Value(tensor);
        }

        X::Value MakeBorrowedDeviceTensor(
            X::TensorDataType dataType,
            const std::vector<int>& dimensions,
            void* memory)
        {
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
            for (const int dimension : dimensions) shape.push_back(dimension);
            tensor->SetDataType(dataType);
            tensor->SetShape(shape);
            if (TensorHelper::AttachBorrowedGPUMemory(tensor, memory) !=
                TensorOpStatus::Success) {
                return X::Value();
            }
            return X::Value(tensor);
        }
    }

    bool CompiledDecodeBatchExecutor::Upload(
        X::Value tensorValue,
        const void* source,
        std::size_t bytes)
    {
        if (!tensorValue.IsTensor() || !source) return false;
        X::Tensor tensor(tensorValue);
        return bytes == static_cast<std::size_t>(tensor->GetDataSize()) &&
            cudaMemcpyAsync(
                TensorHelper::GetGPUMemory(tensor),
                source,
                bytes,
                cudaMemcpyHostToDevice,
                cudaStreamPerThread) == cudaSuccess;
    }

    bool CompiledDecodeBatchExecutor::Initialize(
        std::shared_ptr<CompiledModelRuntime> runtime,
        std::shared_ptr<PagedKVPool> kvPool,
        int bucketSize,
        int maxLogicalPages,
        std::string& errorMessage)
    {
        if (!runtime || !kvPool || !kvPool->KeyPages() ||
            bucketSize <= 0 || maxLogicalPages <= 0) {
            errorMessage = "invalid compiled decode executor configuration";
            return false;
        }
        const PagedKVPoolConfig& config = kvPool->Config();
        if (config.elementBytes != 2) {
            errorMessage = "compiled Qwen decode requires a BF16 KV pool";
            return false;
        }
        m_runtime = std::move(runtime);
        m_kvPool = std::move(kvPool);
        m_bucketSize = bucketSize;
        m_maxLogicalPages = maxLogicalPages;

        m_inputIds = MakeOwnedDeviceTensor(
            X::TensorDataType::LONGLONG, {bucketSize, 1});
        m_positionIds = MakeOwnedDeviceTensor(
            X::TensorDataType::LONGLONG, {3, bucketSize, 1});
        const std::vector<int> arenaShape{
            config.numLayers, config.totalPages, config.pageSize,
            config.numKVHeads, config.headDim};
        m_keyPages = MakeBorrowedDeviceTensor(
            X::TensorDataType::BFLOAT16, arenaShape, m_kvPool->KeyPages());
        m_valuePages = MakeBorrowedDeviceTensor(
            X::TensorDataType::BFLOAT16, arenaShape, m_kvPool->ValuePages());
        m_pageTable = MakeOwnedDeviceTensor(
            X::TensorDataType::INT, {bucketSize, maxLogicalPages});
        m_contextLengths = MakeOwnedDeviceTensor(
            X::TensorDataType::INT, {bucketSize});
        m_slotPositions = MakeOwnedDeviceTensor(
            X::TensorDataType::INT, {bucketSize});
        m_activeMask = MakeOwnedDeviceTensor(
            X::TensorDataType::INT, {bucketSize});
        if (!m_inputIds.IsTensor() || !m_positionIds.IsTensor() ||
            !m_keyPages.IsTensor() || !m_valuePages.IsTensor() ||
            !m_pageTable.IsTensor() || !m_contextLengths.IsTensor() ||
            !m_slotPositions.IsTensor() || !m_activeMask.IsTensor()) {
            errorMessage = "failed to allocate persistent decode tensors";
            return false;
        }

        X::V<X::XList> inputs;
        inputs->AddItem(m_inputIds);
        inputs->AddItem(m_positionIds);
        inputs->AddItem(m_keyPages);
        inputs->AddItem(m_valuePages);
        inputs->AddItem(m_pageTable);
        inputs->AddItem(m_contextLengths);
        inputs->AddItem(m_slotPositions);
        inputs->AddItem(m_activeMask);
        m_inputs = X::Value(inputs);
        errorMessage.clear();
        return true;
    }

    bool CompiledDecodeBatchExecutor::Execute(
        const BatchPlan& plan,
        DecodeBatchExecutionResult& result,
        std::string& errorMessage)
    {
        result = {};
        if (!m_runtime || !m_kvPool ||
            plan.stage != RequestStage::Decode ||
            plan.bucketSize != m_bucketSize ||
            plan.requests.size() > static_cast<std::size_t>(m_bucketSize)) {
            errorMessage = "decode plan does not match the executor bucket";
            return false;
        }

        std::vector<long long> inputIds(m_bucketSize, 0);
        std::vector<long long> positions(3 * m_bucketSize, 0);
        std::vector<int> pageTables(
            static_cast<std::size_t>(m_bucketSize) * m_maxLogicalPages, 0);
        std::vector<int> contextLengths(m_bucketSize, 1);
        std::vector<int> slotPositions(m_bucketSize, 0);
        std::vector<int> activeMask(m_bucketSize, 0);
        for (std::size_t row = 0; row < plan.requests.size(); ++row) {
            const ScheduledRequest& request = plan.requests[row];
            if (request.contextLength != request.slotPosition + 1 ||
                request.pageTable.empty() ||
                request.pageTable.size() >
                    static_cast<std::size_t>(m_maxLogicalPages)) {
                errorMessage = "invalid scheduled decode metadata";
                return false;
            }
            inputIds[row] = request.tokenId;
            const long long position =
                static_cast<long long>(request.slotPosition) +
                request.positionDelta;
            positions[row] = position;
            positions[m_bucketSize + row] = position;
            positions[2 * m_bucketSize + row] = position;
            std::copy(
                request.pageTable.begin(),
                request.pageTable.end(),
                pageTables.begin() +
                    static_cast<std::ptrdiff_t>(row * m_maxLogicalPages));
            contextLengths[row] = request.contextLength;
            slotPositions[row] = request.slotPosition;
            activeMask[row] = 1;
            result.requestIds.push_back(request.id);
        }

        if (!Upload(m_inputIds, inputIds.data(),
                    inputIds.size() * sizeof(long long)) ||
            !Upload(m_positionIds, positions.data(),
                    positions.size() * sizeof(long long)) ||
            !Upload(m_pageTable, pageTables.data(),
                    pageTables.size() * sizeof(int)) ||
            !Upload(m_contextLengths, contextLengths.data(),
                    contextLengths.size() * sizeof(int)) ||
            !Upload(m_slotPositions, slotPositions.data(),
                    slotPositions.size() * sizeof(int)) ||
            !Upload(m_activeMask, activeMask.data(),
                    activeMask.size() * sizeof(int))) {
            errorMessage = "failed to upload scheduled decode metadata";
            return false;
        }

        X::Dict request;
        request->Set("inputs", m_inputs);
        request->Set("sample", X::Value("greedy_batch"));
        request->Set("reuse_output", X::Value(1));
        const auto start = std::chrono::steady_clock::now();
        X::Value responseValue = m_runtime->Forward(X::Value(request));
        result.executionMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        if (!responseValue.IsDict()) {
            errorMessage = "compiled decode returned no response";
            return false;
        }
        X::Dict response(responseValue);
        if (response["status"].ToString() != "ok" ||
            !response["token_ids"].IsList()) {
            errorMessage = response["error_message"].ToString();
            if (errorMessage.empty()) {
                errorMessage = response["error_code"].ToString();
            }
            return false;
        }
        X::List tokens(response["token_ids"]);
        if (tokens->Size() < static_cast<long long>(plan.requests.size())) {
            errorMessage = "compiled decode returned too few sampled tokens";
            return false;
        }
        for (std::size_t row = 0; row < plan.requests.size(); ++row) {
            result.tokenIds.push_back(
                tokens->Get(static_cast<long long>(row)).ToLongLong());
        }
        errorMessage.clear();
        return true;
    }
}
