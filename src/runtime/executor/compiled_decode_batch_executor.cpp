#include "compiled_decode_batch_executor.h"

#include "garnet_tensor.h"
#include "tensor_helper.h"
#include "lowering_context.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>

namespace Garnet
{
    namespace
    {
        X::Value MakeOwnedDeviceTensor(
            X3PackageHost* host,
            X3TensorDType dataType,
            const std::vector<int>& dimensions)
        {
            return TensorHelper::CreateGPU(host, dataType,
                std::vector<int64_t>(dimensions.begin(), dimensions.end()));
        }

        X::Value MakeBorrowedDeviceTensor(
            X3PackageHost* host,
            X3TensorDType dataType,
            const std::vector<int>& dimensions,
            void* memory,
            std::shared_ptr<PagedKVPool> owner)
        {
            std::vector<int64_t> shape(dimensions.begin(), dimensions.end());
            uint64_t bytes = TensorHelper::ItemSize(dataType);
            for (const int dimension : dimensions) {
                if (dimension <= 0 || bytes > UINT64_MAX / static_cast<uint64_t>(dimension))
                    throw std::invalid_argument("invalid paged KV tensor shape");
                bytes *= dimension;
            }
            X3TensorInfo info{};
            info.size = sizeof(info);
            info.dtype = dataType;
            info.shape = shape.data();
            info.rank = static_cast<uint32_t>(shape.size());
            info.data = memory;
            info.byte_size = bytes;
            info.device_type = TensorHelper::CudaDevice;
            info.device_id = owner->Config().deviceId;
            return TensorHelper::WrapBorrowedGPU(host, info, std::move(owner));
        }
    }

    bool CompiledDecodeBatchExecutor::Upload(
        X::Value tensorValue,
        const void* source,
        std::size_t bytes)
    {
        if (!IsTensor(tensorValue) || !source) return false;
        X::Tensor tensor(tensorValue);
        return bytes == static_cast<std::size_t>(tensor.Info().byte_size) &&
            cudaMemcpyAsync(
                TensorHelper::GetGPUMemory(tensor),
                source,
                bytes,
                cudaMemcpyHostToDevice,
                cudaStreamPerThread) == cudaSuccess;
    }

    bool CompiledDecodeBatchExecutor::Initialize(
        X3PackageHost* host,
        std::shared_ptr<CompiledModelRuntime> runtime,
        std::shared_ptr<PagedKVPool> kvPool,
        int bucketSize,
        int maxLogicalPages,
        std::string& errorMessage,
        int positionComponents)
    try
    {
        if (!host || !host->runtime || !runtime || !kvPool || !kvPool->KeyPages() ||
            bucketSize <= 0 || maxLogicalPages <= 0 ||
            (positionComponents != 1 && positionComponents != 3)) {
            errorMessage = "invalid compiled decode executor configuration";
            return false;
        }
        const PagedKVPoolConfig& config = kvPool->Config();
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess || device != config.deviceId) {
            errorMessage = "decode executor CUDA device does not match its KV pool";
            return false;
        }
        if (config.elementBytes != 2) {
            errorMessage = "compiled Qwen decode requires a BF16 KV pool";
            return false;
        }
        m_runtime = std::move(runtime);
        m_kvPool = std::move(kvPool);
        m_bucketSize = bucketSize;
        m_maxLogicalPages = maxLogicalPages;
        m_positionComponents = positionComponents;

        m_inputIds = MakeOwnedDeviceTensor(host,
            X3_TENSOR_INT64, {bucketSize, 1});
        m_positionIds = MakeOwnedDeviceTensor(host,
            X3_TENSOR_INT64,
            {positionComponents, bucketSize, 1});
        const std::vector<int> arenaShape{
            config.numLayers, config.totalPages, config.pageSize,
            config.numKVHeads, config.headDim};
        m_keyPages = MakeBorrowedDeviceTensor(host,
            X3_TENSOR_BFLOAT16, arenaShape, m_kvPool->KeyPages(), m_kvPool);
        m_valuePages = MakeBorrowedDeviceTensor(host,
            X3_TENSOR_BFLOAT16, arenaShape, m_kvPool->ValuePages(), m_kvPool);
        m_pageTable = MakeOwnedDeviceTensor(host,
            X3_TENSOR_INT32, {bucketSize, maxLogicalPages});
        m_contextLengths = MakeOwnedDeviceTensor(host,
            X3_TENSOR_INT32, {bucketSize});
        m_slotPositions = MakeOwnedDeviceTensor(host,
            X3_TENSOR_INT32, {bucketSize});
        m_activeMask = MakeOwnedDeviceTensor(host,
            X3_TENSOR_INT32, {bucketSize});
        if (!IsTensor(m_inputIds) || !IsTensor(m_positionIds) ||
            !IsTensor(m_keyPages) || !IsTensor(m_valuePages) ||
            !IsTensor(m_pageTable) || !IsTensor(m_contextLengths) ||
            !IsTensor(m_slotPositions) || !IsTensor(m_activeMask)) {
            errorMessage = "failed to allocate persistent decode tensors";
            return false;
        }

        m_inputs = X::Value::List(host);
        for (const auto* value : {&m_inputIds, &m_positionIds, &m_keyPages,
                &m_valuePages, &m_pageTable, &m_contextLengths, &m_slotPositions,
                &m_activeMask}) {
            if (!m_inputs.Append(*value)) {
                errorMessage = host->runtime_last_error(host->runtime);
                return false;
            }
        }
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& error)
    {
        errorMessage = error.what();
        return false;
    }

    bool CompiledDecodeBatchExecutor::Execute(
        const BatchPlan& plan,
        DecodeBatchExecutionResult& result,
        std::string& errorMessage)
    try
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
        std::vector<long long> positions(
            static_cast<std::size_t>(m_positionComponents) * m_bucketSize, 0);
        std::vector<int> pageTables(
            static_cast<std::size_t>(m_bucketSize) * m_maxLogicalPages, 0);
        std::vector<int> contextLengths(m_bucketSize, 1);
        std::vector<int> slotPositions(m_bucketSize, 0);
        std::vector<int> activeMask(m_bucketSize, 0);
        struct UploadLifetime {
            bool active = false;
            ~UploadLifetime() {
                if (active) cudaStreamSynchronize(cudaStreamPerThread);
            }
        } uploads;
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess || device != m_kvPool->Config().deviceId) {
            errorMessage = "decode execution CUDA device does not match its KV pool";
            return false;
        }
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
            for (int component = 0;
                 component < m_positionComponents; ++component) {
                positions[
                    static_cast<std::size_t>(component * m_bucketSize) + row] =
                    position;
            }
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

        uploads.active = true;
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

        auto request = X::Value::Dict(m_inputs.host());
        if (!request.SetItem("inputs", m_inputs) ||
            !request.SetItem("sample", X::Value::String(m_inputs.host(), "greedy_batch")) ||
            !request.SetItem("reuse_output", X::Value(1))) {
            errorMessage = "failed to create compiled decode request";
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        X::Value responseValue = m_runtime->Forward(X::Value(request));
        result.executionMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        if (!responseValue.IsDict()) {
            errorMessage = "compiled decode returned no response";
            return false;
        }
        X::Value response(responseValue);
        if (response["status"].ToString() != "ok" ||
            !response["token_ids"].IsList()) {
            auto errorField = [&response](const char* name) {
                for (uint64_t i = 0; i < response.Size(); ++i) {
                    X::Value key, value;
                    if (!response.DictEntry(i, key, value)) break;
                    if (key.ToString() == name) return value.ToString();
                }
                return std::string();
            };
            errorMessage = errorField("error_message");
            if (errorMessage.empty()) {
                errorMessage = errorField("error_code");
            }
            return false;
        }
        X::Value tokens(response["token_ids"]);
        if (tokens.Size() < static_cast<long long>(plan.requests.size())) {
            errorMessage = "compiled decode returned too few sampled tokens";
            return false;
        }
        for (std::size_t row = 0; row < plan.requests.size(); ++row) {
            result.tokenIds.push_back(
                tokens.Get(static_cast<long long>(row)).ToLongLong());
        }
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& error)
    {
        errorMessage = error.what();
        return false;
    }
}
