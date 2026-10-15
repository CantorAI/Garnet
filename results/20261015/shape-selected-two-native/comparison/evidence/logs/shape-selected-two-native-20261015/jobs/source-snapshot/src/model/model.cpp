// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "model.h"
#include "../entry/native_values.h"
#include "trt_builder.h"
#include "../tensor/garnet_tensor.h"
#include "../tensor/tensor_helper.h"
#include "../tokenizer/qwen_tokenizer.h"
#include "../cuda/cuda_lib.h"
#include <cstring>
#include <iostream>

extern "C" int GarnetCreateDevicePagedKVFP32(
    int physicalPageCount,
    int pageSize,
    int logicalPageCount,
    int qHeads,
    int kvHeads,
    int headDim,
    const int* pageTable,
    long long* outputHandle,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" int GarnetDestroyDevicePagedKVFP32(
    long long handle,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" int GarnetDevicePagedKVWriteDeviceFP32(
    long long handle,
    const float* deviceQKV,
    int tokenCount,
    int startPosition,
    char* errorMessage,
    int errorMessageCapacity);

extern "C" int GarnetDevicePagedKVAttentionDeviceFP32(
    long long handle,
    const float* deviceQ,
    float* deviceOutput,
    int sequenceLength,
    char* errorMessage,
    int errorMessageCapacity);

namespace Garnet
{
    namespace
    {
        X::Value MakeShapeList(X::Tensor& tensor)
        {
            auto dims = X::Value::List(tensor.host());
            int dimCount = tensor.Info().rank;
            for (int i = 0; i < dimCount; ++i) {
                X::Value dimValue(static_cast<long long>(tensor.Info().shape[i]));
                if (!dims.Append(dimValue)) throw X::Error("cannot append tensor dimension");
            }
            return dims;
        }

        X::Value TensorSummary(X3PackageHost* host, X::Value value)
        {
            auto summary = X::Value::Dict(host);
            bool isTensor = X::Tensor::IsTensor(value);
            summary.SetItem("is_tensor", X::Value(isTensor));
            if (!isTensor) {
                summary.SetItem("gpu", X::Value(false));
                return summary;
            }

            X::Tensor tensor(value);
            void* gpuMemory = TensorHelper::GetGPUMemory(tensor);
            summary.SetItem("gpu", X::Value(gpuMemory != nullptr));
            summary.SetItem("shape", MakeShapeList(tensor));
            summary.SetItem("dtype", X::Value(static_cast<int>(tensor.Info().dtype)));
            summary.SetItem("bytes", X::Value(static_cast<long long>(tensor.Info().byte_size)));
            summary.SetItem("count", X::Value(static_cast<long long>(TensorCount(tensor))));
            return summary;
        }

        bool HasGPUTensor(X::Value value)
        {
            if (!X::Tensor::IsTensor(value)) {
                return false;
            }
            X::Tensor tensor(value);
            return TensorHelper::GetGPUMemory(tensor) != nullptr;
        }

        X::Value GetObjectField(X::Value value, const char* name)
        {
            if (!value.IsValid()) {
                return X::Value();
            }
            return FindField(value, name);
        }

        X::Value MakeInt64Tensor1D(X3PackageHost* host, const std::vector<long long>& values, bool ensureGpu)
        {
            const std::vector<int64_t> shape{static_cast<int64_t>(values.size())};
            return ensureGpu ? TensorHelper::CreateGPU(host, X3_TENSOR_INT64, shape, values.data()) :
                X::Tensor::Create(host, X3_TENSOR_INT64, shape, values.data(), values.size() * sizeof(long long));
        }

        int GetIntArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t pos, const char* name, int defaultValue)
        {
            for (auto& item : kwParams) {
                if (std::string(item.first) == name) {
                    return CheckedInt(item.second, "argument");
                }
            }
            if (params.size() > pos) {
                return CheckedInt(params[pos], "argument");
            }
            return defaultValue;
        }

        long long GetLongLongArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t pos, const char* name, long long defaultValue)
        {
            for (auto& item : kwParams) {
                if (std::string(item.first) == name) {
                    return item.second.ToLongLong();
                }
            }
            if (params.size() > pos) {
                return params[pos].ToLongLong();
            }
            return defaultValue;
        }

        bool GetBoolArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t pos, const char* name, bool defaultValue)
        {
            for (auto& item : kwParams) {
                if (std::string(item.first) == name) {
                    return (item.second.ToLongLong() != 0);
                }
            }
            if (params.size() > pos) {
                return params[pos].ToLongLong() != 0;
            }
            return defaultValue;
        }

        X::Value GetValueArg(const X::ARGS& params, const X::KWARGS& kwParams, size_t pos, const char* name)
        {
            for (auto& item : kwParams) {
                if (std::string(item.first) == name) {
                    return item.second;
                }
            }
            if (params.size() > pos) {
                return params[pos];
            }
            return X::Value();
        }

        void SetObjectField(X::Value value, const char* name, X::Value field)
        {
            if (!value.IsObject()) {
                return;
            }
            if (!(value.IsDict() ? value.SetItem(name, field) : value.SetAttr(name, field)))
                throw X::Error("cannot update request field");
        }

        X::Value CreateDeviceKVCacheValue(
            X3PackageHost* host,
            int maxTokens,
            int pageSize,
            int qHeads,
            int kvHeads,
            int headDim,
            int physicalPageCount)
        {
            if (maxTokens <= 0 || pageSize <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || (qHeads % kvHeads) != 0) {
                std::cout << "[Model] create_device_kv_cache invalid arguments." << std::endl;
                return X::Value();
            }

            int logicalPageCount = (maxTokens + pageSize - 1) / pageSize;
            if (physicalPageCount <= 0) {
                physicalPageCount = logicalPageCount;
            }
            if (physicalPageCount < logicalPageCount) {
                std::cout << "[Model] create_device_kv_cache physical_pages must cover logical pages for this MVP." << std::endl;
                return X::Value();
            }

            std::vector<int> pageTable(static_cast<size_t>(logicalPageCount));
            for (int i = 0; i < logicalPageCount; ++i) {
                pageTable[static_cast<size_t>(i)] = i;
            }

            long long handle = 0;
            char error[1024] = {};
            int rc = GarnetCreateDevicePagedKVFP32(
                physicalPageCount,
                pageSize,
                logicalPageCount,
                qHeads,
                kvHeads,
                headDim,
                pageTable.data(),
                &handle,
                error,
                static_cast<int>(sizeof(error)));
            if (rc != 0) {
                std::cout << "[Model] create_device_kv_cache failed: " << error << std::endl;
                return X::Value();
            }

            long long bytesPerArena = static_cast<long long>(physicalPageCount)
                * static_cast<long long>(pageSize)
                * static_cast<long long>(kvHeads)
                * static_cast<long long>(headDim)
                * static_cast<long long>(sizeof(float));

            auto result = X::Value::Dict(host);
            result.SetItem("handle", X::Value(handle));
            result.SetItem("max_tokens", X::Value(maxTokens));
            result.SetItem("logical_length", X::Value(0));
            result.SetItem("page_size", X::Value(pageSize));
            result.SetItem("logical_pages", X::Value(logicalPageCount));
            result.SetItem("physical_pages", X::Value(physicalPageCount));
            result.SetItem("q_heads", X::Value(qHeads));
            result.SetItem("kv_heads", X::Value(kvHeads));
            result.SetItem("head_dim", X::Value(headDim));
            result.SetItem("key_bytes", X::Value(bytesPerArena));
            result.SetItem("value_bytes", X::Value(bytesPerArena));
            result.SetItem("status", "ok");
            return result;
        }

        long long GetHandleFromValue(X::Value value)
        {
            if (!value.IsValid()) {
                return 0;
            }
            if (value.IsObject()) {
                X::Value handle = FindField(value, "handle");
                if (handle.IsValid()) {
                    return handle.ToLongLong();
                }
                handle = FindField(value, "kv_handle");
                if (handle.IsValid()) {
                    return handle.ToLongLong();
                }
            }
            return value.ToLongLong();
        }

        int GetKVLogicalLengthFromValue(X::Value value)
        {
            if (!value.IsValid() || !value.IsObject()) {
                return 0;
            }
            X::Value logicalLength = FindField(value, "logical_length");
            if (logicalLength.IsValid()) {
                return static_cast<int>(logicalLength.ToLongLong());
            }
            logicalLength = FindField(value, "kv_logical_length");
            if (logicalLength.IsValid()) {
                return static_cast<int>(logicalLength.ToLongLong());
            }
            X::Value kvCache = FindField(value, "kv_cache");
            if (kvCache.IsValid()) {
                return GetKVLogicalLengthFromValue(kvCache);
            }
            return 0;
        }

        void SetKVLogicalLengthOnValue(X::Value value, int logicalLength)
        {
            if (!value.IsValid() || !value.IsObject()) {
                return;
            }
            SetObjectField(value, "logical_length", X::Value(logicalLength));
            SetObjectField(value, "kv_logical_length", X::Value(logicalLength));
            X::Value kvCache = FindField(value, "kv_cache");
            if (kvCache.IsValid()) {
                SetKVLogicalLengthOnValue(kvCache, logicalLength);
            }
        }

        X::Value RunDeviceKVAttentionTensor(
            long long handle,
            X::Tensor q,
            int sequenceLength,
            int qWidth,
            const char* caller)
        {
            ValidateDenseTensor(q);
            if (handle <= 0 || sequenceLength <= 0 || qWidth <= 0 ||
                q.Info().dtype != X3_TENSOR_FLOAT32 || q.Info().rank != 2 || q.Info().shape[0] < 1 ||
                q.Info().shape[1] < qWidth) {
                std::cout << "[Model] " << caller << " invalid arguments." << std::endl;
                return X::Value();
            }

            if (TensorHelper::EnsureGPUMemory(q) != TensorOpStatus::Success) {
                std::cout << "[Model] " << caller << " failed to ensure q GPU memory." << std::endl;
                return X::Value();
            }
            auto qUse = q.Acquire();
            auto* qBase = static_cast<const float*>(TensorHelper::GetGPUMemory(q));
            if (!qBase) {
                std::cout << "[Model] " << caller << " q tensor has no GPU memory." << std::endl;
                return X::Value();
            }

            int qStride = static_cast<int>(q.Info().shape[1]);
            const float* deviceQ = qBase + static_cast<size_t>(q.Info().shape[0] - 1) * static_cast<size_t>(qStride);
            size_t outputBytes = static_cast<size_t>(qWidth) * sizeof(float);
            float* deviceOutput = nullptr;
            cudaError_t err = cudaMalloc(&deviceOutput, outputBytes);
            if (err != cudaSuccess) {
                std::cout << "[Model] " << caller << " failed to allocate output: " << cudaGetErrorString(err) << std::endl;
                return X::Value();
            }

            char error[1024] = {};
            int rc = GarnetDevicePagedKVAttentionDeviceFP32(
                handle,
                deviceQ,
                deviceOutput,
                sequenceLength,
                error,
                static_cast<int>(sizeof(error)));
            if (rc != 0) {
                cudaFree(deviceOutput);
                std::cout << "[Model] " << caller << " failed: " << error << std::endl;
                return X::Value();
            }

            int device = 0;
            cudaGetDevice(&device);
            const int64_t shape[]{1, qWidth};
            const int64_t strides[]{qWidth * static_cast<int64_t>(sizeof(float)), sizeof(float)};
            X3TensorInfo info{};
            info.size = sizeof(info); info.dtype = X3_TENSOR_FLOAT32; info.rank = 2;
            info.shape = shape; info.strides = strides; info.data = deviceOutput;
            info.byte_size = outputBytes; info.device_type = TensorHelper::CudaDevice; info.device_id = device;
            try { return TensorHelper::WrapGPU(q.host(), info, deviceOutput, device); }
            catch (...) { cudaFree(deviceOutput); throw; }
        }
    }

    X::Value Model::Access(X::Value index)
    {
        auto value = mModel.GetItem(index);
        if (!value.IsValid()) throw X::Error(Host()->runtime_last_error(Host()->runtime));
        return value;
    }
    X::Value Model::Tokenizer(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1) {
            std::cout << "[Model] tokenizer requires input text." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        bool addSpecialTokens = false;
        for (auto& item : kwParams) {
            if (std::string(item.first) == "add_special_tokens") {
                addSpecialTokens = (item.second.ToLongLong() != 0);
            }
        }

        std::string error;
        auto tokenizer = Tokenization::GetCachedQwenTokenizer(mModelPath, &error);
        if (!tokenizer) {
            std::cout << "[Model] native tokenizer load failed: " << error << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::vector<int64_t> ids64 = tokenizer->Encode(params[0].ToString(), addSpecialTokens);
        std::vector<long long> inputIds;
        std::vector<long long> attentionMask;
        inputIds.reserve(ids64.size());
        attentionMask.reserve(ids64.size());
        for (int64_t id : ids64) {
            inputIds.push_back(static_cast<long long>(id));
            attentionMask.push_back(1LL);
        }

        X::Value tensorIds = MakeInt64Tensor1D(Host(), inputIds, true);
        X::Value tensorAttentionMask = MakeInt64Tensor1D(Host(), attentionMask, true);
        if (!X::Tensor::IsTensor(tensorIds) || !X::Tensor::IsTensor(tensorAttentionMask)) {
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        auto dictInputs = X::Value::Dict(Host());
        dictInputs.SetItem("input_ids", tensorIds);
        dictInputs.SetItem("attention_mask", tensorAttentionMask);
        dictInputs.SetItem("token_count", X::Value(static_cast<int>(inputIds.size())));
        dictInputs.SetItem("tokenizer", X::Value::String(Host(), "native_qwen"));
        retValue = NativeValue(Host(), dictInputs);
        return retValue;
    }

    X::Value Model::Detokenizer(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1) {
            std::cout << "[Model] detokenizer requires token ids." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        bool skipSpecialTokens = true;
        for (auto& item : kwParams) {
            if (std::string(item.first) == "skip_special_tokens") {
                skipSpecialTokens = (item.second.ToLongLong() != 0);
            }
        }

        std::vector<int64_t> ids;
        X::Value tokenValue = params[0];
        if (X::Tensor::IsTensor(tokenValue)) {
            X::Tensor tokenTensor(tokenValue);
        ValidateDenseTensor(tokenTensor);
            if (tokenTensor.Info().dtype != X3_TENSOR_INT64) {
                std::cout << "[Model] detokenizer tensor must be INT64/LONGLONG." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            if (TensorHelper::GetGPUMemory(tokenTensor) != nullptr &&
                TensorHelper::CopyResultFromGPU(tokenTensor) != TensorOpStatus::Success) {
                std::cout << "[Model] detokenizer failed to copy final token ids from GPU." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            auto tokenUse = tokenTensor.Acquire();
            auto* data = reinterpret_cast<long long*>(tokenTensor.Info().data);
            long long count = TensorCount(tokenTensor);
            ids.reserve(static_cast<size_t>(count));
            for (long long i = 0; i < count; ++i) {
                ids.push_back(static_cast<int64_t>(data[i]));
            }
        }
        else if (tokenValue.IsList()) {
            X::Value list(tokenValue);
            long long count = list.Size();
            ids.reserve(static_cast<size_t>(count));
            for (long long i = 0; i < count; ++i) {
                ids.push_back(static_cast<int64_t>(list.Get(i).ToLongLong()));
            }
        }
        else {
            ids.push_back(static_cast<int64_t>(tokenValue.ToLongLong()));
        }

        std::string error;
        auto tokenizer = Tokenization::GetCachedQwenTokenizer(mModelPath, &error);
        if (!tokenizer) {
            std::cout << "[Model] native tokenizer load failed: " << error << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::string text = tokenizer->Decode(ids, skipSpecialTokens);
        auto result = X::Value::Dict(Host());
        result.SetItem("text", X::Value::String(Host(), text));
        result.SetItem("token_count", X::Value(static_cast<int>(ids.size())));
        result.SetItem("tokenizer", X::Value::String(Host(), "native_qwen"));
        retValue = NativeValue(Host(), result);
        return retValue;
    }

    X::Value Model::DebugProbe(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1) {
            std::cout << "[Model] debug_probe requires a probe key." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        std::string key = params[0].ToString();
        if (mCompiledRuntime) {
            X::Value argument = params.size() >= 2 ? params[1] : X::Value();
            retValue = NativeValue(Host(), mCompiledRuntime->DebugProbe(key, argument));
            return retValue;
        }
        if (key == "logits_top1") {
            X::ARGS sampleParams;
            sampleParams.reserve(1);
            if (params.size() >= 2) {
                sampleParams.push_back(params[1]);
            }
            retValue = SampleLogits(sampleParams, kwParams);
            if (retValue.IsObject() && retValue.IsDict()) {
                X::Value sample(retValue);
                auto result = X::Value::Dict(Host());
                result.SetItem("probe", X::Value::String(Host(), key));
                result.SetItem("status", sample["status"]);
                result.SetItem("token_id", sample["token_id"]);
                result.SetItem("token_value", sample["token_value"]);
                result.SetItem("rows", sample["rows"]);
                result.SetItem("vocab_size", sample["vocab_size"]);
                retValue = NativeValue(Host(), result);
            }
            return retValue;
        }

        std::cout << "[Model] unknown debug_probe key: " << key << std::endl;
        retValue = NativeValue(Host(), X::Value());
        return retValue;
    }

    X::Value Model::SampleLogits(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1 || !X::Tensor::IsTensor(params[0])) {
            std::cout << "[Model] sample_logits requires a logits tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        X::Tensor logits(params[0]);
        ValidateDenseTensor(logits);
        if (logits.Info().dtype != X3_TENSOR_FLOAT32 || logits.Info().rank < 1 || logits.Info().rank > 2) {
            std::cout << "[Model] sample_logits expects FLOAT32 logits shaped [vocab] or [tokens, vocab]." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        int rows = 1;
        int vocabSize = 0;
        if (logits.Info().rank == 1) {
            vocabSize = static_cast<int>(logits.Info().shape[0]);
        }
        else {
            rows = static_cast<int>(logits.Info().shape[0]);
            vocabSize = static_cast<int>(logits.Info().shape[1]);
        }
        if (rows <= 0 || vocabSize <= 0) {
            std::cout << "[Model] sample_logits invalid logits shape." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (TensorHelper::EnsureGPUMemory(logits) != TensorOpStatus::Success) {
            std::cout << "[Model] sample_logits failed to ensure logits GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        auto logitsUse = logits.Acquire();
        auto* deviceLogits = static_cast<const float*>(TensorHelper::GetGPUMemory(logits));
        if (!deviceLogits) {
            std::cout << "[Model] sample_logits logits tensor has no GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        long long* deviceTokenId = nullptr;
        float* deviceTokenValue = nullptr;
        cudaStream_t stream = nullptr;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) err = cudaMalloc(&deviceTokenId, sizeof(long long));
        if (err == cudaSuccess) err = cudaMalloc(&deviceTokenValue, sizeof(float));
        if (err == cudaSuccess) {
            err = runLogitsTop1FP32(deviceLogits, deviceTokenId, deviceTokenValue, rows, vocabSize, stream);
        }
        long long hostTokenId = -1;
        float hostTokenValue = 0.0f;
        if (err == cudaSuccess) err = cudaMemcpyAsync(&hostTokenId, deviceTokenId, sizeof(long long), cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) err = cudaMemcpyAsync(&hostTokenValue, deviceTokenValue, sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
        if (stream) cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            if (deviceTokenId) cudaFree(deviceTokenId);
            if (deviceTokenValue) cudaFree(deviceTokenValue);
            std::cout << "[Model] sample_logits failed: " << cudaGetErrorString(err) << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (deviceTokenValue) {
            cudaFree(deviceTokenValue);
        }

        if (deviceTokenId) {
            cudaFree(deviceTokenId);
        }

        auto result = X::Value::Dict(Host());
        result.SetItem("status", X::Value::String(Host(), "ok"));
        result.SetItem("token_id", X::Value(hostTokenId));
        result.SetItem("token_value", X::Value(hostTokenValue));
        result.SetItem("rows", X::Value(rows));
        result.SetItem("vocab_size", X::Value(vocabSize));
        retValue = NativeValue(Host(), result);
        return retValue;
    }

    bool Model::InitializeCompiledRuntime(
        const std::string& rootXModel,
        const std::string& cacheDirectory,
        const std::string& weightsLocation,
        const std::string& entryFunction,
        const std::string& frontend,
        const std::vector<std::vector<int>>& inputShapes,
        const std::vector<std::string>& inputDataTypes,
        const FusionPartitionOptions& partitionOptions,
        const std::string& backend,
        const std::string& precision)
    {
        mCompiledRuntime = std::make_shared<CompiledModelRuntime>(Host());
        mCompiledMode = true;
        return mCompiledRuntime->Initialize(
            rootXModel, cacheDirectory, weightsLocation, entryFunction, frontend,
            inputShapes, inputDataTypes, partitionOptions, backend, precision);
    }

    X::Value Model::RuntimeStatus(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (!mCompiledRuntime) {
            auto status = X::Value::Dict(Host());
            status.SetItem("mode", X::Value::String(Host(), mCompiledMode ? "compiled_xmodel" : "legacy"));
            status.SetItem("state", X::Value::String(Host(), mCompiledMode ? "released" : "legacy_runner"));
            status.SetItem("ready", X::Value(!mCompiledMode && GetEngine().IsValid()));
            retValue = NativeValue(Host(), status);
            return retValue;
        }
        retValue = NativeValue(Host(), mCompiledRuntime->Status());
        return retValue;
    }

    void Model::SetEngine(X::Value engine)
    {
        FixtureExecution retired;
        {
            std::lock_guard<std::mutex> guard(mFixtureMutex);
            retired = std::move(mFixtureExecution);
            mFixtureExecution = {};
            m_engine = std::move(engine);
        }
    }

    X::Value Model::GetEngine() const
    {
        std::lock_guard<std::mutex> guard(mFixtureMutex);
        return m_engine;
    }

    void Model::ReleaseFixtureExecution()
    {
        FixtureExecution retired;
        {
            std::lock_guard<std::mutex> guard(mFixtureMutex);
            retired = std::move(mFixtureExecution);
            mFixtureExecution = {};
        }
    }

    Model::FixtureExecution Model::RetainFixtureExecution()
    {
        std::lock_guard<std::mutex> guard(mFixtureMutex);
        const std::string path = m_engine.ToString();
        if (path.empty()) throw X::Error("fixture model has no engine");
        if (path == "cuda_text_mlp" || path == "cuda_exact_vision_attention" ||
            path == "cuda_linear_transpose" || path == "cuda_linear_bias_transpose")
            return {path, {}};
        if (mFixtureExecution.owner && mFixtureExecution.path == path)
            return mFixtureExecution;
        std::string error;
        auto owner = TRTBuilder::RetainCachedExecution(path, nullptr, error);
        if (!owner) throw X::Error(error.empty() ? "failed to retain fixture engine" : error);
        mFixtureExecution = {path, std::move(owner)};
        return mFixtureExecution;
    }

    X::Value Model::ReleaseRuntime(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (mCompiledRuntime) {
            mCompiledRuntime->ReleaseDeviceMemory();
            mCompiledRuntime.reset();
        }
        ReleaseFixtureExecution();
        retValue = NativeValue(Host(), X::Value(true));
        return retValue;
    }

    X::Value Model::Forward(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (mCompiledRuntime) {
            X::Value request = params.size() == 0 ? X::Value() : params[0];
            retValue = NativeValue(Host(), mCompiledRuntime->Forward(request));
            return retValue;
        }
        if (mCompiledMode) {
            auto result = X::Value::Dict(Host());
            result.SetItem("status", X::Value::String(Host(), "error"));
            result.SetItem("error_code", X::Value::String(Host(), "compiled_graph_not_ready"));
            return result;
        }
        std::cout << "[Model] Executing forward pass..." << std::endl;
        if (!GetEngine().IsValid()) {
            std::cout << "[Model] No compiled engine attached." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (params.size() < 1) {
            std::cout << "[Model] Forward requires an input tensor." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        if (!mModel.IsObject() || !mModel.IsDict()) {
            std::cout << "[Model] Loaded weights are not a dictionary." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        X::Value weights(mModel);
        // Keep a call-local owner even if release_runtime or engine replacement
        // drops the model's persistent reference while this call is executing.
        const auto fixture = RetainFixtureExecution();
        TRTBuilder builder(Host());
        X::Value gate = FindField(weights, "language_model.layers.0.mlp.gate_proj.weight");
        X::Value up = FindField(weights, "language_model.layers.0.mlp.up_proj.weight");
        X::Value down = FindField(weights, "language_model.layers.0.mlp.down_proj.weight");
        if ((mSubgraph == "qwen3_text_mlp" || mSubgraph.empty()) && gate.IsValid() && up.IsValid() && down.IsValid()) {
            retValue = NativeValue(Host(), builder.RunTextMLPEngine(fixture.path, params[0], gate, up, down));
            return retValue;
        }

        X::Value qProj = FindField(weights, "language_model.layers.0.self_attn.q_proj.weight");
        X::Value kProj = FindField(weights, "language_model.layers.0.self_attn.k_proj.weight");
        X::Value vProj = FindField(weights, "language_model.layers.0.self_attn.v_proj.weight");
        X::Value qNorm = FindField(weights, "language_model.layers.0.self_attn.q_norm.weight");
        X::Value kNorm = FindField(weights, "language_model.layers.0.self_attn.k_norm.weight");
        if (mSubgraph == "text_rope_apply" && params.size() >= 3) {
            X::Value ropeOutput = builder.RunTextRoPEEngine(fixture.path, params[0], params[1], params[2]);
            X::Value kvOwner = GetValueArg(params, kwParams, 3, "kv_cache");
            if (!kvOwner.IsValid()) {
                kvOwner = GetValueArg(params, kwParams, 3, "request");
            }
            long long kvHandle = GetHandleFromValue(kvOwner);
            if (kvHandle <= 0) {
                kvHandle = GetLongLongArg(params, kwParams, 3, "kv_handle", 0);
            }
            if (kvHandle <= 0) {
                retValue = NativeValue(Host(), ropeOutput);
                return retValue;
            }

            if (!X::Tensor::IsTensor(ropeOutput)) {
                std::cout << "[Model] text_rope_apply kv write expected TensorRT tensor output." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            X::Tensor qkv(ropeOutput);
        ValidateDenseTensor(qkv);
            int tokenCount = GetIntArg(params, kwParams, 4, "token_count", qkv.Info().rank > 0 ? static_cast<int>(qkv.Info().shape[0]) : 0);
            int startPosition = GetIntArg(params, kwParams, 5, "start_position", 0);
            if (qkv.Info().dtype != X3_TENSOR_FLOAT32 || qkv.Info().rank != 2 ||
                tokenCount <= 0 || tokenCount > qkv.Info().shape[0] || startPosition < 0) {
                std::cout << "[Model] text_rope_apply kv write invalid output tensor or range." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success) {
                std::cout << "[Model] text_rope_apply kv write failed to ensure GPU memory." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }
            auto qkvUse = qkv.Acquire();
            auto* deviceQKV = static_cast<const float*>(TensorHelper::GetGPUMemory(qkv));
            if (!deviceQKV) {
                std::cout << "[Model] text_rope_apply kv write tensor has no GPU memory." << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            char error[1024] = {};
            int rc = GarnetDevicePagedKVWriteDeviceFP32(
                kvHandle,
                deviceQKV,
                tokenCount,
                startPosition,
                error,
                static_cast<int>(sizeof(error)));
            if (rc != 0) {
                std::cout << "[Model] text_rope_apply kv write failed: " << error << std::endl;
                retValue = NativeValue(Host(), X::Value());
                return retValue;
            }

            auto result = X::Value::Dict(Host());
            int logicalLength = startPosition + tokenCount;
            if (GetKVLogicalLengthFromValue(kvOwner) > logicalLength) {
                logicalLength = GetKVLogicalLengthFromValue(kvOwner);
            }
            SetKVLogicalLengthOnValue(kvOwner, logicalLength);
            result.SetItem("status", X::Value::String(Host(), "ok"));
            result.SetItem("kv_written", X::Value(true));
            result.SetItem("kv_handle", X::Value(kvHandle));
            result.SetItem("kv_logical_length", X::Value(logicalLength));
            result.SetItem("token_count", X::Value(tokenCount));
            result.SetItem("start_position", X::Value(startPosition));
            result.SetItem("output", TensorSummary(Host(), ropeOutput));
            result.SetItem("output_tensor", ropeOutput);
            retValue = NativeValue(Host(), result);
            return retValue;
        }
        if (mSubgraph == "text_attention_core") {
            X::Value kvOwner = GetValueArg(params, kwParams, 1, "kv_cache");
            if (!kvOwner.IsValid()) {
                kvOwner = GetValueArg(params, kwParams, 1, "request");
            }
            long long kvHandle = GetHandleFromValue(kvOwner);
            if (kvHandle <= 0) {
                kvHandle = GetLongLongArg(params, kwParams, 1, "kv_handle", 0);
            }
            if (kvHandle > 0) {
                X::Tensor qkv(params[0]);
        ValidateDenseTensor(qkv);
                int sequenceLength = GetIntArg(params, kwParams, 2, "sequence_length", qkv.Info().rank == 2 ? static_cast<int>(qkv.Info().shape[0]) : 0);
                int qWidth = GetIntArg(params, kwParams, 3, "q_width", 2048);
                int logicalLength = GetKVLogicalLengthFromValue(kvOwner);
                if (logicalLength > 0 && sequenceLength > logicalLength) {
                    std::cout << "[Model] text_attention_core sequence_length exceeds KV logical length." << std::endl;
                    retValue = NativeValue(Host(), X::Value());
                    return retValue;
                }
                X::Value kvAttention = RunDeviceKVAttentionTensor(kvHandle, qkv, sequenceLength, qWidth, "text_attention_core kv attention");
                if (!X::Tensor::IsTensor(kvAttention)) {
                    retValue = NativeValue(Host(), X::Value());
                    return retValue;
                }
                auto result = X::Value::Dict(Host());
                result.SetItem("status", X::Value::String(Host(), "ok"));
                result.SetItem("kv_read", X::Value(true));
                result.SetItem("kv_handle", X::Value(kvHandle));
                result.SetItem("sequence_length", X::Value(sequenceLength));
                result.SetItem("q_width", X::Value(qWidth));
                result.SetItem("output", TensorSummary(Host(), kvAttention));
                result.SetItem("output_tensor", kvAttention);
                retValue = NativeValue(Host(), result);
                return retValue;
            }
            retValue = NativeValue(Host(), builder.RunTextAttentionEngine(fixture.path, params[0]));
            return retValue;
        }
        if (mSubgraph == "vision_attention_core") {
            retValue = NativeValue(Host(), builder.RunVisionAttentionEngine(fixture.path, params[0]));
            return retValue;
        }
        if ((mSubgraph == "text_qkv_head_norm" || mSubgraph.empty()) && qProj.IsValid() && kProj.IsValid() && vProj.IsValid() && qNorm.IsValid() && kNorm.IsValid()) {
            retValue = NativeValue(Host(), builder.RunTextQKVHeadNormEngine(fixture.path, params[0], qProj, kProj, vProj, qNorm, kNorm));
            return retValue;
        }
        if ((mSubgraph == "text_qkv_proj" || mSubgraph.empty()) && qProj.IsValid() && kProj.IsValid() && vProj.IsValid()) {
            retValue = NativeValue(Host(), builder.RunTextQKVEngine(fixture.path, params[0], qProj, kProj, vProj));
            return retValue;
        }

        X::Value oProj = FindField(weights, "language_model.layers.0.self_attn.o_proj.weight");
        if ((mSubgraph == "text_o_proj" || mSubgraph.empty()) && oProj.IsValid()) {
            retValue = NativeValue(Host(), builder.RunLinearTransposeEngine(fixture.path, params[0], oProj));
            return retValue;
        }
        X::Value embedTokens = FindField(weights, "language_model.embed_tokens.weight");
        if ((mSubgraph == "text_lm_head" || mSubgraph.empty()) && embedTokens.IsValid()) {
            retValue = NativeValue(Host(), builder.RunLinearTransposeEngine(fixture.path, params[0], embedTokens));
            return retValue;
        }

        X::Value fc1 = FindField(weights, "visual.blocks.0.mlp.linear_fc1.weight");
        X::Value fc1Bias = FindField(weights, "visual.blocks.0.mlp.linear_fc1.bias");
        X::Value fc2 = FindField(weights, "visual.blocks.0.mlp.linear_fc2.weight");
        X::Value fc2Bias = FindField(weights, "visual.blocks.0.mlp.linear_fc2.bias");
        X::Value patchWeight = FindField(weights, "visual.patch_embed.proj.weight");
        X::Value patchBias = FindField(weights, "visual.patch_embed.proj.bias");
        if ((mSubgraph == "vision_patch_embed" || mSubgraph.empty()) && patchWeight.IsValid() && patchBias.IsValid()) {
            retValue = NativeValue(Host(), builder.RunLinearBiasTransposeEngine(fixture.path, params[0], patchWeight, patchBias));
            return retValue;
        }
        X::Value genericWeight = FindField(weights, "W");
        X::Value genericBias = FindField(weights, "B");
        if ((mSubgraph == "linear_bias" || mSubgraph.empty()) && genericWeight.IsValid() && genericBias.IsValid()) {
            retValue = NativeValue(Host(), builder.RunLinearBiasTransposeEngine(fixture.path, params[0], genericWeight, genericBias));
            return retValue;
        }
        if ((mSubgraph == "vision_mlp" || mSubgraph.empty()) && fc1.IsValid() && fc1Bias.IsValid() && fc2.IsValid() && fc2Bias.IsValid()) {
            retValue = NativeValue(Host(), builder.RunVisionMLPEngine(fixture.path, params[0], fc1, fc1Bias, fc2, fc2Bias));
            return retValue;
        }

        X::Value rmsWeight = mRmsNormWeight.IsValid()
            ? mRmsNormWeight
            : FindField(weights, "language_model.layers.0.input_layernorm.weight");
        if ((mSubgraph == "rms_norm" || mSubgraph.empty()) && rmsWeight.IsValid()) {
            retValue = NativeValue(Host(), builder.RunRMSNormEngine(fixture.path, params[0], rmsWeight));
            return retValue;
        }
        X::Value postAttentionRmsWeight = mRmsNormWeight.IsValid()
            ? mRmsNormWeight
            : FindField(weights, "language_model.layers.0.post_attention_layernorm.weight");
        if ((mSubgraph == "text_post_attention_rms_norm" || mSubgraph.empty()) && postAttentionRmsWeight.IsValid()) {
            retValue = NativeValue(Host(), builder.RunRMSNormEngine(fixture.path, params[0], postAttentionRmsWeight));
            return retValue;
        }

        X::Value lnWeight = FindField(weights, "visual.blocks.0.norm1.weight");
        X::Value lnBias = FindField(weights, "visual.blocks.0.norm1.bias");
        if ((mSubgraph == "layer_norm" || mSubgraph.empty()) && lnWeight.IsValid() && lnBias.IsValid()) {
            retValue = NativeValue(Host(), builder.RunLayerNormEngine(fixture.path, params[0], lnWeight, lnBias));
            return retValue;
        }

        X::Value weight = FindField(weights, "W");
        if ((mSubgraph == "matmul" || mSubgraph.empty()) && weight.IsValid()) {
            retValue = NativeValue(Host(), builder.RunMatmulEngine(fixture.path, params[0], weight));
            return retValue;
        }

        std::cout << "[Model] Missing supported loaded weights for subgraph: " << mSubgraph << std::endl;
        retValue = NativeValue(Host(), X::Value());
        return retValue;
    }

    X::Value Model::ForwardRequest(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 1) {
            std::cout << "[Model] forward_request requires a QwenVLRequestContext or compatible request object." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        X::Value request = params[0];
        X::Value inputIds = GetObjectField(request, "input_ids");
        X::Value mmTokenTypeIds = GetObjectField(request, "mm_token_type_ids");
        X::Value pixelValues = GetObjectField(request, "pixel_values");
        X::Value imageGridTHW = GetObjectField(request, "image_grid_thw");

        auto result = X::Value::Dict(Host());
        result.SetItem("model_path", X::Value::String(Host(), mModelPath));
        result.SetItem("subgraph", X::Value::String(Host(), mSubgraph));
        result.SetItem("engine", GetEngine());
        result.SetItem("input_ids", TensorSummary(Host(), inputIds));
        result.SetItem("mm_token_type_ids", TensorSummary(Host(), mmTokenTypeIds));
        result.SetItem("pixel_values", TensorSummary(Host(), pixelValues));
        result.SetItem("image_grid_thw", TensorSummary(Host(), imageGridTHW));

        bool requestReady =
            HasGPUTensor(inputIds) &&
            HasGPUTensor(mmTokenTypeIds) &&
            HasGPUTensor(pixelValues) &&
            X::Tensor::IsTensor(imageGridTHW);
        result.SetItem("request_gpu_ready", X::Value(requestReady));

        if (!requestReady) {
            result.SetItem("status", X::Value::String(Host(), "request_not_gpu_ready"));
            retValue = NativeValue(Host(), result);
            return retValue;
        }

        bool allocateKV = GetBoolArg(params, kwParams, 1, "allocate_kv", false);
        if (allocateKV) {
            X::Tensor idsTensor(inputIds);
        ValidateDenseTensor(idsTensor);
            int promptTokens = static_cast<int>(idsTensor.Info().shape[0]);
            int maxNewTokens = GetIntArg(params, kwParams, 2, "max_new_tokens", 64);
            int pageSize = GetIntArg(params, kwParams, 3, "page_size", 16);
            int qHeads = GetIntArg(params, kwParams, 4, "q_heads", 16);
            int kvHeads = GetIntArg(params, kwParams, 5, "kv_heads", 8);
            int headDim = GetIntArg(params, kwParams, 6, "head_dim", 128);
            int physicalPages = GetIntArg(params, kwParams, 7, "physical_pages", 0);
            int maxTokens = promptTokens + std::max(0, maxNewTokens);
            X::Value kvCache = CreateDeviceKVCacheValue(Host(), maxTokens, pageSize, qHeads, kvHeads, headDim, physicalPages);
            if (!kvCache.IsValid()) {
                result.SetItem("status", X::Value::String(Host(), "kv_allocation_failed"));
                retValue = NativeValue(Host(), result);
                return retValue;
            }
            X::Value kvHandle = GetObjectField(kvCache, "handle");
            X::Value kvLogicalPages = GetObjectField(kvCache, "logical_pages");
            X::Value kvPhysicalPages = GetObjectField(kvCache, "physical_pages");
            result.SetItem("kv_cache", kvCache);
            result.SetItem("kv_allocated", X::Value(true));
            result.SetItem("kv_handle", kvHandle);
            result.SetItem("kv_logical_length", X::Value(0));

            request.SetAttr("kv_cache", kvCache);
            request.SetAttr("kv_handle", kvHandle);
            request.SetAttr("kv_max_tokens", X::Value(maxTokens));
            request.SetAttr("kv_logical_length", X::Value(0));
            request.SetAttr("kv_page_size", X::Value(pageSize));
            request.SetAttr("kv_logical_pages", kvLogicalPages);
            request.SetAttr("kv_physical_pages", kvPhysicalPages);
            request.SetAttr("kv_q_heads", X::Value(qHeads));
            request.SetAttr("kv_heads", X::Value(kvHeads));
            request.SetAttr("kv_head_dim", X::Value(headDim));
        }
        else {
            result.SetItem("kv_allocated", X::Value(false));
        }

        if (!GetEngine().IsValid()) {
            result.SetItem("status", X::Value::String(Host(), "model_engine_not_ready"));
            retValue = NativeValue(Host(), result);
            return retValue;
        }

        if (!mModel.IsObject() || !mModel.IsDict()) {
            result.SetItem("status", X::Value::String(Host(), "model_weights_not_loaded"));
            retValue = NativeValue(Host(), result);
            return retValue;
        }

        if (mSubgraph == "vision_patch_embed") {
            X::Value weights(mModel);
            X::Value patchWeight = FindField(weights, "visual.patch_embed.proj.weight");
            X::Value patchBias = FindField(weights, "visual.patch_embed.proj.bias");
            if (!X::Tensor::IsTensor(patchWeight) || !X::Tensor::IsTensor(patchBias)) {
                result.SetItem("status", X::Value::String(Host(), "vision_patch_embed_weights_missing"));
                retValue = NativeValue(Host(), result);
                return retValue;
            }

            TRTBuilder builder(Host());
            const auto fixture = RetainFixtureExecution();
            X::Value patchOutput = builder.RunLinearBiasTransposeEngine(
                fixture.path,
                pixelValues,
                patchWeight,
                patchBias);
            result.SetItem("stage", X::Value::String(Host(), "vision_patch_embed"));
            result.SetItem("output", TensorSummary(Host(), patchOutput));
            result.SetItem("output_tensor", patchOutput);
            result.SetItem("status", X::Value::String(Host(), X::Tensor::IsTensor(patchOutput) && HasGPUTensor(patchOutput)
                ? "ok"
                : "vision_patch_embed_failed"));
            retValue = NativeValue(Host(), result);
            return retValue;
        }

        result.SetItem("status", X::Value::String(Host(), "request_validated_no_runner_for_subgraph"));
        retValue = NativeValue(Host(), result);
        return retValue;
    }

    X::Value Model::CreateDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        int maxTokens = GetIntArg(params, kwParams, 0, "max_tokens", 0);
        int pageSize = GetIntArg(params, kwParams, 1, "page_size", 16);
        int qHeads = GetIntArg(params, kwParams, 2, "q_heads", 16);
        int kvHeads = GetIntArg(params, kwParams, 3, "kv_heads", 8);
        int headDim = GetIntArg(params, kwParams, 4, "head_dim", 128);
        int physicalPageCount = GetIntArg(params, kwParams, 5, "physical_pages", 0);
        retValue = NativeValue(Host(), CreateDeviceKVCacheValue(Host(), maxTokens, pageSize, qHeads, kvHeads, headDim, physicalPageCount));
        return retValue;
    }

    X::Value Model::DestroyDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        long long handle = GetLongLongArg(params, kwParams, 0, "handle", 0);
        if (handle <= 0) {
            std::cout << "[Model] destroy_device_kv_cache requires handle." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }

        char error[1024] = {};
        int rc = GarnetDestroyDevicePagedKVFP32(handle, error, static_cast<int>(sizeof(error)));
        if (rc != 0) {
            std::cout << "[Model] destroy_device_kv_cache failed: " << error << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        retValue = NativeValue(Host(), X::Value(true));
        return retValue;
    }

    X::Value Model::WriteDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 2 || !X::Tensor::IsTensor(params[1])) {
            std::cout << "[Model] write_device_kv_cache(handle_or_cache, qkv_tensor, token_count, start_position) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }

        long long handle = GetHandleFromValue(params[0]);
        X::Tensor qkv(params[1]);
        ValidateDenseTensor(qkv);
        int tokenCount = GetIntArg(params, kwParams, 2, "token_count", static_cast<int>(qkv.Info().shape[0]));
        int startPosition = GetIntArg(params, kwParams, 3, "start_position", 0);
        if (handle <= 0 || qkv.Info().dtype != X3_TENSOR_FLOAT32 || qkv.Info().rank != 2 ||
            tokenCount <= 0 || tokenCount > qkv.Info().shape[0] || startPosition < 0) {
            std::cout << "[Model] write_device_kv_cache invalid arguments." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }

        if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success) {
            std::cout << "[Model] write_device_kv_cache failed to ensure qkv GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        auto qkvUse = qkv.Acquire();
        auto* deviceQKV = static_cast<const float*>(TensorHelper::GetGPUMemory(qkv));
        if (!deviceQKV) {
            std::cout << "[Model] write_device_kv_cache qkv tensor has no GPU memory." << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }

        char error[1024] = {};
        int rc = GarnetDevicePagedKVWriteDeviceFP32(
            handle,
            deviceQKV,
            tokenCount,
            startPosition,
            error,
            static_cast<int>(sizeof(error)));
        if (rc != 0) {
            std::cout << "[Model] write_device_kv_cache failed: " << error << std::endl;
            retValue = NativeValue(Host(), X::Value(false));
            return retValue;
        }
        int logicalLength = startPosition + tokenCount;
        int previousLogicalLength = GetKVLogicalLengthFromValue(params[0]);
        if (previousLogicalLength > logicalLength) {
            logicalLength = previousLogicalLength;
        }
        SetKVLogicalLengthOnValue(params[0], logicalLength);
        retValue = NativeValue(Host(), X::Value(true));
        return retValue;
    }

    X::Value Model::AttentionDeviceKVCache(const X::ARGS& params, const X::KWARGS& kwParams)
    {
        X::Value retValue;
        auto* rt = Host()->runtime;
        if (params.size() < 3 || !X::Tensor::IsTensor(params[1])) {
            std::cout << "[Model] attention_device_kv_cache(handle_or_cache, q_tensor, sequence_length, q_width=None) expected." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }

        long long handle = GetHandleFromValue(params[0]);
        X::Tensor q(params[1]);
        int sequenceLength = GetIntArg(params, kwParams, 2, "sequence_length", 0);
        int qWidth = GetIntArg(params, kwParams, 3, "q_width", q.Info().rank == 2 ? static_cast<int>(q.Info().shape[1]) : 0);
        int logicalLength = GetKVLogicalLengthFromValue(params[0]);
        if (logicalLength > 0 && sequenceLength > logicalLength) {
            std::cout << "[Model] attention_device_kv_cache sequence_length exceeds KV logical length." << std::endl;
            retValue = NativeValue(Host(), X::Value());
            return retValue;
        }
        retValue = NativeValue(Host(), RunDeviceKVAttentionTensor(handle, q, sequenceLength, qWidth, "attention_device_kv_cache"));
        return retValue;
    }
}
