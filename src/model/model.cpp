#include "model.h"
#include "../trt/trt_builder.h"
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
            X::V<X::XList> dims;
            int dimCount = tensor->GetDimCount();
            for (int i = 0; i < dimCount; ++i) {
                X::Value dimValue(static_cast<long long>(tensor->GetDimSize(i)));
                dims->AddItem(dimValue);
            }
            return dims;
        }

        X::Value TensorSummary(X::Value value)
        {
            X::Dict summary;
            bool isTensor = value.IsTensor();
            summary->Set("is_tensor", X::Value(isTensor));
            if (!isTensor) {
                summary->Set("gpu", X::Value(false));
                return summary;
            }

            X::Tensor tensor(value);
            void* gpuMemory = TensorHelper::GetGPUMemory(tensor);
            summary->Set("gpu", X::Value(gpuMemory != nullptr));
            summary->Set("shape", MakeShapeList(tensor));
            summary->Set("dtype", X::Value(static_cast<int>(tensor->GetDataType())));
            summary->Set("bytes", X::Value(static_cast<long long>(tensor->GetDataSize())));
            summary->Set("count", X::Value(static_cast<long long>(tensor->GetCount())));
            return summary;
        }

        bool HasGPUTensor(X::Value value)
        {
            if (!value.IsTensor()) {
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
            return value[name];
        }

        X::Value MakeInt64Tensor1D(const std::vector<long long>& values, bool ensureGpu)
        {
            X::Tensor tensor;
            X::Port::vector<int> shape(1);
            shape.push_back(static_cast<int>(values.size()));
            tensor->SetDataType(X::TensorDataType::LONGLONG);
            tensor->SetShape(shape);

            X::Value init;
            if (!tensor->Create(init) || tensor->GetData() == nullptr) {
                std::cout << "[Model] failed to allocate INT64 tensor count=" << values.size() << std::endl;
                return X::Value();
            }
            if (!values.empty()) {
                std::memcpy(tensor->GetData(), values.data(), values.size() * sizeof(long long));
            }
            if (ensureGpu && TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success) {
                std::cout << "[Model] failed to move INT64 tensor to GPU count=" << values.size() << std::endl;
                return X::Value();
            }
            return X::Value(tensor);
        }

        int GetIntArg(X::ARGS& params, X::KWARGS& kwParams, size_t pos, const char* name, int defaultValue)
        {
            for (auto& item : kwParams) {
                if (std::string(item.key) == name) {
                    return static_cast<int>(item.val.ToLongLong());
                }
            }
            if (params.size() > pos) {
                return static_cast<int>(params[pos].ToLongLong());
            }
            return defaultValue;
        }

        long long GetLongLongArg(X::ARGS& params, X::KWARGS& kwParams, size_t pos, const char* name, long long defaultValue)
        {
            for (auto& item : kwParams) {
                if (std::string(item.key) == name) {
                    return item.val.ToLongLong();
                }
            }
            if (params.size() > pos) {
                return params[pos].ToLongLong();
            }
            return defaultValue;
        }

        bool GetBoolArg(X::ARGS& params, X::KWARGS& kwParams, size_t pos, const char* name, bool defaultValue)
        {
            for (auto& item : kwParams) {
                if (std::string(item.key) == name) {
                    return item.val.ToBool();
                }
            }
            if (params.size() > pos) {
                return params[pos].ToBool();
            }
            return defaultValue;
        }

        X::Value GetValueArg(X::ARGS& params, X::KWARGS& kwParams, size_t pos, const char* name)
        {
            for (auto& item : kwParams) {
                if (std::string(item.key) == name) {
                    return item.val;
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
            if (value.GetObj()->GetType() == X::ObjType::Dict) {
                X::Dict dict(value);
                dict->Set(name, field);
                return;
            }
            value.SetPropValue(name, field);
        }

        X::Value CreateDeviceKVCacheValue(
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

            X::Dict result;
            result->Set("handle", X::Value(handle));
            result->Set("max_tokens", X::Value(maxTokens));
            result->Set("logical_length", X::Value(0));
            result->Set("page_size", X::Value(pageSize));
            result->Set("logical_pages", X::Value(logicalPageCount));
            result->Set("physical_pages", X::Value(physicalPageCount));
            result->Set("q_heads", X::Value(qHeads));
            result->Set("kv_heads", X::Value(kvHeads));
            result->Set("head_dim", X::Value(headDim));
            result->Set("key_bytes", X::Value(bytesPerArena));
            result->Set("value_bytes", X::Value(bytesPerArena));
            result->Set("status", X::Value("ok"));
            return result;
        }

        long long GetHandleFromValue(X::Value value)
        {
            if (!value.IsValid()) {
                return 0;
            }
            if (value.IsObject()) {
                X::Value handle = value["handle"];
                if (handle.IsValid()) {
                    return handle.ToLongLong();
                }
                handle = value["kv_handle"];
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
            X::Value logicalLength = value["logical_length"];
            if (logicalLength.IsValid()) {
                return static_cast<int>(logicalLength.ToLongLong());
            }
            logicalLength = value["kv_logical_length"];
            if (logicalLength.IsValid()) {
                return static_cast<int>(logicalLength.ToLongLong());
            }
            X::Value kvCache = value["kv_cache"];
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
            X::Value kvCache = value["kv_cache"];
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
            if (handle <= 0 || sequenceLength <= 0 || qWidth <= 0 ||
                q->GetDataType() != X::TensorDataType::FLOAT32 || q->GetDimCount() != 2 || q->GetDimSize(0) < 1 ||
                q->GetDimSize(1) < qWidth) {
                std::cout << "[Model] " << caller << " invalid arguments." << std::endl;
                return X::Value();
            }

            if (TensorHelper::EnsureGPUMemory(q) != TensorOpStatus::Success) {
                std::cout << "[Model] " << caller << " failed to ensure q GPU memory." << std::endl;
                return X::Value();
            }
            auto* qBase = static_cast<const float*>(TensorHelper::GetGPUMemory(q));
            if (!qBase) {
                std::cout << "[Model] " << caller << " q tensor has no GPU memory." << std::endl;
                return X::Value();
            }

            int qStride = static_cast<int>(q->GetDimSize(1));
            const float* deviceQ = qBase + static_cast<size_t>(q->GetDimSize(0) - 1) * static_cast<size_t>(qStride);
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

            X::Tensor output;
            output->SetDataType(X::TensorDataType::FLOAT32);
            X::Port::vector<int> outputShape(2);
            outputShape.push_back(1);
            outputShape.push_back(qWidth);
            output->SetShape(outputShape);
            if (TensorHelper::AttachGPUMemory(output, deviceOutput) != TensorOpStatus::Success) {
                cudaFree(deviceOutput);
                std::cout << "[Model] " << caller << " failed to attach output GPU memory." << std::endl;
                return X::Value();
            }
            return X::Value(output);
        }
    }

    X::Value Model::Access(X::Port::vector<X::Value>& IdxAry)
    {
		return mModel.GetObjectValue(IdxAry);
    }
    void Model::Tokenizer(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1) {
            std::cout << "[Model] tokenizer requires input text." << std::endl;
            retValue = X::Value();
            return;
        }

        bool addSpecialTokens = false;
        for (auto& item : kwParams) {
            if (std::string(item.key) == "add_special_tokens") {
                addSpecialTokens = item.val.ToBool();
            }
        }

        std::string error;
        auto tokenizer = Tokenization::GetCachedQwenTokenizer(mModelPath, &error);
        if (!tokenizer) {
            std::cout << "[Model] native tokenizer load failed: " << error << std::endl;
            retValue = X::Value();
            return;
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

        X::Value tensorIds = MakeInt64Tensor1D(inputIds, true);
        X::Value tensorAttentionMask = MakeInt64Tensor1D(attentionMask, true);
        if (!tensorIds.IsTensor() || !tensorAttentionMask.IsTensor()) {
            retValue = X::Value();
            return;
        }

        X::Dict dictInputs;
        dictInputs->Set("input_ids", tensorIds);
        dictInputs->Set("attention_mask", tensorAttentionMask);
        dictInputs->Set("token_count", X::Value(static_cast<int>(inputIds.size())));
        dictInputs->Set("tokenizer", X::Value("native_qwen"));
        retValue = dictInputs;
    }

    void Model::Detokenizer(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1) {
            std::cout << "[Model] detokenizer requires token ids." << std::endl;
            retValue = X::Value();
            return;
        }

        bool skipSpecialTokens = true;
        for (auto& item : kwParams) {
            if (std::string(item.key) == "skip_special_tokens") {
                skipSpecialTokens = item.val.ToBool();
            }
        }

        std::vector<int64_t> ids;
        X::Value tokenValue = params[0];
        if (tokenValue.IsTensor()) {
            X::Tensor tokenTensor(tokenValue);
            if (tokenTensor->GetDataType() != X::TensorDataType::LONGLONG) {
                std::cout << "[Model] detokenizer tensor must be INT64/LONGLONG." << std::endl;
                retValue = X::Value();
                return;
            }
            if (TensorHelper::GetGPUMemory(tokenTensor) != nullptr &&
                TensorHelper::CopyResultFromGPU(tokenTensor) != TensorOpStatus::Success) {
                std::cout << "[Model] detokenizer failed to copy final token ids from GPU." << std::endl;
                retValue = X::Value();
                return;
            }
            auto* data = reinterpret_cast<long long*>(tokenTensor->GetData());
            long long count = tokenTensor->GetCount();
            ids.reserve(static_cast<size_t>(count));
            for (long long i = 0; i < count; ++i) {
                ids.push_back(static_cast<int64_t>(data[i]));
            }
        }
        else if (tokenValue.IsList()) {
            X::List list(tokenValue);
            long long count = list->Size();
            ids.reserve(static_cast<size_t>(count));
            for (long long i = 0; i < count; ++i) {
                ids.push_back(static_cast<int64_t>(list->Get(i).ToLongLong()));
            }
        }
        else {
            ids.push_back(static_cast<int64_t>(tokenValue.ToLongLong()));
        }

        std::string error;
        auto tokenizer = Tokenization::GetCachedQwenTokenizer(mModelPath, &error);
        if (!tokenizer) {
            std::cout << "[Model] native tokenizer load failed: " << error << std::endl;
            retValue = X::Value();
            return;
        }

        std::string text = tokenizer->Decode(ids, skipSpecialTokens);
        X::Dict result;
        result->Set("text", X::Value(text));
        result->Set("token_count", X::Value(static_cast<int>(ids.size())));
        result->Set("tokenizer", X::Value("native_qwen"));
        retValue = result;
    }

    void Model::DebugProbe(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1) {
            std::cout << "[Model] debug_probe requires a probe key." << std::endl;
            retValue = X::Value();
            return;
        }

        std::string key = params[0].ToString();
        if (mCompiledRuntime) {
            X::Value argument = params.size() >= 2 ? params[1] : X::Value();
            retValue = mCompiledRuntime->DebugProbe(key, argument);
            return;
        }
        if (key == "logits_top1") {
            X::ARGS sampleParams(1);
            if (params.size() >= 2) {
                sampleParams.push_back(params[1]);
            }
            SampleLogits(rt, pContext, sampleParams, kwParams, retValue);
            if (retValue.IsObject() && retValue.GetObj()->GetType() == X::ObjType::Dict) {
                X::Dict sample(retValue);
                X::Dict result;
                result->Set("probe", X::Value(key));
                result->Set("status", sample["status"]);
                result->Set("token_id", sample["token_id"]);
                result->Set("token_value", sample["token_value"]);
                result->Set("rows", sample["rows"]);
                result->Set("vocab_size", sample["vocab_size"]);
                retValue = result;
            }
            return;
        }

        std::cout << "[Model] unknown debug_probe key: " << key << std::endl;
        retValue = X::Value();
    }

    void Model::SampleLogits(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1 || !params[0].IsTensor()) {
            std::cout << "[Model] sample_logits requires a logits tensor." << std::endl;
            retValue = X::Value();
            return;
        }

        X::Tensor logits(params[0]);
        if (logits->GetDataType() != X::TensorDataType::FLOAT32 || logits->GetDimCount() < 1 || logits->GetDimCount() > 2) {
            std::cout << "[Model] sample_logits expects FLOAT32 logits shaped [vocab] or [tokens, vocab]." << std::endl;
            retValue = X::Value();
            return;
        }
        int rows = 1;
        int vocabSize = 0;
        if (logits->GetDimCount() == 1) {
            vocabSize = static_cast<int>(logits->GetDimSize(0));
        }
        else {
            rows = static_cast<int>(logits->GetDimSize(0));
            vocabSize = static_cast<int>(logits->GetDimSize(1));
        }
        if (rows <= 0 || vocabSize <= 0) {
            std::cout << "[Model] sample_logits invalid logits shape." << std::endl;
            retValue = X::Value();
            return;
        }
        if (TensorHelper::EnsureGPUMemory(logits) != TensorOpStatus::Success) {
            std::cout << "[Model] sample_logits failed to ensure logits GPU memory." << std::endl;
            retValue = X::Value();
            return;
        }
        auto* deviceLogits = static_cast<const float*>(TensorHelper::GetGPUMemory(logits));
        if (!deviceLogits) {
            std::cout << "[Model] sample_logits logits tensor has no GPU memory." << std::endl;
            retValue = X::Value();
            return;
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
            retValue = X::Value();
            return;
        }
        if (deviceTokenValue) {
            cudaFree(deviceTokenValue);
        }

        if (deviceTokenId) {
            cudaFree(deviceTokenId);
        }

        X::Dict result;
        result->Set("status", X::Value("ok"));
        result->Set("token_id", X::Value(hostTokenId));
        result->Set("token_value", X::Value(hostTokenValue));
        result->Set("rows", X::Value(rows));
        result->Set("vocab_size", X::Value(vocabSize));
        retValue = result;
    }

    bool Model::InitializeCompiledRuntime(
        const std::string& rootXModel,
        const std::string& cacheDirectory,
        const std::string& weightsLocation,
        const std::string& entryFunction,
        const std::string& frontend,
        const std::vector<std::vector<int>>& inputShapes,
        const std::vector<std::string>& inputDataTypes)
    {
        mCompiledRuntime = std::make_shared<CompiledModelRuntime>();
        return mCompiledRuntime->Initialize(
            rootXModel, cacheDirectory, weightsLocation, entryFunction, frontend, inputShapes, inputDataTypes);
    }

    void Model::RuntimeStatus(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (!mCompiledRuntime) {
            X::Dict status;
            status->Set("mode", X::Value("legacy"));
            status->Set("state", X::Value("legacy_runner"));
            status->Set("ready", X::Value(m_engine.IsValid()));
            retValue = status;
            return;
        }
        retValue = mCompiledRuntime->Status();
    }

    void Model::Forward(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (mCompiledRuntime) {
            X::Value request = params.size() == 0 ? X::Value() : params[0];
            retValue = mCompiledRuntime->Forward(request);
            return;
        }
        std::cout << "[Model] Executing forward pass..." << std::endl;
        if (!m_engine.IsValid()) {
            std::cout << "[Model] No compiled engine attached." << std::endl;
            retValue = X::Value();
            return;
        }
        if (params.size() < 1) {
            std::cout << "[Model] Forward requires an input tensor." << std::endl;
            retValue = X::Value();
            return;
        }
        if (!mModel.IsObject() || mModel.GetObj()->GetType() != X::ObjType::Dict) {
            std::cout << "[Model] Loaded weights are not a dictionary." << std::endl;
            retValue = X::Value();
            return;
        }
        X::Dict weights(mModel);
        TRTBuilder builder;
        X::Value gate = weights["language_model.layers.0.mlp.gate_proj.weight"];
        X::Value up = weights["language_model.layers.0.mlp.up_proj.weight"];
        X::Value down = weights["language_model.layers.0.mlp.down_proj.weight"];
        if ((mSubgraph == "qwen3_text_mlp" || mSubgraph.empty()) && gate.IsValid() && up.IsValid() && down.IsValid()) {
            retValue = builder.RunTextMLPEngine(m_engine.ToString(), params[0], gate, up, down);
            return;
        }

        X::Value qProj = weights["language_model.layers.0.self_attn.q_proj.weight"];
        X::Value kProj = weights["language_model.layers.0.self_attn.k_proj.weight"];
        X::Value vProj = weights["language_model.layers.0.self_attn.v_proj.weight"];
        X::Value qNorm = weights["language_model.layers.0.self_attn.q_norm.weight"];
        X::Value kNorm = weights["language_model.layers.0.self_attn.k_norm.weight"];
        if (mSubgraph == "text_rope_apply" && params.size() >= 3) {
            X::Value ropeOutput = builder.RunTextRoPEEngine(m_engine.ToString(), params[0], params[1], params[2]);
            X::Value kvOwner = GetValueArg(params, kwParams, 3, "kv_cache");
            if (!kvOwner.IsValid()) {
                kvOwner = GetValueArg(params, kwParams, 3, "request");
            }
            long long kvHandle = GetHandleFromValue(kvOwner);
            if (kvHandle <= 0) {
                kvHandle = GetLongLongArg(params, kwParams, 3, "kv_handle", 0);
            }
            if (kvHandle <= 0) {
                retValue = ropeOutput;
                return;
            }

            if (!ropeOutput.IsTensor()) {
                std::cout << "[Model] text_rope_apply kv write expected TensorRT tensor output." << std::endl;
                retValue = X::Value();
                return;
            }
            X::Tensor qkv(ropeOutput);
            int tokenCount = GetIntArg(params, kwParams, 4, "token_count", qkv->GetDimCount() > 0 ? static_cast<int>(qkv->GetDimSize(0)) : 0);
            int startPosition = GetIntArg(params, kwParams, 5, "start_position", 0);
            if (qkv->GetDataType() != X::TensorDataType::FLOAT32 || qkv->GetDimCount() != 2 ||
                tokenCount <= 0 || tokenCount > qkv->GetDimSize(0) || startPosition < 0) {
                std::cout << "[Model] text_rope_apply kv write invalid output tensor or range." << std::endl;
                retValue = X::Value();
                return;
            }
            if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success) {
                std::cout << "[Model] text_rope_apply kv write failed to ensure GPU memory." << std::endl;
                retValue = X::Value();
                return;
            }
            auto* deviceQKV = static_cast<const float*>(TensorHelper::GetGPUMemory(qkv));
            if (!deviceQKV) {
                std::cout << "[Model] text_rope_apply kv write tensor has no GPU memory." << std::endl;
                retValue = X::Value();
                return;
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
                retValue = X::Value();
                return;
            }

            X::Dict result;
            int logicalLength = startPosition + tokenCount;
            if (GetKVLogicalLengthFromValue(kvOwner) > logicalLength) {
                logicalLength = GetKVLogicalLengthFromValue(kvOwner);
            }
            SetKVLogicalLengthOnValue(kvOwner, logicalLength);
            result->Set("status", X::Value("ok"));
            result->Set("kv_written", X::Value(true));
            result->Set("kv_handle", X::Value(kvHandle));
            result->Set("kv_logical_length", X::Value(logicalLength));
            result->Set("token_count", X::Value(tokenCount));
            result->Set("start_position", X::Value(startPosition));
            result->Set("output", TensorSummary(ropeOutput));
            result->Set("output_tensor", ropeOutput);
            retValue = result;
            return;
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
                int sequenceLength = GetIntArg(params, kwParams, 2, "sequence_length", qkv->GetDimCount() == 2 ? static_cast<int>(qkv->GetDimSize(0)) : 0);
                int qWidth = GetIntArg(params, kwParams, 3, "q_width", 2048);
                int logicalLength = GetKVLogicalLengthFromValue(kvOwner);
                if (logicalLength > 0 && sequenceLength > logicalLength) {
                    std::cout << "[Model] text_attention_core sequence_length exceeds KV logical length." << std::endl;
                    retValue = X::Value();
                    return;
                }
                X::Value kvAttention = RunDeviceKVAttentionTensor(kvHandle, qkv, sequenceLength, qWidth, "text_attention_core kv attention");
                if (!kvAttention.IsTensor()) {
                    retValue = X::Value();
                    return;
                }
                X::Dict result;
                result->Set("status", X::Value("ok"));
                result->Set("kv_read", X::Value(true));
                result->Set("kv_handle", X::Value(kvHandle));
                result->Set("sequence_length", X::Value(sequenceLength));
                result->Set("q_width", X::Value(qWidth));
                result->Set("output", TensorSummary(kvAttention));
                result->Set("output_tensor", kvAttention);
                retValue = result;
                return;
            }
            retValue = builder.RunTextAttentionEngine(m_engine.ToString(), params[0]);
            return;
        }
        if (mSubgraph == "vision_attention_core") {
            retValue = builder.RunVisionAttentionEngine(m_engine.ToString(), params[0]);
            return;
        }
        if ((mSubgraph == "text_qkv_head_norm" || mSubgraph.empty()) && qProj.IsValid() && kProj.IsValid() && vProj.IsValid() && qNorm.IsValid() && kNorm.IsValid()) {
            retValue = builder.RunTextQKVHeadNormEngine(m_engine.ToString(), params[0], qProj, kProj, vProj, qNorm, kNorm);
            return;
        }
        if ((mSubgraph == "text_qkv_proj" || mSubgraph.empty()) && qProj.IsValid() && kProj.IsValid() && vProj.IsValid()) {
            retValue = builder.RunTextQKVEngine(m_engine.ToString(), params[0], qProj, kProj, vProj);
            return;
        }

        X::Value oProj = weights["language_model.layers.0.self_attn.o_proj.weight"];
        if ((mSubgraph == "text_o_proj" || mSubgraph.empty()) && oProj.IsValid()) {
            retValue = builder.RunLinearTransposeEngine(m_engine.ToString(), params[0], oProj);
            return;
        }
        X::Value embedTokens = weights["language_model.embed_tokens.weight"];
        if ((mSubgraph == "text_lm_head" || mSubgraph.empty()) && embedTokens.IsValid()) {
            retValue = builder.RunLinearTransposeEngine(m_engine.ToString(), params[0], embedTokens);
            return;
        }

        X::Value fc1 = weights["visual.blocks.0.mlp.linear_fc1.weight"];
        X::Value fc1Bias = weights["visual.blocks.0.mlp.linear_fc1.bias"];
        X::Value fc2 = weights["visual.blocks.0.mlp.linear_fc2.weight"];
        X::Value fc2Bias = weights["visual.blocks.0.mlp.linear_fc2.bias"];
        X::Value patchWeight = weights["visual.patch_embed.proj.weight"];
        X::Value patchBias = weights["visual.patch_embed.proj.bias"];
        if ((mSubgraph == "vision_patch_embed" || mSubgraph.empty()) && patchWeight.IsValid() && patchBias.IsValid()) {
            retValue = builder.RunLinearBiasTransposeEngine(m_engine.ToString(), params[0], patchWeight, patchBias);
            return;
        }
        X::Value genericWeight = weights["W"];
        X::Value genericBias = weights["B"];
        if ((mSubgraph == "linear_bias" || mSubgraph.empty()) && genericWeight.IsValid() && genericBias.IsValid()) {
            retValue = builder.RunLinearBiasTransposeEngine(m_engine.ToString(), params[0], genericWeight, genericBias);
            return;
        }
        if ((mSubgraph == "vision_mlp" || mSubgraph.empty()) && fc1.IsValid() && fc1Bias.IsValid() && fc2.IsValid() && fc2Bias.IsValid()) {
            retValue = builder.RunVisionMLPEngine(m_engine.ToString(), params[0], fc1, fc1Bias, fc2, fc2Bias);
            return;
        }

        X::Value rmsWeight = mRmsNormWeight.IsValid()
            ? mRmsNormWeight
            : weights["language_model.layers.0.input_layernorm.weight"];
        if ((mSubgraph == "rms_norm" || mSubgraph.empty()) && rmsWeight.IsValid()) {
            retValue = builder.RunRMSNormEngine(m_engine.ToString(), params[0], rmsWeight);
            return;
        }
        X::Value postAttentionRmsWeight = mRmsNormWeight.IsValid()
            ? mRmsNormWeight
            : weights["language_model.layers.0.post_attention_layernorm.weight"];
        if ((mSubgraph == "text_post_attention_rms_norm" || mSubgraph.empty()) && postAttentionRmsWeight.IsValid()) {
            retValue = builder.RunRMSNormEngine(m_engine.ToString(), params[0], postAttentionRmsWeight);
            return;
        }

        X::Value lnWeight = weights["visual.blocks.0.norm1.weight"];
        X::Value lnBias = weights["visual.blocks.0.norm1.bias"];
        if ((mSubgraph == "layer_norm" || mSubgraph.empty()) && lnWeight.IsValid() && lnBias.IsValid()) {
            retValue = builder.RunLayerNormEngine(m_engine.ToString(), params[0], lnWeight, lnBias);
            return;
        }

        X::Value weight = weights["W"];
        if ((mSubgraph == "matmul" || mSubgraph.empty()) && weight.IsValid()) {
            retValue = builder.RunMatmulEngine(m_engine.ToString(), params[0], weight);
            return;
        }

        std::cout << "[Model] Missing supported loaded weights for subgraph: " << mSubgraph << std::endl;
        retValue = X::Value();
    }

    void Model::ForwardRequest(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 1) {
            std::cout << "[Model] forward_request requires a QwenVLRequestContext or compatible request object." << std::endl;
            retValue = X::Value();
            return;
        }

        X::Value request = params[0];
        X::Value inputIds = GetObjectField(request, "input_ids");
        X::Value mmTokenTypeIds = GetObjectField(request, "mm_token_type_ids");
        X::Value pixelValues = GetObjectField(request, "pixel_values");
        X::Value imageGridTHW = GetObjectField(request, "image_grid_thw");

        X::Dict result;
        result->Set("model_path", X::Value(mModelPath));
        result->Set("subgraph", X::Value(mSubgraph));
        result->Set("engine", m_engine);
        result->Set("input_ids", TensorSummary(inputIds));
        result->Set("mm_token_type_ids", TensorSummary(mmTokenTypeIds));
        result->Set("pixel_values", TensorSummary(pixelValues));
        result->Set("image_grid_thw", TensorSummary(imageGridTHW));

        bool requestReady =
            HasGPUTensor(inputIds) &&
            HasGPUTensor(mmTokenTypeIds) &&
            HasGPUTensor(pixelValues) &&
            imageGridTHW.IsTensor();
        result->Set("request_gpu_ready", X::Value(requestReady));

        if (!requestReady) {
            result->Set("status", X::Value("request_not_gpu_ready"));
            retValue = result;
            return;
        }

        bool allocateKV = GetBoolArg(params, kwParams, 1, "allocate_kv", false);
        if (allocateKV) {
            X::Tensor idsTensor(inputIds);
            int promptTokens = static_cast<int>(idsTensor->GetDimSize(0));
            int maxNewTokens = GetIntArg(params, kwParams, 2, "max_new_tokens", 64);
            int pageSize = GetIntArg(params, kwParams, 3, "page_size", 16);
            int qHeads = GetIntArg(params, kwParams, 4, "q_heads", 16);
            int kvHeads = GetIntArg(params, kwParams, 5, "kv_heads", 8);
            int headDim = GetIntArg(params, kwParams, 6, "head_dim", 128);
            int physicalPages = GetIntArg(params, kwParams, 7, "physical_pages", 0);
            int maxTokens = promptTokens + std::max(0, maxNewTokens);
            X::Value kvCache = CreateDeviceKVCacheValue(maxTokens, pageSize, qHeads, kvHeads, headDim, physicalPages);
            if (!kvCache.IsValid()) {
                result->Set("status", X::Value("kv_allocation_failed"));
                retValue = result;
                return;
            }
            X::Value kvHandle = GetObjectField(kvCache, "handle");
            X::Value kvLogicalPages = GetObjectField(kvCache, "logical_pages");
            X::Value kvPhysicalPages = GetObjectField(kvCache, "physical_pages");
            result->Set("kv_cache", kvCache);
            result->Set("kv_allocated", X::Value(true));
            result->Set("kv_handle", kvHandle);
            result->Set("kv_logical_length", X::Value(0));

            request.SetPropValue("kv_cache", kvCache);
            request.SetPropValue("kv_handle", kvHandle);
            request.SetPropValue("kv_max_tokens", X::Value(maxTokens));
            request.SetPropValue("kv_logical_length", X::Value(0));
            request.SetPropValue("kv_page_size", X::Value(pageSize));
            request.SetPropValue("kv_logical_pages", kvLogicalPages);
            request.SetPropValue("kv_physical_pages", kvPhysicalPages);
            request.SetPropValue("kv_q_heads", X::Value(qHeads));
            request.SetPropValue("kv_heads", X::Value(kvHeads));
            request.SetPropValue("kv_head_dim", X::Value(headDim));
        }
        else {
            result->Set("kv_allocated", X::Value(false));
        }

        if (!m_engine.IsValid()) {
            result->Set("status", X::Value("model_engine_not_ready"));
            retValue = result;
            return;
        }

        if (!mModel.IsObject() || mModel.GetObj()->GetType() != X::ObjType::Dict) {
            result->Set("status", X::Value("model_weights_not_loaded"));
            retValue = result;
            return;
        }

        if (mSubgraph == "vision_patch_embed") {
            X::Dict weights(mModel);
            X::Value patchWeight = weights["visual.patch_embed.proj.weight"];
            X::Value patchBias = weights["visual.patch_embed.proj.bias"];
            if (!patchWeight.IsTensor() || !patchBias.IsTensor()) {
                result->Set("status", X::Value("vision_patch_embed_weights_missing"));
                retValue = result;
                return;
            }

            TRTBuilder builder;
            X::Value patchOutput = builder.RunLinearBiasTransposeEngine(
                m_engine.ToString(),
                pixelValues,
                patchWeight,
                patchBias);
            result->Set("stage", X::Value("vision_patch_embed"));
            result->Set("output", TensorSummary(patchOutput));
            result->Set("output_tensor", patchOutput);
            result->Set("status", X::Value(patchOutput.IsTensor() && HasGPUTensor(patchOutput)
                ? "ok"
                : "vision_patch_embed_failed"));
            retValue = result;
            return;
        }

        result->Set("status", X::Value("request_validated_no_runner_for_subgraph"));
        retValue = result;
    }

    void Model::CreateDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        int maxTokens = GetIntArg(params, kwParams, 0, "max_tokens", 0);
        int pageSize = GetIntArg(params, kwParams, 1, "page_size", 16);
        int qHeads = GetIntArg(params, kwParams, 2, "q_heads", 16);
        int kvHeads = GetIntArg(params, kwParams, 3, "kv_heads", 8);
        int headDim = GetIntArg(params, kwParams, 4, "head_dim", 128);
        int physicalPageCount = GetIntArg(params, kwParams, 5, "physical_pages", 0);
        retValue = CreateDeviceKVCacheValue(maxTokens, pageSize, qHeads, kvHeads, headDim, physicalPageCount);
    }

    void Model::DestroyDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        long long handle = GetLongLongArg(params, kwParams, 0, "handle", 0);
        if (handle <= 0) {
            std::cout << "[Model] destroy_device_kv_cache requires handle." << std::endl;
            retValue = X::Value(false);
            return;
        }

        char error[1024] = {};
        int rc = GarnetDestroyDevicePagedKVFP32(handle, error, static_cast<int>(sizeof(error)));
        if (rc != 0) {
            std::cout << "[Model] destroy_device_kv_cache failed: " << error << std::endl;
            retValue = X::Value(false);
            return;
        }
        retValue = X::Value(true);
    }

    void Model::WriteDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 2 || !params[1].IsTensor()) {
            std::cout << "[Model] write_device_kv_cache(handle_or_cache, qkv_tensor, token_count, start_position) expected." << std::endl;
            retValue = X::Value(false);
            return;
        }

        long long handle = GetHandleFromValue(params[0]);
        X::Tensor qkv(params[1]);
        int tokenCount = GetIntArg(params, kwParams, 2, "token_count", static_cast<int>(qkv->GetDimSize(0)));
        int startPosition = GetIntArg(params, kwParams, 3, "start_position", 0);
        if (handle <= 0 || qkv->GetDataType() != X::TensorDataType::FLOAT32 || qkv->GetDimCount() != 2 ||
            tokenCount <= 0 || tokenCount > qkv->GetDimSize(0) || startPosition < 0) {
            std::cout << "[Model] write_device_kv_cache invalid arguments." << std::endl;
            retValue = X::Value(false);
            return;
        }

        if (TensorHelper::EnsureGPUMemory(qkv) != TensorOpStatus::Success) {
            std::cout << "[Model] write_device_kv_cache failed to ensure qkv GPU memory." << std::endl;
            retValue = X::Value(false);
            return;
        }
        auto* deviceQKV = static_cast<const float*>(TensorHelper::GetGPUMemory(qkv));
        if (!deviceQKV) {
            std::cout << "[Model] write_device_kv_cache qkv tensor has no GPU memory." << std::endl;
            retValue = X::Value(false);
            return;
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
            retValue = X::Value(false);
            return;
        }
        int logicalLength = startPosition + tokenCount;
        int previousLogicalLength = GetKVLogicalLengthFromValue(params[0]);
        if (previousLogicalLength > logicalLength) {
            logicalLength = previousLogicalLength;
        }
        SetKVLogicalLengthOnValue(params[0], logicalLength);
        retValue = X::Value(true);
    }

    void Model::AttentionDeviceKVCache(X::XRuntime* rt, X::XObj* pContext, X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() < 3 || !params[1].IsTensor()) {
            std::cout << "[Model] attention_device_kv_cache(handle_or_cache, q_tensor, sequence_length, q_width=None) expected." << std::endl;
            retValue = X::Value();
            return;
        }

        long long handle = GetHandleFromValue(params[0]);
        X::Tensor q(params[1]);
        int sequenceLength = GetIntArg(params, kwParams, 2, "sequence_length", 0);
        int qWidth = GetIntArg(params, kwParams, 3, "q_width", q->GetDimCount() == 2 ? static_cast<int>(q->GetDimSize(1)) : 0);
        int logicalLength = GetKVLogicalLengthFromValue(params[0]);
        if (logicalLength > 0 && sequenceLength > logicalLength) {
            std::cout << "[Model] attention_device_kv_cache sequence_length exceeds KV logical length." << std::endl;
            retValue = X::Value();
            return;
        }
        retValue = RunDeviceKVAttentionTensor(handle, q, sequenceLength, qWidth, "attention_device_kv_cache");
    }
}
