// SPDX-License-Identifier: Apache-2.0
#include "trt_builder.h"
#include "gpt_oss_extension.h"
#include "operator_plugins.h"
#include <limits>
#include <cmath>
using namespace nvinfer1;
namespace Garnet {
ITensor* TRTBuilder::GetGptOssPackedWeight(const std::string& name) {
    auto found = weightTensorMap.find(name); if (found != weightTensorMap.end()) return found->second;
    const auto* m = capturedWeightIndex ? capturedWeightIndex->Find(name) : nullptr;
    if (!m || m->dataType != "U8" || m->shape.empty() || m->shape.size() > Dims::MAX_DIMS) {
        loweringError = "GPT-OSS requires original checkpoint U8 MXFP4 weight: " + name; return nullptr;
    }
    Dims d{}; d.nbDims = int(m->shape.size()); int64_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (m->shape[i] <= 0 || m->shape[i] > INT_MAX || n > INT64_MAX / m->shape[i]) {
            loweringError = "invalid MXFP4 shape: " + name; return nullptr;
        }
        d.d[i] = int(m->shape[i]); n *= m->shape[i];
    }
    if (uint64_t(n) != m->dataSize) { loweringError = "MXFP4 byte count mismatch: " + name; return nullptr; }
    auto& file = capturedWeightFiles[m->filePath.string()];
    if (!file) { file = std::make_unique<SafeTensorsMappedFile>(); if (!file->Open(m->filePath, loweringError)) return nullptr; }
    const void* bytes = file->DataAt(m->dataOffset, m->dataSize); if (!bytes) return nullptr;
    // INT8 is an opaque byte carrier. No quantize/dequantize or arithmetic is applied.
    Weights w{DataType::kINT8, bytes, n}; auto* layer = network->addConstant(d, w);
    if (!layer || !network->setWeightsName(w, name.c_str()) || !network->markWeightsRefittable(name.c_str())) {
        loweringError = "MXFP4 byte constant creation failed: " + name; return nullptr;
    }
    auto* result = layer->getOutput(0); result->setName(name.c_str());
    weightTensorMap[name] = result; return result;
}
ITensor* TRTBuilder::LowerGptOss(const std::string& op, ITensor* source, ITensor* right, X::KWARGS& kw) {
    if (!OperatorPluginSymbol("gpt_oss", "GarnetCreateOperatorPlugin", loweringError)) return nullptr;
    auto item = [&](const char* key) -> X::Value* {
        for (auto& entry : kw) if (entry.first == key) return &entry.second; return nullptr;
    };
    auto integer = [&](const char* key, int fallback) { auto* v = item(key); return v ? int(v->ToLongLong()) : fallback; };
    auto real = [&](const char* key, float fallback) { auto* v = item(key); return v ? float(v->ToDouble()) : fallback; };
    auto text = [&](const char* key) { auto* v = item(key); return v ? v->ToString() : std::string(); };
    auto asFloat = [&](ITensor* t) -> ITensor* {
        if (!t || t->getType() == DataType::kFLOAT) return t;
        auto* c = network->addCast(*t, DataType::kFLOAT); return c ? c->getOutput(0) : nullptr;
    };
    auto weight = [&](const char* key, bool packed = false) {
        return packed ? GetGptOssPackedWeight(text(key)) : asFloat(GetOrCreateTRTWeight(text(key)));
    };
    if (op == "gpt_oss_round_bf16") {
        auto* c = network->addCast(*source, DataType::kBF16);
        return c ? asFloat(c->getOutput(0)) : nullptr;
    }
    GptOssOptions o;
    o.qHeads = integer("num_heads", 0); o.kvHeads = integer("num_kv_heads", 0); o.headDim = integer("head_dim", 0);
    o.theta = real("rope_theta", 150000); o.factor = real("rope_scaling_factor", 32);
    o.initialContext = real("initial_context_length", 4096);
    o.betaFast = real("rope_ntk_beta", 32); o.betaSlow = real("rope_ntk_alpha", 1);
    std::vector<ITensor*> inputs{asFloat(source)};
    if (op == "gpt_oss_apply_yarn_rope_packed") {
        o.kind = 0; inputs.push_back(right);
        if (!right || right->getType() != DataType::kINT64) { loweringError = "GPT-OSS YaRN requires INT64 positions"; return nullptr; }
    } else if (op == "gpt_oss_moe_mxfp4") {
        o.kind = 2; o.hidden = integer("hidden_size", 0); o.intermediate = integer("intermediate_size", 0);
        o.experts = integer("num_experts", 0); o.topK = integer("experts_per_token", 0); o.limit = real("swiglu_limit", 7);
        inputs.push_back(weight("router_weight_name")); inputs.push_back(weight("router_bias_name"));
        inputs.push_back(weight("gate_up_blocks_name", true)); inputs.push_back(weight("gate_up_scales_name", true));
        inputs.push_back(weight("gate_up_bias_name")); inputs.push_back(weight("down_blocks_name", true));
        inputs.push_back(weight("down_scales_name", true)); inputs.push_back(weight("down_bias_name"));
    } else if (op == "gpt_oss_tp_all_reduce") {
        o.kind = 3; o.hidden = integer("hidden_size", 0); o.tpRank = integer("tp_rank", -1);
    } else if (op == "gpt_oss_paged_attention") {
        o.kind = 1; o.layer = pendingKVLayerIndex; o.pageSize = integer("page_size", 16);
        o.window = integer("sliding_window", 0); o.prefill = integer("prefill", 0);
        inputs.insert(inputs.end(), {pendingKVKeyPages, pendingKVValuePages, pendingKVPageTable,
            pendingKVContextLength, pendingKVSlotPosition, pendingKVActiveMask, weight("sinks_weight_name")});
        // Binding state belongs to exactly one attention invocation.
        pendingKVKeyPages = nullptr; pendingKVValuePages = nullptr; pendingKVPageTable = nullptr;
        pendingKVContextLength = nullptr; pendingKVSlotPosition = nullptr; pendingKVActiveMask = nullptr;
        pendingKVLayerIndex = -1;
    } else { loweringError = "unknown GPT-OSS operator: " + op; return nullptr; }
    for (auto* t : inputs) if (!t) { if (loweringError.empty()) loweringError = op + " has missing weights/cache bindings"; return nullptr; }
    auto* plugin = CreateGptOssPlugin(o, loweringError);
    if (!plugin) return nullptr;
    if (plugin->initialize() != 0) { plugin->destroy(); loweringError = "invalid GPT-OSS operator attributes: " + op; return nullptr; }
    ownedPlugins.push_back(plugin);
    auto* layer = network->addPluginV2(inputs.data(), int(inputs.size()), *plugin);
    if (!layer) { loweringError = "GPT-OSS plugin lowering failed: " + op; return nullptr; }
    const std::string layerName = op + "_" + std::to_string(network->getNbLayers());
    layer->setName(layerName.c_str()); return layer->getOutput(0);
}
}
