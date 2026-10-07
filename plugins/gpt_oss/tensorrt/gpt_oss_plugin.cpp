// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_plugin.h"
#include <NvInferPlugin.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <limits>

using namespace nvinfer1;
namespace Garnet {
namespace {
constexpr const char* kName = "GarnetGptOss";
void reportMarlinFallback(bool initialized, int tokens) {
    static std::atomic<int> count{0};
    if (std::getenv("GARNET_GPT_OSS_DEBUG_MARLIN") && count.fetch_add(1) < 16)
        std::fprintf(stderr, "GPT-OSS Marlin: fallback (plugin initialized=%d, rows=%d)\n", initialized, tokens);
}
// The serialized options layout remains compatible with version 7 engines.
constexpr const char* kVersion = "7";
bool valid(const GptOssOptions& o) {
    if (o.kind < 0 || o.kind > 6) return false;
    if (o.bf16Communication < 0 || o.bf16Communication > 1 ||
        (o.bf16Communication && o.kind != 3)) return false;
    if (o.kind == 3 || o.kind == 4) return o.hidden > 0 && o.tpRank >= 0 && o.tpRank < 2;
    if (o.kind == 5) return o.hidden > 0 && o.epsilon > 0.f;
    if (o.kind == 6) return o.hidden > 0 && o.hidden % 32 == 0 &&
        o.intermediate > 0 && o.intermediate <= 16384 &&
        (o.tpRank == 0 || o.tpRank == 1) &&
        (o.qHeads == 0 || (o.qHeads > 0 && o.kvHeads > 0 &&
         o.headDim > 0 && o.qHeads % 2 == 0 && o.kvHeads % 2 == 0 &&
         o.intermediate == (o.qHeads / 2 + o.kvHeads) * o.headDim));
    if (o.kind == 2) return o.hidden > 0 && o.intermediate > 0 &&
        o.hidden % 32 == 0 && o.intermediate % 32 == 0 && o.experts > 0 &&
        o.experts <= 256 && o.topK > 0 && o.topK <= 8 && o.topK <= o.experts && o.limit > 0 &&
        o.tpRank >= -1 && o.tpRank < 2;
    return o.qHeads > 0 && o.kvHeads > 0 && o.qHeads % o.kvHeads == 0 &&
        o.headDim > 0 && o.headDim <= 128 && o.headDim % 2 == 0 &&
        o.pageSize > 0 && o.layer >= 0 && o.window >= 0 &&
        o.theta > 1 && o.factor >= 1 && o.initialContext > 0 &&
        o.betaFast > o.betaSlow && o.betaSlow > 0;
}
int rows(const Dims& d) {
    if (d.nbDims != 3 || d.d[0] <= 0 || d.d[1] <= 0 || d.d[2] <= 0 ||
        d.d[0] > std::numeric_limits<int>::max() / d.d[1]) return 0;
    return d.d[0] * d.d[1];
}
bool shape(const Dims& d, std::initializer_list<int64_t> expected) {
    if (d.nbDims != int(expected.size())) return false;
    int i = 0; for (int64_t x : expected) { if (d.d[i++] != x) return false; }
    return true;
}
}
GptOssPlugin::GptOssPlugin(const GptOssOptions& o) : m_options(o), m_valid(valid(o)) {}
GptOssPlugin::GptOssPlugin(const void* data, size_t size) {
    if (data && size == sizeof(m_options)) { std::memcpy(&m_options, data, size); m_valid = valid(m_options); }
}
IPluginV2DynamicExt* GptOssPlugin::clone() const noexcept {
    auto* p = new (std::nothrow) GptOssPlugin(m_options);
    if (p) { p->m_valid = m_valid; p->setPluginNamespace(m_namespace.c_str()); } return p;
}
DimsExprs GptOssPlugin::getOutputDimensions(int, const DimsExprs* in, int count, IExprBuilder& b) noexcept {
    if (!in || count < 1 || in[0].nbDims != 3) return {};
    auto out = in[0];
    if (m_options.kind == 1) out.d[2] = b.constant(m_options.qHeads * m_options.headDim);
    if (m_options.kind == 4) {
        const auto* two = b.constant(2);
        out.d[2] = two ? b.operation(DimensionOperation::kPROD, *in[0].d[2], *two) : nullptr;
    }
    if (m_options.kind == 6) out.d[2] = b.constant(m_options.intermediate);
    return out;
}
bool GptOssPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* d, int count, int outputs) noexcept {
    if (!d || outputs != 1 || pos < 0 || pos > count ||
        count != (m_options.kind == 0 ? 2 : m_options.kind == 1 ? 8 :
                  m_options.kind == 2 ? 9 :
                  (m_options.kind == 5 || m_options.kind == 6) ? 2 : 1)) return false;
    DataType expected = DataType::kFLOAT;
    if (m_options.kind == 6) expected = DataType::kBF16;
    if (pos < count) {
        if (m_options.kind == 0 && pos == 1) expected = DataType::kINT64;
        if (m_options.kind == 1) {
            if (pos == 1 || pos == 2) expected = DataType::kBF16;
            if (pos >= 3 && pos <= 6) expected = DataType::kINT32;
        }
        if (m_options.kind == 2 && (pos == 3 || pos == 4 || pos == 6 || pos == 7)) expected = DataType::kINT8;
    }
    return d[pos].format == TensorFormat::kLINEAR && d[pos].type == expected;
}
void GptOssPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int count,
    const DynamicPluginTensorDesc*, int) noexcept {
    m_valid = valid(m_options) && in && count ==
        (m_options.kind == 0 ? 2 : m_options.kind == 1 ? 8 :
         m_options.kind == 2 ? 9 :
         (m_options.kind == 5 || m_options.kind == 6) ? 2 : 1);
    if (!m_valid) return;
    const auto& o = m_options; const Dims d = in[0].desc.dims;
    m_valid = rows(d) > 0;
    if (!m_valid) return;
    const int batch = d.d[0], tokens = d.d[1];
    if (o.kind == 0) {
        m_valid = d.d[2] == (o.qHeads + 2 * o.kvHeads) * o.headDim &&
            rows(in[0].desc.dims) == [&] { int n = 1; const auto p = in[1].desc.dims;
                for (int i = 0; i < p.nbDims; ++i) n *= p.d[i]; return n; }();
    } else if (o.kind == 1) {
        const auto k = in[1].desc.dims, v = in[2].desc.dims, table = in[3].desc.dims;
        m_valid = d.d[2] == (o.qHeads + 2 * o.kvHeads) * o.headDim &&
            k.nbDims == 5 && k.d[0] > o.layer && k.d[1] > 0 &&
            shape(k, {k.d[0], k.d[1], o.pageSize, o.kvHeads, o.headDim}) &&
            shape(v, {k.d[0], k.d[1], o.pageSize, o.kvHeads, o.headDim}) &&
            table.nbDims == 2 && table.d[0] == batch && table.d[1] > 0 &&
            shape(in[4].desc.dims, {batch}) && shape(in[5].desc.dims, {batch}) &&
            shape(in[6].desc.dims, {batch}) && shape(in[7].desc.dims, {o.qHeads}) &&
            (o.prefill || tokens == 1);
    } else if (o.kind == 2) {
        m_valid = d.d[2] == o.hidden && shape(in[1].desc.dims, {o.experts, o.hidden}) &&
            shape(in[2].desc.dims, {o.experts}) &&
            shape(in[3].desc.dims, {o.experts, 2 * o.intermediate, o.hidden / 32, 16}) &&
            shape(in[4].desc.dims, {o.experts, 2 * o.intermediate, o.hidden / 32}) &&
            shape(in[5].desc.dims, {o.experts, 2 * o.intermediate}) &&
            shape(in[6].desc.dims, {o.experts, o.hidden, o.intermediate / 32, 16}) &&
            shape(in[7].desc.dims, {o.experts, o.hidden, o.intermediate / 32}) &&
            shape(in[8].desc.dims, {o.experts, o.hidden});
    } else if (o.kind == 3) {
        m_valid = d.d[2] == o.hidden;
    } else if (o.kind == 5) {
        m_valid = d.d[2] == o.hidden && shape(in[1].desc.dims, {o.hidden});
    } else if (o.kind == 6) {
        m_valid = rows(d) == 1 && d.d[2] == o.hidden &&
            (o.qHeads > 0
                ? shape(in[1].desc.dims,
                    {(o.qHeads + 2 * o.kvHeads) * o.headDim, o.hidden})
                : shape(in[1].desc.dims, {o.intermediate, 2 * o.hidden}));
    } else {
        m_valid = d.nbDims == 3 && d.d[0] > 0 && d.d[1] > 0 && d.d[2] > 0;
    }
}
size_t GptOssPlugin::getWorkspaceSize(const PluginTensorDesc* in, int,
    const PluginTensorDesc*, int) const noexcept {
    if (m_options.kind == 3 && m_options.bf16Communication && in &&
        in[0].dims.nbDims == 3 && in[0].dims.d[0] > 0 &&
        in[0].dims.d[1] >= 128 && in[0].dims.d[2] == m_options.hidden) {
        const size_t count = size_t(in[0].dims.d[0]) * in[0].dims.d[1] * in[0].dims.d[2];
        return count <= std::numeric_limits<size_t>::max() / (2 * sizeof(uint16_t))
            ? count * 2 * sizeof(uint16_t) : 0;
    }
    if (m_options.kind == 1 && !m_options.prefill && in &&
        in[0].dims.nbDims == 3 && in[0].dims.d[0] > 0 && in[0].dims.d[1] == 1)
        return size_t(in[0].dims.d[0]) * m_options.qHeads * 64 *
            (2 + 128) * sizeof(float);
    if (m_options.kind == 4) {
        if (!in || in[0].dims.nbDims != 3 || in[0].dims.d[0] <= 0 ||
            in[0].dims.d[1] <= 0 || in[0].dims.d[2] <= 0) return 0;
        const size_t count = size_t(in[0].dims.d[0]) * in[0].dims.d[1] * in[0].dims.d[2];
        return count <= std::numeric_limits<size_t>::max() / (2 * sizeof(float))
            ? count * 2 * sizeof(float) : 0;
    }
    if (m_options.kind != 2) return 0;
    const int tokens = rows(in[0].dims);
    return std::max(GptOssMoeWorkspace(tokens, m_options),
        GptOssMarlin::Workspace(tokens, m_options));
}
int GptOssPlugin::enqueue(const PluginTensorDesc* d, const PluginTensorDesc*,
    const void* const* in, void* const* out, void* workspace, cudaStream_t stream) noexcept {
    if (!m_valid || !d || !in || !out || rows(d[0].dims) <= 0) return 1;
    const int n = rows(d[0].dims); cudaError_t status;
    if (m_options.kind == 0) status = RunGptOssRope((const float*)in[0], (const std::int64_t*)in[1],
        (float*)out[0], n, m_options, stream);
    else if (m_options.kind == 1) status = RunGptOssAttention(in, (float*)out[0], workspace,
        d[0].dims.d[0], d[0].dims.d[1], d[3].dims.d[1], d[1].dims.d[1], m_options, stream);
    else if (m_options.kind == 3) {
        const size_t elements = size_t(d[0].dims.d[0]) * d[0].dims.d[1] * d[0].dims.d[2];
        static const bool useBf16 = [] {
            const char* flag = std::getenv("GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE");
            return flag && std::strcmp(flag, "1") == 0;
        }();
        if (useBf16 && m_options.bf16Communication && n >= 128 && workspace)
            status = GptOssTpAllReduceBf16(static_cast<const float*>(in[0]),
                static_cast<float*>(out[0]), workspace, elements,
                m_options.tpRank, stream);
        else
            status = GptOssTpAllReduce(static_cast<const float*>(in[0]),
                static_cast<float*>(out[0]), elements, m_options.tpRank, stream);
    }
    else if (m_options.kind == 4) {
        const size_t rows = size_t(d[0].dims.d[0]) * d[0].dims.d[1];
        const size_t localVocab = d[0].dims.d[2];
        status = GptOssTpAllGather(static_cast<const float*>(in[0]),
            static_cast<float*>(out[0]), static_cast<float*>(workspace),
            rows * localVocab, static_cast<int>(rows), static_cast<int>(localVocab),
            m_options.tpRank, stream);
    }
    else if (m_options.kind == 5) status = RunGptOssRmsNorm(
        static_cast<const float*>(in[0]), static_cast<const float*>(in[1]),
        static_cast<float*>(out[0]), n, m_options.hidden, m_options.epsilon,
        m_options.hidden >= 1024 ? 1024 : 256, stream);
    else if (m_options.kind == 6) status = RunGptOssDecodeGemv(
        in[0], in[1], out[0], m_options.intermediate, m_options.hidden,
        m_options.qHeads > 0 ? m_options.hidden : 2 * m_options.hidden,
        m_options.qHeads > 0 ? 0 : m_options.tpRank * m_options.hidden,
        m_options.qHeads, m_options.kvHeads, m_options.headDim,
        m_options.tpRank, stream);
    else if (!workspace) status = cudaErrorInvalidValue;
    else {
        // Some TensorRT execution paths invoke enqueue without initialize().
        // Construct the per-plugin state lazily so optimized MoE dispatch is
        // still available for those engines.
        if (!m_marlin) {
            try { m_marlin.reset(new GptOssMarlin(m_options)); }
            catch (...) { reportMarlinFallback(false, n); }
        }
        status = m_marlin
            ? m_marlin->Run(in, (float*)out[0], workspace, n, stream)
                          : cudaErrorNotSupported;
        if (status == cudaErrorNotSupported) {
            reportMarlinFallback(m_marlin != nullptr, n);
            status = RunGptOssMoe(in, (float*)out[0], workspace, n, m_options, stream);
        }
    }
    if (status != cudaSuccess) {
        static std::atomic<int> failures{0};
        if (failures.fetch_add(1) < 16) {
            int device = -1;
            cudaGetDevice(&device);
            std::fprintf(stderr,
                "GPT-OSS plugin execution failed: kind=%d rank=%d device=%d rows=%d CUDA=%d (%s)\n",
                m_options.kind, m_options.tpRank, device, n, int(status),
                cudaGetErrorString(status));
        }
    }
    return status == cudaSuccess ? 0 : 1;
}
DataType GptOssPlugin::getOutputDataType(int, const DataType*, int) const noexcept {
    return m_options.kind == 6 ? DataType::kBF16 : DataType::kFLOAT;
}
const char* GptOssPlugin::getPluginType() const noexcept { return kName; }
const char* GptOssPlugin::getPluginVersion() const noexcept { return kVersion; }
int GptOssPlugin::getNbOutputs() const noexcept { return 1; }
int GptOssPlugin::initialize() noexcept {
    if (!m_valid) return 1;
    if ((m_options.kind == 3 || m_options.kind == 4) && !m_tpInitialized) {
        const auto status = GptOssTpAcquire();
        if (status != cudaSuccess) return 1;
        m_tpInitialized = true;
    }
    if (m_options.kind == 2 && !m_marlin) {
        try { m_marlin.reset(new GptOssMarlin(m_options)); }
        catch (...) { return 1; }
    }
    return 0;
}
void GptOssPlugin::terminate() noexcept {
    m_marlin.reset();
    if (m_tpInitialized) { GptOssTpRelease(); m_tpInitialized = false; }
}
size_t GptOssPlugin::getSerializationSize() const noexcept { return sizeof(m_options); }
void GptOssPlugin::serialize(void* data) const noexcept { std::memcpy(data, &m_options, sizeof(m_options)); }
void GptOssPlugin::destroy() noexcept { terminate(); delete this; }
void GptOssPlugin::setPluginNamespace(const char* ns) noexcept { m_namespace = ns ? ns : ""; }
const char* GptOssPlugin::getPluginNamespace() const noexcept { return m_namespace.c_str(); }
namespace {
class Creator final : public IPluginCreator {
    std::string ns; PluginFieldCollection fields{};
public:
    const char* getPluginName() const noexcept override { return kName; }
    const char* getPluginVersion() const noexcept override { return kVersion; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &fields; }
    IPluginV2* createPlugin(const char*, const PluginFieldCollection*) noexcept override { return nullptr; }
    IPluginV2* deserializePlugin(const char*, const void* data, size_t size) noexcept override {
        if (!data || size != sizeof(GptOssOptions)) return nullptr;
        GptOssOptions o; std::memcpy(&o, data, size); if (!valid(o)) return nullptr;
        auto* p = new (std::nothrow) GptOssPlugin(o); if (p) p->setPluginNamespace(ns.c_str()); return p;
    }
    void setPluginNamespace(const char* value) noexcept override { ns = value ? value : ""; }
    const char* getPluginNamespace() const noexcept override { return ns.c_str(); }
};
}
bool EnsureGptOssPluginRegistered() {
    static Creator creator; static std::once_flag once; static bool registered = false;
    std::call_once(once, [&] { auto* r = getPluginRegistry(); registered = r && r->registerCreator(creator, ""); });
    return registered;
}
}

#if defined(_WIN32)
#define GPT_OSS_EXPORT __declspec(dllexport)
#else
#define GPT_OSS_EXPORT __attribute__((visibility("default")))
#endif
extern "C" GPT_OSS_EXPORT const char* GarnetOperatorPluginManifest() {
#ifdef GARNET_GPT_OSS_ENABLE_NCCL
    return R"({"id":"gpt_oss","module":"garnet_gpt_oss","abi":1,"version":"0.8.0","backend":"tensorrt","operators":["gpt_oss_round_bf16","gpt_oss_apply_yarn_rope_packed","gpt_oss_paged_attention","gpt_oss_moe_mxfp4","gpt_oss_tp_all_reduce","gpt_oss_tp_all_gather","gpt_oss_rms_norm","gpt_oss_decode_gemv"]})";
#else
    return R"({"id":"gpt_oss","module":"garnet_gpt_oss","abi":1,"version":"0.8.0","backend":"tensorrt","operators":["gpt_oss_round_bf16","gpt_oss_apply_yarn_rope_packed","gpt_oss_paged_attention","gpt_oss_moe_mxfp4","gpt_oss_rms_norm","gpt_oss_decode_gemv"]})";
#endif
}
extern "C" GPT_OSS_EXPORT int GarnetRegisterOperatorPlugin() { return Garnet::EnsureGptOssPluginRegistered() ? 1 : 0; }
extern "C" GPT_OSS_EXPORT nvinfer1::IPluginV2DynamicExt* GarnetCreateOperatorPlugin(
    const void* opaqueOptions, size_t size) {
    auto* options = static_cast<const Garnet::GptOssOptions*>(opaqueOptions);
    if (!options || size != sizeof(Garnet::GptOssOptions) || !Garnet::valid(*options)) return nullptr;
    return new (std::nothrow) Garnet::GptOssPlugin(*options);
}
