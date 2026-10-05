// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_plugin.h"
#include <NvInferPlugin.h>
#include <cstring>
#include <mutex>
#include <new>
#include <limits>

using namespace nvinfer1;
namespace Garnet {
namespace {
constexpr const char* kName = "GarnetGptOss";
constexpr const char* kVersion = "1";
bool valid(const GptOssOptions& o) {
    if (o.kind < 0 || o.kind > 2) return false;
    if (o.kind == 2) return o.hidden > 0 && o.intermediate > 0 &&
        o.hidden % 32 == 0 && o.intermediate % 32 == 0 && o.experts > 0 &&
        o.experts <= 256 && o.topK > 0 && o.topK <= 8 && o.topK <= o.experts && o.limit > 0;
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
    return out;
}
bool GptOssPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* d, int count, int outputs) noexcept {
    if (!d || outputs != 1 || pos < 0 || pos > count ||
        count != (m_options.kind == 0 ? 2 : m_options.kind == 1 ? 8 : 9)) return false;
    DataType expected = DataType::kFLOAT;
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
    m_valid = valid(m_options) && in && count == (m_options.kind == 0 ? 2 : m_options.kind == 1 ? 8 : 9);
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
    } else {
        m_valid = d.d[2] == o.hidden && shape(in[1].desc.dims, {o.experts, o.hidden}) &&
            shape(in[2].desc.dims, {o.experts}) &&
            shape(in[3].desc.dims, {o.experts, 2 * o.intermediate, o.hidden / 32, 16}) &&
            shape(in[4].desc.dims, {o.experts, 2 * o.intermediate, o.hidden / 32}) &&
            shape(in[5].desc.dims, {o.experts, 2 * o.intermediate}) &&
            shape(in[6].desc.dims, {o.experts, o.hidden, o.intermediate / 32, 16}) &&
            shape(in[7].desc.dims, {o.experts, o.hidden, o.intermediate / 32}) &&
            shape(in[8].desc.dims, {o.experts, o.hidden});
    }
}
size_t GptOssPlugin::getWorkspaceSize(const PluginTensorDesc* in, int,
    const PluginTensorDesc*, int) const noexcept {
    return m_options.kind == 2 ? GptOssMoeWorkspace(rows(in[0].dims), m_options) : 0;
}
int GptOssPlugin::enqueue(const PluginTensorDesc* d, const PluginTensorDesc*,
    const void* const* in, void* const* out, void* workspace, cudaStream_t stream) noexcept {
    if (!m_valid || !d || !in || !out || rows(d[0].dims) <= 0) return 1;
    const int n = rows(d[0].dims); cudaError_t status;
    if (m_options.kind == 0) status = RunGptOssRope((const float*)in[0], (const std::int64_t*)in[1],
        (float*)out[0], n, m_options, stream);
    else if (m_options.kind == 1) status = RunGptOssAttention(in, (float*)out[0],
        d[0].dims.d[0], d[0].dims.d[1], d[3].dims.d[1], d[1].dims.d[1], m_options, stream);
    else status = workspace ? RunGptOssMoe(in, (float*)out[0], workspace, n, m_options, stream) : cudaErrorInvalidValue;
    return status == cudaSuccess ? 0 : 1;
}
DataType GptOssPlugin::getOutputDataType(int, const DataType*, int) const noexcept { return DataType::kFLOAT; }
const char* GptOssPlugin::getPluginType() const noexcept { return kName; }
const char* GptOssPlugin::getPluginVersion() const noexcept { return kVersion; }
int GptOssPlugin::getNbOutputs() const noexcept { return 1; }
int GptOssPlugin::initialize() noexcept { return m_valid ? 0 : 1; }
void GptOssPlugin::terminate() noexcept {}
size_t GptOssPlugin::getSerializationSize() const noexcept { return sizeof(m_options); }
void GptOssPlugin::serialize(void* data) const noexcept { std::memcpy(data, &m_options, sizeof(m_options)); }
void GptOssPlugin::destroy() noexcept { delete this; }
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
    return R"({"id":"gpt_oss","module":"garnet_gpt_oss","abi":1,"version":"0.1.0","backend":"tensorrt","operators":["gpt_oss_round_bf16","gpt_oss_apply_yarn_rope_packed","gpt_oss_paged_attention","gpt_oss_moe_mxfp4"]})";
}
extern "C" GPT_OSS_EXPORT int GarnetRegisterOperatorPlugin() { return Garnet::EnsureGptOssPluginRegistered() ? 1 : 0; }
extern "C" GPT_OSS_EXPORT nvinfer1::IPluginV2DynamicExt* GarnetCreateOperatorPlugin(
    const void* opaqueOptions, size_t size) {
    auto* options = static_cast<const Garnet::GptOssOptions*>(opaqueOptions);
    if (!options || size != sizeof(Garnet::GptOssOptions) || !Garnet::valid(*options)) return nullptr;
    return new (std::nothrow) Garnet::GptOssPlugin(*options);
}
