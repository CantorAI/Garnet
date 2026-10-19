// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_plugin.h"
#include "gpt_oss_peer_group.h"
#include "gpt_oss_profile_validation.h"
#include "gpt_oss_tp_workspace.h"
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
#include "gpt_oss_flash_prefill.h"
#endif
#include <NvInferPlugin.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <limits>

using namespace nvinfer1;
namespace Garnet {
namespace {
using namespace GptOssProfileValidation;
constexpr const char* kName = "GarnetGptOss";
void reportMarlinFallback(bool initialized, int tokens) {
    static std::atomic<int> count{0};
    if (std::getenv("GARNET_GPT_OSS_DEBUG_MARLIN") && count.fetch_add(1) < 16)
        std::fprintf(stderr, "GPT-OSS Marlin: fallback (plugin initialized=%d, rows=%d)\n", initialized, tokens);
}
// Version11 adds opt-in bounded token-local prefill MoE scratch to V10 decode
// BF16 reservation. Old workspace contracts must rebuild.
// V12 adds an explicit runtime owner/stream contract; old engines rebuild.
// V13 requires deterministic script-handle release; V14 adds the composite
// intermediate-shard MoE + resident BF16 peer-reduction contract.
constexpr const char* kVersion = "14";
bool valid(const GptOssOptions& o) {
    if (o.kind < 0 || o.kind > 8) return false;
    if(o.marlinPrepacked!=0&&(o.marlinPrepacked!=1||(o.kind!=2&&o.kind!=8)||
       o.hidden>16384||o.intermediate>65536||(o.tpRank>=0&&!o.expertWeightsSharded)))return false;
    if (o.expertWeightsSharded != 0 &&
        (o.expertWeightsSharded != 1 || o.kind != 2 || o.tpRank < 0 || o.tpRank > 1 || o.experts < 2))
        return false;
    if (o.bf16Communication < 0 || o.bf16Communication > 1 ||
        (o.bf16Communication && o.kind != 3)) return false;
    if (o.kind == 3 && o.prefill != 0 && o.prefill != 1) return false;
    if (o.kind == 3 || o.kind == 4) return o.hidden > 0 && o.tpRank >= 0 && o.tpRank < 2;
    if (o.kind == 5) return o.hidden > 0 && o.epsilon > 0.f;
    if (o.kind == 7) return o.hidden > 0 && o.hidden <= (1 << 23) &&
        (o.tpRank == 0 || o.tpRank == 1);
    if (o.kind == 6) return o.hidden > 0 && o.hidden % 32 == 0 &&
        o.intermediate > 0 && o.intermediate <= 16384 &&
        (o.tpRank == 0 || o.tpRank == 1) &&
        (o.qHeads == 0 || (o.qHeads > 0 && o.kvHeads > 0 &&
         o.headDim > 0 && o.qHeads % 2 == 0 && o.kvHeads % 2 == 0 &&
         o.intermediate == (o.qHeads / 2 + o.kvHeads) * o.headDim));
    if (o.kind == 8 && (o.peerRank < 0 || o.peerRank > 1 || o.prefill != 1 ||
        o.tpRank != -1 || o.expertWeightsSharded != 0)) return false;
    if (o.kind == 2 || o.kind == 8) return o.hidden > 0 && o.intermediate > 0 &&
        (o.prefill == 0 || o.prefill == 1) &&
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
bool dynamicProfileDebugEnabled() noexcept {
    const char* legacy = std::getenv("GARNET_TRT_DYNAMIC_PROFILE_DEBUG");
    const char* scoped = std::getenv("GARNET_GPT_OSS_DYNAMIC_PROFILE_DEBUG");
    return (legacy && legacy[0] != '\0' && legacy[0] != '0') ||
        (scoped && std::strcmp(scoped, "1") == 0);
}
void reportInitializeFailure(const GptOssOptions& options, const char* stage,
    cudaError_t status) noexcept {
    static std::atomic<int> reported{0};
    if (!dynamicProfileDebugEnabled() ||
        reported.fetch_add(1, std::memory_order_relaxed) >= 32) return;
    std::fprintf(stderr,
        "GPT-OSS plugin initialize failed stage=%s kind=%d rank=%d prefill=%d status=%d cuda=%s\n",
        stage ? stage : "unknown", options.kind, options.tpRank, options.prefill,
        static_cast<int>(status), cudaGetErrorString(status));
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
    if (m_options.kind == 7) out.d[2] = b.constant(2);
    return out;
}
bool GptOssPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* d, int count, int outputs) noexcept {
    if (!d || outputs != 1 || pos < 0 || pos > count ||
        count != (m_options.kind == 0 ? 2 : m_options.kind == 1 ? 8 :
                  (m_options.kind == 2 || m_options.kind == 8) ? 9 :
                  (m_options.kind == 5 || m_options.kind == 6) ? 2 : 1)) return false;
    DataType expected = DataType::kFLOAT;
    if (m_options.kind == 6) expected = DataType::kBF16;
    if (pos < count) {
        if (m_options.kind == 0 && pos == 1) expected = DataType::kINT64;
        if (m_options.kind == 1) {
            if (pos == 1 || pos == 2) expected = DataType::kBF16;
            if (pos >= 3 && pos <= 6) expected = DataType::kINT32;
        }
        if ((m_options.kind == 2 || m_options.kind == 8) &&
            (pos == 3 || pos == 4 || pos == 6 || pos == 7)) expected = DataType::kINT8;
    }
    return d[pos].format == TensorFormat::kLINEAR && d[pos].type == expected;
}
void GptOssPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int count,
    const DynamicPluginTensorDesc*, int) noexcept {
    static std::atomic<int> rejectedProfileDebugCount{0};
    const bool debugDynamicProfile = dynamicProfileDebugEnabled();
    const auto reportDynamicProfile = [&]() noexcept {
        if (!debugDynamicProfile ||
            rejectedProfileDebugCount.fetch_add(1, std::memory_order_relaxed) >= 16) return;
        std::fprintf(stderr, "GPT-OSS dynamic profile: kind=%d rank=%d prefill=%d count=%d valid=%d\n",
            m_options.kind, m_options.tpRank, m_options.prefill, count, m_valid ? 1 : 0);
        if (!in || count < 0 || count > 16) return;
        for (int i = 0; i < count; ++i) {
            const auto printDims = [](const char* label, const Dims& dims) noexcept {
                std::fprintf(stderr, "  %s[", label);
                for (int d = 0; d < dims.nbDims; ++d)
                    std::fprintf(stderr, "%s%lld", d ? "," : "", static_cast<long long>(dims.d[d]));
                std::fprintf(stderr, "]");
            };
            std::fprintf(stderr, "  input=%d type=%d format=%d", i,
                static_cast<int>(in[i].desc.type), static_cast<int>(in[i].desc.format));
            printDims("desc", in[i].desc.dims); printDims("min", in[i].min);
            printDims("max", in[i].max);
            std::fprintf(stderr, "\n");
        }
    };
    m_valid = valid(m_options) && in && count ==
        (m_options.kind == 0 ? 2 : m_options.kind == 1 ? 8 :
         (m_options.kind == 2 || m_options.kind == 8) ? 9 :
         (m_options.kind == 5 || m_options.kind == 6) ? 2 : 1);
    if (!m_valid) { reportDynamicProfile(); return; }
    for (int index = 0; index < count; ++index) {
        if (!validProfileRange(in[index])) { m_valid = false; reportDynamicProfile(); return; }
    }
    const auto& o = m_options;
    if (o.expertWeightsSharded != 0 &&
        (o.expertWeightsSharded != 1 || o.kind != 2 || o.tpRank < 0 || o.tpRank > 1)) {
        m_valid = false; reportDynamicProfile(); return;
    }
    const int activationTokens = in[0].min.nbDims == 3 ?
        static_cast<int>(in[0].min.d[1]) : 0;
    const int activationWidth = in[0].min.nbDims == 3 ?
        static_cast<int>(in[0].min.d[2]) : 0;
    const auto activation = [&](int width) {
        return in[0].min.nbDims == 3 && activationTokens > 0 &&
            profileShape(in[0], {-1, activationTokens, width});
    };
    if (o.kind == 0) {
        const int width = (o.qHeads + 2 * o.kvHeads) * o.headDim;
        m_valid = activation(width) && in[1].min.nbDims == 2 &&
            profileShape(in[1], {-1, activationTokens}) &&
            profileAxisEqual(in[0], 0, in[1], 0) &&
            profileAxisEqual(in[0], 1, in[1], 1);
    } else if (o.kind == 1) {
        const int width = (o.qHeads + 2 * o.kvHeads) * o.headDim;
        const int minBatch = static_cast<int>(in[0].min.d[0]);
        const int maxBatch = static_cast<int>(in[0].max.d[0]);
        const auto& tableMin = in[3].min;
        const auto& tableMax = in[3].max;
        const auto& kMin = in[1].min;
        const auto& vMin = in[2].min;
        m_valid = activation(width) &&
            (o.prefill || activationTokens == 1) &&
            profileShape(in[1], {kMin.d[0], -1, o.pageSize, o.kvHeads, o.headDim}) &&
            profileShape(in[2], {vMin.d[0], -1, o.pageSize, o.kvHeads, o.headDim}) &&
            kMin.d[0] > o.layer && vMin.d[0] == kMin.d[0] &&
            profileAxisEqual(in[1], 1, in[2], 1) &&
            profileAxisEqual(in[1], 0, in[2], 0) &&
            proportionalPageRange(in[1], in[0]) && proportionalPageRange(in[2], in[0]) &&
            profileShape(in[3], {-1, tableMin.d[1]}) &&
            tableMin.nbDims == 2 && tableMax.nbDims == 2 &&
            tableMin.d[0] == minBatch && tableMax.d[0] == maxBatch &&
            tableMin.d[1] > 0 && tableMin.d[1] == tableMax.d[1] &&
            profileAxisEqual(in[3], 0, in[0], 0) &&
            profileShape(in[4], {-1}) && profileShape(in[5], {-1}) &&
            profileShape(in[6], {-1}) &&
            profileAxisEqual(in[4], 0, in[0], 0) &&
            profileAxisEqual(in[5], 0, in[0], 0) &&
            profileAxisEqual(in[6], 0, in[0], 0) &&
            profileShape(in[7], {o.qHeads});
    } else if (o.kind == 2 || o.kind == 8) {
        const int storedExperts = o.expertWeightsSharded ? (o.experts + 1 - o.tpRank) / 2 : o.experts;
        const int upK=(o.hidden+63)/64*64,upN=(2*o.intermediate+127)/128*128;
        const int downK=(o.intermediate+127)/128*128,downN=(o.hidden+63)/64*64;
        const bool quantShapes=o.marlinPrepacked
            ? profileShape(in[3],{storedExperts,upK/32,upN,16}) &&
              profileShape(in[4],{storedExperts,upK/32,upN}) &&
              profileShape(in[6],{storedExperts,downK/32,downN,16}) &&
              profileShape(in[7],{storedExperts,downK/32,downN})
            : profileShape(in[3],{storedExperts,2*o.intermediate,o.hidden/32,16}) &&
              profileShape(in[4],{storedExperts,2*o.intermediate,o.hidden/32}) &&
              profileShape(in[6],{storedExperts,o.hidden,o.intermediate/32,16}) &&
              profileShape(in[7],{storedExperts,o.hidden,o.intermediate/32});
        m_valid = activation(o.hidden) && profileShape(in[1], {o.experts, o.hidden}) &&
            profileShape(in[2], {o.experts}) && quantShapes &&
            profileShape(in[5], {storedExperts, 2 * o.intermediate}) &&
            profileShape(in[8], {storedExperts, o.hidden});
    } else if (o.kind == 3 || o.kind == 7) {
        m_valid = activation(o.hidden);
    } else if (o.kind == 5) {
        m_valid = activation(o.hidden) && profileShape(in[1], {o.hidden});
    } else if (o.kind == 6) {
        m_valid = profileShape(in[0], {1, 1, o.hidden}) &&
            (o.qHeads > 0
                ? profileShape(in[1],
                    {(o.qHeads + 2 * o.kvHeads) * o.headDim, o.hidden})
                : profileShape(in[1], {o.intermediate, 2 * o.hidden}));
    } else {
        m_valid = activationWidth > 0 && activation(activationWidth);
    }
    if (!m_valid) reportDynamicProfile();
}
size_t GptOssPlugin::getWorkspaceSize(const PluginTensorDesc* in, int,
    const PluginTensorDesc*, int) const noexcept {
#ifdef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
    // Reserve independently of the runtime flag so baseline/candidate engines
    // share a safe workspace contract. Build identity includes plugin bytes.
    if(m_options.kind==1 && in && in[0].dims.nbDims==3 && in[3].dims.nbDims==2) {
        const auto flash=GptOssFlashAttentionWorkspace(in[0].dims.d[0],in[0].dims.d[1],in[3].dims.d[1],m_options);
        if(m_options.prefill)return flash;
        if(in[0].dims.d[1]==1)
            return std::max(flash,size_t(in[0].dims.d[0])*m_options.qHeads*64*130*sizeof(float));
    }
#endif
    if (m_options.kind == 3) {
        if (!in || in[0].dims.nbDims != 3) return 0;
        const auto& d = in[0].dims;
        return GptOssTpBf16Workspace(m_options.kind, m_options.bf16Communication,
            m_options.prefill, m_options.tpRank, m_options.hidden, d.nbDims,
            d.d[0], d.d[1], d.d[2]);
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
    if (m_options.kind != 2 && m_options.kind != 8) return 0;
    const int tokens = rows(in[0].dims);
    return std::max(GptOssMoeWorkspace(tokens, m_options),
        GptOssMarlin::Workspace(tokens, m_options));
}
int GptOssPlugin::enqueue(const PluginTensorDesc* d, const PluginTensorDesc*,
    const void* const* in, void* const* out, void* workspace, cudaStream_t stream) noexcept {
    static std::atomic<int> earlyRejectCount{0};
    const bool debugReject = std::getenv("GARNET_TRT_DYNAMIC_PROFILE_DEBUG") != nullptr &&
        earlyRejectCount.fetch_add(1, std::memory_order_relaxed) < 32;
    const auto reject = [&](const char* reason) noexcept {
        if (debugReject) {
            std::fprintf(stderr,
                "GPT-OSS enqueue rejected: kind=%d rank=%d prefill=%d reason=%s valid=%d workspace=%d\n",
                m_options.kind, m_options.tpRank, m_options.prefill, reason,
                m_valid ? 1 : 0, workspace ? 1 : 0);
            const auto printDims = [](int index, const PluginTensorDesc* desc) noexcept {
                if (!desc || index < 0 || index >= 9) return;
                const auto& dims = desc[index].dims;
                std::fprintf(stderr, "  input=%d dims[", index);
                for (int axis = 0; axis < dims.nbDims; ++axis)
                    std::fprintf(stderr, "%s%lld", axis ? "," : "",
                        static_cast<long long>(dims.d[axis]));
                std::fprintf(stderr, "]\n");
            };
            if (d) {
                printDims(0, d);
                if (m_options.kind == 1) {
                    for (int index = 1; index <= 6; ++index) printDims(index, d);
                }
            }
        }
        return 1;
    };
    if (!m_valid) return reject("invalid_plugin_state");
    if (!d || !in || !out) return reject("missing_tensor_pointer");
    if (rows(d[0].dims) <= 0) return reject("invalid_input0_rows");
    if (m_options.kind == 1) {
        const int batch = d[0].dims.d[0];
        if (d[3].dims.nbDims != 2 || d[3].dims.d[0] != batch ||
            d[1].dims.nbDims != 5 || d[2].dims.nbDims != 5 ||
            d[1].dims.d[1] != d[2].dims.d[1] ||
            d[3].dims.d[1] <= 0) return reject("attention_kv_or_table_shape");
        for (int index = 4; index <= 6; ++index) {
            if (d[index].dims.nbDims != 1 || d[index].dims.d[0] != batch)
                return reject("attention_batch_vector_shape");
        }
    }
    const int n = rows(d[0].dims); cudaError_t status;
    if (m_options.kind == 0) status = RunGptOssRope((const float*)in[0], (const std::int64_t*)in[1],
        (float*)out[0], n, m_options, stream);
    else if (m_options.kind == 1) status = RunGptOssAttention(in, (float*)out[0], workspace,
        d[0].dims.d[0], d[0].dims.d[1], d[3].dims.d[1], d[1].dims.d[1], m_options, stream);
    else if (m_options.kind == 3) {
        const size_t elements = size_t(d[0].dims.d[0]) * d[0].dims.d[1] * d[0].dims.d[2];
        static const bool useBf16Prefill = [] {
            const char* flag = std::getenv("GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE");
            return flag && std::strcmp(flag, "1") == 0;
        }();
        static const bool useBf16Decode = [] {
            const char* flag = std::getenv("GARNET_GPT_OSS_BF16_DECODE_ALLREDUCE");
            return flag && std::strcmp(flag, "1") == 0;
        }();
        const auto& dims = d[0].dims;
        if (dims.d[2] != m_options.hidden) return reject("tp_hidden_size");
        const size_t scratch = GptOssTpBf16Workspace(m_options.kind,
            m_options.bf16Communication, m_options.prefill, m_options.tpRank,
            m_options.hidden, dims.nbDims, dims.d[0], dims.d[1], dims.d[2]);
        const bool useBf16 = m_options.prefill ? useBf16Prefill : useBf16Decode;
        if (useBf16 && scratch && !workspace) return reject("tp_bf16_workspace_missing");
        if (useBf16 && scratch)
        {
            static const bool ownedPeer=[] {
                const char* flag=std::getenv("GARNET_GPT_OSS_BF16_PEER_GROUP");
                return flag && std::strcmp(flag,"1")==0;
            }();
            status=ownedPeer?GptOssPeerGroupAllReduce(static_cast<const float*>(in[0]),
                static_cast<float*>(out[0]),elements,m_options.tpRank,m_options.prefill,stream):
                GptOssTpAllReduceBf16(static_cast<const float*>(in[0]),
                static_cast<float*>(out[0]),workspace,elements,m_options.tpRank,stream);
        }
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
    else if (m_options.kind == 7) status = RunGptOssVocabTop1(
        static_cast<const float*>(in[0]), static_cast<float*>(out[0]),
        n, m_options.hidden, m_options.tpRank, stream);
    else if (m_options.kind == 8) {
        const char* enabled=std::getenv("GARNET_GPT_OSS_FUSED_MOE_TP_REDUCE");
        if(!enabled || std::strcmp(enabled,"1")) status=cudaErrorNotSupported;
        else {
            status=cudaSuccess;
            if(!m_marlin) {
                try { m_marlin.reset(new GptOssMarlin(m_options)); }
                catch (...) { status=cudaErrorMemoryAllocation; }
            }
            void* packed=nullptr;
            const size_t elements=size_t(n)*size_t(m_options.hidden);
            if(status==cudaSuccess)status=GptOssPeerGroupPackedBuffer(elements,
                m_options.peerRank,1,stream,&packed);
            if(status==cudaSuccess)status=m_marlin->Run(in,nullptr,workspace,n,stream,packed);
            if(status==cudaSuccess)status=GptOssPeerGroupAllReducePacked(packed,
                static_cast<float*>(out[0]),elements,m_options.peerRank,1,stream);
        }
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
        if (status == cudaErrorNotSupported && !m_options.marlinPrepacked) {
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
    if (!m_valid) {
        reportInitializeFailure(m_options, "invalid_plugin_configuration", cudaErrorInvalidValue);
        return 1;
    }
    if ((m_options.kind == 3 || m_options.kind == 4) && !m_tpInitialized) {
        const auto status = GptOssTpAcquire();
        if (status != cudaSuccess) {
            reportInitializeFailure(m_options, "tp_communicator_acquire", status);
            return 1;
        }
        m_tpInitialized = true;
    }
    if ((m_options.kind == 2 || m_options.kind == 8) && !m_marlin) {
        try { m_marlin.reset(new GptOssMarlin(m_options)); }
        catch (const std::bad_alloc&) {
            reportInitializeFailure(m_options, "marlin_host_allocation", cudaErrorMemoryAllocation);
            return 1;
        }
        catch (const std::exception&) {
            reportInitializeFailure(m_options, "marlin_state_construction", cudaErrorUnknown);
            return 1;
        }
        catch (...) {
            reportInitializeFailure(m_options, "marlin_unknown_exception", cudaErrorUnknown);
            return 1;
        }
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
    return R"({"id":"gpt_oss","module":"garnet_gpt_oss","abi":1,"version":"0.15.0","backend":"tensorrt","operators":["gpt_oss_round_bf16","gpt_oss_apply_yarn_rope_packed","gpt_oss_paged_attention","gpt_oss_moe_mxfp4","gpt_oss_moe_tp_reduce_bf16","gpt_oss_tp_all_reduce","gpt_oss_tp_all_gather","gpt_oss_rms_norm","gpt_oss_decode_gemv","gpt_oss_vocab_top1"]})";
#else
    return R"({"id":"gpt_oss","module":"garnet_gpt_oss","abi":1,"version":"0.15.0","backend":"tensorrt","operators":["gpt_oss_round_bf16","gpt_oss_apply_yarn_rope_packed","gpt_oss_paged_attention","gpt_oss_moe_mxfp4","gpt_oss_moe_tp_reduce_bf16","gpt_oss_rms_norm","gpt_oss_decode_gemv","gpt_oss_vocab_top1"]})";
#endif
}
extern "C" GPT_OSS_EXPORT int GarnetRegisterOperatorPlugin() { return Garnet::EnsureGptOssPluginRegistered() ? 1 : 0; }
extern "C" GPT_OSS_EXPORT nvinfer1::IPluginV2DynamicExt* GarnetCreateOperatorPlugin(
    const void* opaqueOptions, size_t size) {
    auto* options = static_cast<const Garnet::GptOssOptions*>(opaqueOptions);
    if (!options || size != sizeof(Garnet::GptOssOptions) || !Garnet::valid(*options)) return nullptr;
    return new (std::nothrow) Garnet::GptOssPlugin(*options);
}
