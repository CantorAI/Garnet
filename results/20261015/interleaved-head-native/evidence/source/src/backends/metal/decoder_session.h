// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "compiled_graph.h"
#include <array>
namespace Garnet {
// One synchronous request context. Graph owners are retained; request state is
// private to this session, including when graphs are reused by another session.
class MetalDecoderSession {
    struct State {
        std::array<MetalMatrix,2> matrix;
        std::array<std::vector<int64_t>,2> shape;
        std::array<size_t,2> columns;
        size_t rows;
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
        std::array<bool,2> prefixLayout{false,false},zeroOutsideDirty{false,false};
        std::array<size_t,2> dirtyExtent{0,0};
#endif
    };
    std::unique_ptr<MetalContext> gpu_;
    std::vector<X::Value> owners_;
    std::vector<State> states_;
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    std::vector<float> resetZeroSource_;
#endif
    size_t reusedSourceBytes_=0,reusedCopyBytes_=0,reusedCopyStates_=0;
    MetalCompiledGraph* head_=nullptr;
    uint64_t headGeneration_=0;
    MetalMatrix hidden_;
    MetalMatrix prediction_;
    std::vector<int64_t> hiddenShape_;
    size_t kvInplaceUpdates_=0;
    size_t joinedPrefillPairs_=0;
    size_t constantHeadDispatches_=0;
    size_t singleCoderHeadDispatches_=0;
    size_t residualNormDispatches_=0;
    size_t packedPrefillNormDispatches_=0;
    size_t globalHalfAlphaDispatches_=0;
    size_t joinedGlobalMlpBlocks_=0;
    size_t prefillPartialSubmissions_=0;
    size_t fusedHeadArgmaxDispatches_=0;
    size_t screenedHeadDispatches_=0;
    size_t screenedHeadSeedRows_=0;
    size_t interleavedScreenDispatches_=0;
    size_t bodyShapeDispatches_=0,bodyShapeMisses_=0;
    size_t deferredZeroBytes_=0,deferredZeroStates_=0;
    size_t dirtyPrefixResetBytes_=0,dirtyPrefixResetStates_=0,dirtyPrefixResetFallbackStates_=0;
    bool poisoned_=false;
    double inferenceMs_=0,prepareMs_=0,encodeMs_=0,gpuMs_=0,waitMs_=0;
    std::mutex mutex_;
    void Clear();
    void ResetUnlocked();
public:
    bool Build(X::Value graphs,X::Value head);
    X::Value Run(X::Value graphs,X::Value hidden,X::Value positions,
        X::Value slots,X::Value active,int64_t predictionToken);
    void Reset();
    X::Value Snapshot(int64_t layer);
    X::Value Hidden();
    void Close();
    size_t LastKVInplaceUpdates() const {return kvInplaceUpdates_;}
    size_t LastJoinedPrefillPairs() const {return joinedPrefillPairs_;}
    size_t LastConstantHeadDispatches() const {return constantHeadDispatches_;}
    size_t LastSingleCoderHeadDispatches() const {return singleCoderHeadDispatches_;}
    size_t LastFusedHeadArgmaxDispatches() const {return fusedHeadArgmaxDispatches_;}
    size_t LastScreenedHeadDispatches() const {return screenedHeadDispatches_;}
    size_t LastScreenedHeadSeedRows() const {return screenedHeadSeedRows_;}
    size_t LastInterleavedScreenDispatches() const {return interleavedScreenDispatches_;}
    size_t LastBodyShapeDispatches() const {return bodyShapeDispatches_;}
    size_t LastBodyShapeMisses() const {return bodyShapeMisses_;}
    size_t BodyShapePipelineCount() const {return gpu_?gpu_->BodyShapePipelineCount():0;}
    size_t LastDeferredZeroBytes() const {return deferredZeroBytes_;}
    size_t LastDeferredZeroStates() const {return deferredZeroStates_;}
    size_t ReusedZeroSourceBytes() const {return reusedSourceBytes_;}
    size_t LastReusedZeroCopyBytes() const {return reusedCopyBytes_;}
    size_t LastReusedZeroCopyStates() const {return reusedCopyStates_;}
    size_t LastResidualNormDispatches() const {return residualNormDispatches_;}
    size_t LastPackedPrefillNormDispatches() const {return packedPrefillNormDispatches_;}
    size_t LastGlobalHalfAlphaDispatches() const {return globalHalfAlphaDispatches_;}
    size_t LastJoinedGlobalMlpBlocks() const {return joinedGlobalMlpBlocks_;}
    size_t LastPrefillPartialSubmissions() const {return prefillPartialSubmissions_;}
    size_t LastDirtyPrefixResetBytes() const {return dirtyPrefixResetBytes_;}
    size_t LastDirtyPrefixResetStates() const {return dirtyPrefixResetStates_;}
    size_t LastDirtyPrefixResetFallbackStates() const {return dirtyPrefixResetFallbackStates_;}
    double LastInferenceMs() const {return inferenceMs_;}
    double PrepareMs() const {return prepareMs_;}
    double EncodeMs() const {return encodeMs_;}
    double GPUMs() const {return gpuMs_;}
    double WaitMs() const {return waitMs_;}
    BEGIN_PACKAGE(MetalDecoderSession)
        APISET().AddFunc<2>("build",&MetalDecoderSession::Build);
        APISET().AddFunc<6>("run",&MetalDecoderSession::Run);
        APISET().AddFunc<0>("reset",&MetalDecoderSession::Reset);
        APISET().AddFunc<1>("snapshot",&MetalDecoderSession::Snapshot);
        APISET().AddFunc<0>("hidden",&MetalDecoderSession::Hidden);
        APISET().AddFunc<0>("close",&MetalDecoderSession::Close);
        APISET().AddProp("kv_inplace_updates",&MetalDecoderSession::LastKVInplaceUpdates);
        APISET().AddProp("joined_prefill_pairs",&MetalDecoderSession::LastJoinedPrefillPairs);
        APISET().AddProp("constant_head_dispatches",&MetalDecoderSession::LastConstantHeadDispatches);
        APISET().AddProp("single_coder_head_dispatches",&MetalDecoderSession::LastSingleCoderHeadDispatches);
        APISET().AddProp("fused_head_argmax_dispatches",&MetalDecoderSession::LastFusedHeadArgmaxDispatches);
        APISET().AddProp("screened_head_dispatches",&MetalDecoderSession::LastScreenedHeadDispatches);
        APISET().AddProp("screened_head_seed_rows",&MetalDecoderSession::LastScreenedHeadSeedRows);
        APISET().AddProp("interleaved_screen_dispatches",&MetalDecoderSession::LastInterleavedScreenDispatches);
        APISET().AddProp("body_shape_dispatches",&MetalDecoderSession::LastBodyShapeDispatches);
        APISET().AddProp("body_shape_misses",&MetalDecoderSession::LastBodyShapeMisses);
        APISET().AddProp("body_shape_pipeline_count",&MetalDecoderSession::BodyShapePipelineCount);
        APISET().AddProp("deferred_zero_bytes",&MetalDecoderSession::LastDeferredZeroBytes);
        APISET().AddProp("deferred_zero_states",&MetalDecoderSession::LastDeferredZeroStates);
        APISET().AddProp("reused_zero_source_bytes",&MetalDecoderSession::ReusedZeroSourceBytes);
        APISET().AddProp("reused_zero_copy_bytes",&MetalDecoderSession::LastReusedZeroCopyBytes);
        APISET().AddProp("reused_zero_copy_states",&MetalDecoderSession::LastReusedZeroCopyStates);
        APISET().AddProp("residual_norm_dispatches",&MetalDecoderSession::LastResidualNormDispatches);
        APISET().AddProp("packed_prefill_norm_dispatches",&MetalDecoderSession::LastPackedPrefillNormDispatches);
        APISET().AddProp("global_half_alpha_dispatches",&MetalDecoderSession::LastGlobalHalfAlphaDispatches);
        APISET().AddProp("joined_global_mlp_blocks",&MetalDecoderSession::LastJoinedGlobalMlpBlocks);
        APISET().AddProp("prefill_partial_submissions",&MetalDecoderSession::LastPrefillPartialSubmissions);
        APISET().AddProp("dirty_prefix_reset_bytes",&MetalDecoderSession::LastDirtyPrefixResetBytes);
        APISET().AddProp("dirty_prefix_reset_states",&MetalDecoderSession::LastDirtyPrefixResetStates);
        APISET().AddProp("dirty_prefix_reset_fallback_states",&MetalDecoderSession::LastDirtyPrefixResetFallbackStates);
        APISET().AddProp("prepare_ms",&MetalDecoderSession::PrepareMs);
        APISET().AddProp("encode_ms",&MetalDecoderSession::EncodeMs);
        APISET().AddProp("gpu_ms",&MetalDecoderSession::GPUMs);
        APISET().AddProp("wait_ms",&MetalDecoderSession::WaitMs);
        APISET().AddProp("last_inference_ms",&MetalDecoderSession::LastInferenceMs);
    END_PACKAGE
};
}
