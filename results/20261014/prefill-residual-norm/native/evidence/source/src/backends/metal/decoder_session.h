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
    };
    std::unique_ptr<MetalContext> gpu_;
    std::vector<X::Value> owners_;
    std::vector<State> states_;
    MetalCompiledGraph* head_=nullptr;
    uint64_t headGeneration_=0;
    MetalMatrix hidden_;
    MetalMatrix prediction_;
    std::vector<int64_t> hiddenShape_;
    size_t kvInplaceUpdates_=0;
    size_t joinedPrefillPairs_=0;
    size_t constantHeadDispatches_=0;
    size_t residualNormDispatches_=0;
    size_t packedPrefillNormDispatches_=0;
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
    size_t LastResidualNormDispatches() const {return residualNormDispatches_;}
    size_t LastPackedPrefillNormDispatches() const {return packedPrefillNormDispatches_;}
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
        APISET().AddProp("residual_norm_dispatches",&MetalDecoderSession::LastResidualNormDispatches);
        APISET().AddProp("packed_prefill_norm_dispatches",&MetalDecoderSession::LastPackedPrefillNormDispatches);
        APISET().AddProp("prepare_ms",&MetalDecoderSession::PrepareMs);
        APISET().AddProp("encode_ms",&MetalDecoderSession::EncodeMs);
        APISET().AddProp("gpu_ms",&MetalDecoderSession::GPUMs);
        APISET().AddProp("wait_ms",&MetalDecoderSession::WaitMs);
        APISET().AddProp("last_inference_ms",&MetalDecoderSession::LastInferenceMs);
    END_PACKAGE
};
}
