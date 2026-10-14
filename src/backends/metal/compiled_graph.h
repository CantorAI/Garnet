// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "metal_context.h"
#include "safetensors_index.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
namespace Garnet {
class MetalCompiledGraph {
    friend class MetalDecoderSession;
    struct Slot {MetalMatrix matrix;size_t rows=0,columns=0;std::vector<int64_t> shape;MetalBF16Weight bf16;};
    enum class Kind {MatMul,BF16MatVec,Norm,Silu,Add,Bias,Mask,Conv,Delta,Slice,Prepare,GateNorm,Concat,NormRope,KVUpdate,KVChunk,AttentionChunk,Scores,AttentionContext,Vision,Layout,DeltaSplit,ConvChunk,DeltaChunk};
    struct Op {Kind kind;size_t a,b,out;float epsilon=0,offset=0;bool transpose=false;
        std::vector<size_t> arguments={},dimensions={};bool contiguousSlots=false;size_t auxiliaryOutput=SIZE_MAX;};
    std::unique_ptr<MetalContext> gpu_;
    std::vector<Slot> slots_;
    std::vector<Op> ops_;
    std::vector<size_t> inputs_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::unordered_map<uint64_t,size_t> tensors_;
    std::unordered_map<std::string,size_t> weights_;
    struct SharedWeight {Slot slot;SafeTensorMetadata metadata;
        std::filesystem::file_time_type modified;uintmax_t bytes;};
    std::unordered_map<std::string,SharedWeight> sharedWeights_,weightRecords_;
    std::unordered_map<uint64_t,std::unordered_map<std::string,size_t>> bindings_;
    std::unordered_map<size_t,std::pair<size_t,size_t>> controlSources_;
    size_t output_=0;
    std::vector<size_t> outputParts_;
    std::vector<std::vector<size_t>> outputStorageGroups_;
    std::vector<uint8_t> residentOps_;
    uint64_t generation_=0;
    double compileMs_=0,inferenceMs_=0;
    size_t visionGlobalAlphaDispatches_=0;
    bool ready_=false,bf16Projections_=false;
    std::mutex mutex_;
    SafeTensorsIndex index_;
    size_t Weight(const std::string&);
    size_t BF16Weight(const std::string&,bool visionPrefill=false);
    size_t Allocate(size_t,size_t);
    size_t LowerVision(size_t input,const X::Value& attributes);
    void Clear();
    void PrepareInputs(X::Value,const std::unordered_set<size_t>& skip={});
    void Encode(MetalContext&,bool packedOutput=true);
    bool CanUpdateKVInPlace(size_t part) const;
    bool SwapOutputStorage(size_t part,MetalMatrix& replacement);
    bool BuildInternal(X::Value,X::Value,const std::string&,std::unordered_map<std::string,SharedWeight>,bool bf16Projections=false);
public:
    bool Build(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source);
    X::Value Run(X::Value inputs);
    void Close();
    bool Ready() const {return ready_;}
    double CompileMs() const {return compileMs_;}
    double LastInferenceMs() const {return inferenceMs_;}
    size_t OperationCount() const {return ops_.size();}
    size_t LastVisionGlobalAlphaDispatches() const {return visionGlobalAlphaDispatches_;}
    BEGIN_PACKAGE(MetalCompiledGraph)
        APISET().AddFunc<3>("build",&MetalCompiledGraph::Build);
        APISET().AddFunc<3>("build_bf16",&MetalCompiledGraph::BuildBF16);
        APISET().AddFunc<4>("build_shared",&MetalCompiledGraph::BuildShared);
        APISET().AddFunc<1>("run",&MetalCompiledGraph::Run);
        APISET().AddFunc<0>("close",&MetalCompiledGraph::Close);
        APISET().AddProp("ready",&MetalCompiledGraph::Ready);
        APISET().AddProp("compile_ms",&MetalCompiledGraph::CompileMs);
        APISET().AddProp("last_inference_ms",&MetalCompiledGraph::LastInferenceMs);
        APISET().AddProp("operation_count",&MetalCompiledGraph::OperationCount);
        APISET().AddProp("vision_global_alpha_dispatches",&MetalCompiledGraph::LastVisionGlobalAlphaDispatches);
    END_PACKAGE
};
}
