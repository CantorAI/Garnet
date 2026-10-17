// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "metal_context.h"
#include "resident_workspace_plan.h"
#include "safetensors_index.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
namespace Garnet {
class MetalCompiledGraph {
    friend class MetalDecoderSession;
    struct Slot {MetalMatrix matrix;size_t rows=0,columns=0;std::vector<int64_t> shape;MetalBF16Weight bf16;};
    enum class Kind {MatMul,BF16MatVec,Norm,Silu,Add,Bias,Mask,Conv,Delta,Slice,Prepare,GateNorm,Concat,NormRope,KVUpdate,KVChunk,AttentionChunk,Scores,AttentionContext,Vision,Layout,DeltaSplit,ConvChunk,DeltaChunk,ExpandBF16Weight};
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
    size_t visionParallelDispatches_=0;
    size_t visionPrefetchDispatches_=0,visionVectorDispatches_=0,mlpGraphDispatches_=0;
    bool resetKVPrefix_[2]={false,false};
    bool ready_=false,bf16Projections_=false;
    mutable std::mutex mutex_;
    SafeTensorsIndex index_;
    size_t Weight(const std::string&);
    size_t BF16Weight(const std::string&,bool visionPrefill=false,bool rawOnly=false);
    size_t Allocate(size_t,size_t);
    size_t LowerVision(size_t input,const X::Value& attributes);
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    bool lazyDefaultPlanReady_=false;std::vector<size_t> lazyDefaultSlots_;
    size_t lazyRetiredBytes_=0,lazyRehydratedBytes_=0;
#endif
    size_t residentAvoidedStorageBytes_=0;
    bool privateStorage_=false;
    size_t privateExecutionRuns_=0,scopedExecutionRuns_=0;
    std::unordered_set<size_t> materializedWeightSlots_,rawVisionWeightSlots_;
    size_t weightExpansionDispatches_=0;
    std::map<std::pair<size_t,size_t>,std::vector<MetalMatrix>> graphWorkspacePool_;
    size_t graphWorkspaceBoundSlots_=0;
#ifdef GARNET_METAL_GRAPH_WORKSPACE_TEST_FAULTS
    int64_t graphWorkspaceTestFailAfter_=-1,graphWorkspaceTestEncodeOp_=-1;
#endif
    void EnsureGraphWorkspace();
    bool workspacePlanReady_=false;
    std::vector<WorkspaceBinding> workspacePlan_;
    void BuildWorkspacePlan();
#ifdef GARNET_METAL_LAZY_DEFAULTS_TEST_FAULTS
    int64_t lazyTestFailAfter_=-1;
#endif
    void RetireResidentDefaults();
    void MaterializeDefault(size_t slot);
    void Clear();
    void PrepareInputs(X::Value,const std::unordered_set<size_t>& skip={});
    void Encode(MetalContext&,bool packedOutput=true,const MetalMatrix* greedyOutput=nullptr);
    void EncodeGreedy(MetalContext&,const MetalMatrix& prediction);
    bool CanUpdateKVInPlace(size_t part) const;
    bool ComputeResetKVPrefix(size_t part) const;
    bool CanResetKVPrefix(size_t part) const {return part>=1&&part<=2&&resetKVPrefix_[part-1];}
    bool SwapOutputStorage(size_t part,MetalMatrix& replacement);
    bool BuildInternal(X::Value,X::Value,const std::string&,std::unordered_map<std::string,SharedWeight>,bool bf16Projections=false,bool residentDefaults=false);
    bool BuildInternalUnscoped(X::Value,X::Value,const std::string&,std::unordered_map<std::string,SharedWeight>,bool,bool);
    X::Value RunUnscoped(X::Value);
    bool BuildSharedInternal(X::Value,X::Value,const std::string&,X::Value,bool);
public:
    bool Build(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint);
    bool BuildBF16Resident(X::Value,X::Value,const std::string&);
    bool BuildBF16Private(X::Value,X::Value,const std::string&);
    bool BuildBF16PrivateResident(X::Value,X::Value,const std::string&);
    bool PrivateStorage() const;
    bool BuildPrivate(X::Value,X::Value,const std::string&);
    size_t PrivateExecutionRuns() const;
    size_t ScopedExecutionRuns() const;
    bool BuildSharedResident(X::Value,X::Value,const std::string&,X::Value);
    size_t ResidentAvoidedStorageBytes() const;
    std::string WorkspacePlanJson();
    size_t GraphWorkspaceBytes() const;
#ifdef GARNET_METAL_GRAPH_WORKSPACE_TEST_FAULTS
    void TestGraphWorkspaceFailAfter(int64_t);
    void TestGraphWorkspaceEncodeOp(int64_t);
#endif
    size_t GraphWorkspaceCount() const;
    size_t GraphWorkspaceSlots() const;
    size_t GraphWorkspaceVisionSlots() const;
    size_t MaterializedWeightSlots() const;
    size_t PooledWeightSlots() const;
    size_t RawVisionWeightBytes() const;
    size_t WeightExpansionDispatches() const;
    bool BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source);
    X::Value Run(X::Value inputs);
    void Close();
    std::vector<MetalResourceReference> MemoryReferences();
    size_t LazyRetiredBytes() const;
    size_t LazyRehydratedBytes() const;
    size_t LazyPlannedSlots() const;
#ifdef GARNET_METAL_LAZY_DEFAULTS_TEST_FAULTS
    void TestLazyFailAfter(int64_t n);
#endif
    bool Ready() const {return ready_;}
    double CompileMs() const {return compileMs_;}
    double LastInferenceMs() const {return inferenceMs_;}
    size_t OperationCount() const {return ops_.size();}
    size_t LastVisionGlobalAlphaDispatches() const {return visionGlobalAlphaDispatches_;}
    size_t LastVisionParallelDispatches() const {return visionParallelDispatches_;}
    size_t LastVisionPrefetchDispatches() const {return visionPrefetchDispatches_;}
    size_t LastVisionVectorDispatches() const {return visionVectorDispatches_;}
    size_t LastMlpGraphDispatches() const {return mlpGraphDispatches_;}
    BEGIN_PACKAGE(MetalCompiledGraph)
        APISET().AddFunc<3>("build",&MetalCompiledGraph::Build);
        APISET().AddFunc<3>("build_bf16",&MetalCompiledGraph::BuildBF16);
        APISET().AddFunc<3>("build_bf16_private",&MetalCompiledGraph::BuildBF16Private);
        APISET().AddFunc<3>("build_bf16_private_resident",&MetalCompiledGraph::BuildBF16PrivateResident);
        APISET().AddProp("private_storage",&MetalCompiledGraph::PrivateStorage);
        APISET().AddFunc<3>("build_private",&MetalCompiledGraph::BuildPrivate);
        APISET().AddProp("private_execution_runs",&MetalCompiledGraph::PrivateExecutionRuns);
        APISET().AddProp("scoped_execution_runs",&MetalCompiledGraph::ScopedExecutionRuns);
        APISET().AddFunc<3>("build_bf16_resident",&MetalCompiledGraph::BuildBF16Resident);
        APISET().AddFunc<4>("build_shared_resident",&MetalCompiledGraph::BuildSharedResident);
#ifdef GARNET_METAL_GRAPH_WORKSPACE_TEST_FAULTS
        APISET().AddFunc<1>("_test_graph_workspace_fail_after",&MetalCompiledGraph::TestGraphWorkspaceFailAfter);
        APISET().AddFunc<1>("_test_graph_workspace_encode_op",&MetalCompiledGraph::TestGraphWorkspaceEncodeOp);
#endif
        APISET().AddProp("graph_workspace_allocated_bytes",&MetalCompiledGraph::GraphWorkspaceBytes);
        APISET().AddProp("graph_workspace_buffer_count",&MetalCompiledGraph::GraphWorkspaceCount);
        APISET().AddProp("vision_materialized_weight_slots",&MetalCompiledGraph::MaterializedWeightSlots);
        APISET().AddProp("vision_pooled_weight_slots",&MetalCompiledGraph::PooledWeightSlots);
        APISET().AddProp("vision_raw_weight_payload_bytes",&MetalCompiledGraph::RawVisionWeightBytes);
        APISET().AddProp("vision_weight_expansion_dispatches",&MetalCompiledGraph::WeightExpansionDispatches);
        APISET().AddProp("graph_workspace_vision_slots",&MetalCompiledGraph::GraphWorkspaceVisionSlots);
        APISET().AddProp("graph_workspace_bound_slots",&MetalCompiledGraph::GraphWorkspaceSlots);
        APISET().AddFunc<0>("workspace_plan_json",&MetalCompiledGraph::WorkspacePlanJson);
        APISET().AddProp("resident_avoided_storage_bytes",&MetalCompiledGraph::ResidentAvoidedStorageBytes);
        APISET().AddFunc<4>("build_shared",&MetalCompiledGraph::BuildShared);
        APISET().AddFunc<1>("run",&MetalCompiledGraph::Run);
        APISET().AddFunc<0>("close",&MetalCompiledGraph::Close);
#ifdef GARNET_METAL_LAZY_DEFAULTS_TEST_FAULTS
        APISET().AddFunc<1>("_test_lazy_fail_after",&MetalCompiledGraph::TestLazyFailAfter);
#endif
        APISET().AddProp("lazy_default_retired_bytes",&MetalCompiledGraph::LazyRetiredBytes);
        APISET().AddProp("lazy_default_rehydrated_bytes",&MetalCompiledGraph::LazyRehydratedBytes);
        APISET().AddProp("lazy_default_planned_slots",&MetalCompiledGraph::LazyPlannedSlots);
        APISET().AddProp("ready",&MetalCompiledGraph::Ready);
        APISET().AddProp("compile_ms",&MetalCompiledGraph::CompileMs);
        APISET().AddProp("last_inference_ms",&MetalCompiledGraph::LastInferenceMs);
        APISET().AddProp("operation_count",&MetalCompiledGraph::OperationCount);
        APISET().AddProp("vision_global_alpha_dispatches",&MetalCompiledGraph::LastVisionGlobalAlphaDispatches);
        APISET().AddProp("vision_parallel_dispatches",&MetalCompiledGraph::LastVisionParallelDispatches);
        APISET().AddProp("vision_prefetch_dispatches",&MetalCompiledGraph::LastVisionPrefetchDispatches);
        APISET().AddProp("vision_vector_dispatches",&MetalCompiledGraph::LastVisionVectorDispatches);
        APISET().AddProp("mlp_prefill_graph_dispatches",&MetalCompiledGraph::LastMlpGraphDispatches);
    END_PACKAGE
};
}
