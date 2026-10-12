// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "metal_context.h"
#include "safetensors_index.h"
#include <unordered_map>
#include <vector>
#include <mutex>
namespace Garnet {
class MetalCompiledGraph {
    struct Slot {MetalMatrix matrix;size_t rows=0,columns=0;std::vector<int64_t> shape;};
    enum class Kind {MatMul,Norm,Silu,Add,Mask,Conv,Delta,Slice,Prepare,GateNorm,Concat,NormRope,KVUpdate,Scores,AttentionContext,Vision};
    struct Op {Kind kind;size_t a,b,out;float epsilon=0,offset=0;bool transpose=false;
        std::vector<size_t> arguments={},dimensions={};};
    std::unique_ptr<MetalContext> gpu_;
    std::vector<Slot> slots_;
    std::vector<Op> ops_;
    std::vector<size_t> inputs_;
    std::vector<std::vector<int64_t>> inputShapes_;
    std::unordered_map<uint64_t,size_t> tensors_;
    std::unordered_map<std::string,size_t> weights_;
    std::unordered_map<uint64_t,std::unordered_map<std::string,size_t>> bindings_;
    size_t output_=0;
    double compileMs_=0,inferenceMs_=0;
    bool ready_=false;
    std::mutex mutex_;
    SafeTensorsIndex index_;
    size_t Weight(const std::string&);
    size_t Allocate(size_t,size_t);
    size_t LowerVision(size_t input,const X::Value& attributes);
    void Clear();
public:
    bool Build(X::Value graph,X::Value symbols,const std::string& checkpoint);
    X::Value Run(X::Value inputs);
    void Close();
    bool Ready() const {return ready_;}
    double CompileMs() const {return compileMs_;}
    double LastInferenceMs() const {return inferenceMs_;}
    BEGIN_PACKAGE(MetalCompiledGraph)
        APISET().AddFunc<3>("build",&MetalCompiledGraph::Build);
        APISET().AddFunc<1>("run",&MetalCompiledGraph::Run);
        APISET().AddFunc<0>("close",&MetalCompiledGraph::Close);
        APISET().AddProp("ready",&MetalCompiledGraph::Ready);
        APISET().AddProp("compile_ms",&MetalCompiledGraph::CompileMs);
        APISET().AddProp("last_inference_ms",&MetalCompiledGraph::LastInferenceMs);
    END_PACKAGE
};
}
