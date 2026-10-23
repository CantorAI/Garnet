// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_compiled_graph.h"
#include "trt_builder.h"
#include "qwen35_delta_plugin.h"
#include "tensor_helper.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
namespace Garnet {
CUDACompiledGraph::CUDACompiledGraph() {
    static std::atomic<uint64_t> next{0};
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    cacheRoot_=(std::filesystem::temp_directory_path()/
        ("garnet-cuda-graph-"+std::to_string(stamp)+"-"+std::to_string(++next))).string()+"/";
    enginePath_=cacheRoot_+"model.engine";
}
CUDACompiledGraph::~CUDACompiledGraph(){Close();}
void CUDACompiledGraph::CloseUnlocked(){deviceOutput_=X::Value();externalOutput_=false;execution_.reset();TRTBuilder::ReleaseCachedExecutions(cacheRoot_);ready_=false;externalWeightCount_=externalWeightBytes_=joinedProjectionInputs_=nativeFP8WeightCount_=nativeNVFP4WeightCount_=0;weightPrecision_="bf16";}
void CUDACompiledGraph::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDACompiledGraph::Ready() const{std::lock_guard<std::mutex> lock(mutex_);return ready_;}
double CUDACompiledGraph::CompileMs() const{std::lock_guard<std::mutex> lock(mutex_);return compileMs_;}
double CUDACompiledGraph::LastInferenceMs() const{std::lock_guard<std::mutex> lock(mutex_);return lastMs_;}
void CUDACompiledGraph::SetCudaGraph(bool enabled){std::lock_guard<std::mutex> lock(mutex_);useCudaGraph_=enabled;}
void CUDACompiledGraph::SetSerialGraphReuse(bool enabled){std::lock_guard<std::mutex> lock(mutex_);serialGraphReuse_=enabled;}
void CUDACompiledGraph::SetBuilderWorkspaceBytes(long long bytes){
    std::lock_guard<std::mutex> lock(mutex_);
    if(ready_)throw X::Error("Set builder workspace before building a CUDA graph");
    if(bytes<(1LL<<20)||bytes>(1LL<<30))throw X::Error("CUDA graph builder workspace requires 1MiB..1GiB");
    builderWorkspaceBytes_=static_cast<unsigned long long>(bytes);
}
void CUDACompiledGraph::SetOutputStorage(X::Value output){
    std::lock_guard<std::mutex> lock(mutex_);
    if(!ready_||deviceOutput_.IsValid())throw X::Error("Bind external output once after build and before first execution");
    if(output.runtime()!=Host()->runtime)throw X::Error("External CUDA output must belong to this runtime");
    std::string error;if(!TRTBuilder::CapturedOutputCompatible(execution_,output,error))throw X::Error(error);
    deviceOutput_=output;externalOutput_=true;
}
std::string CUDACompiledGraph::StatsJson() const{
    std::lock_guard<std::mutex> lock(mutex_);auto info=nlohmann::json::parse(TRTBuilder::RetainedExecutionStats(execution_));
    info["cuda_graph_requested"]=useCudaGraph_;
    info["qwen35_whole_program_cuda"]=Qwen35WholeProgramCUDA();
    info["serial_graph_reuse_requested"]=serialGraphReuse_;
    info["builder_workspace_limit_bytes"]=builderWorkspaceBytes_;
    info["external_bf16_weight_inputs"]=externalWeightCount_;
    info["checkpoint_weight_precision"]=weightPrecision_;
    info["native_nvfp4_projection_weights"]=nativeNVFP4WeightCount_;
    info["native_fp8_projection_weights"]=nativeFP8WeightCount_;
    info["joined_projection_inputs"]=joinedProjectionInputs_;
    info["external_bf16_weight_bytes"]=externalWeightBytes_;
    info["external_weight_scope"]="Caller-owned symbolic BF16 inputs; excluded from graph-owned allocations";
    uint64_t span=0,footprint=0;
    if(deviceOutput_.IsValid()){
        auto value=X::Tensor(deviceOutput_).Info();span=value.byte_size;footprint=TensorHelper::ItemSize(value.dtype);
        for(uint32_t i=0;i<value.rank;++i)footprint*=value.shape[i];
    }
    info["device_output_bytes"]=footprint;
    info["owned_device_output_bytes"]=externalOutput_?0:span;
    info["external_output_storage"]=externalOutput_;
    info["external_output_storage_span_bytes"]=externalOutput_?span:0;
    info["output_storage_scope"]="Logical output footprint; externally bound storage is excluded from owned bytes and must be counted once by its caller";
    return info.dump();
}
bool CUDACompiledGraph::Build(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();
    if(graph.runtime()!=Host()->runtime||symbols.runtime()!=Host()->runtime||!symbols.IsList())
        throw X::Error("CUDA graph and symbols must belong to this runtime");
    std::string error;if(!weights_.Open(checkpoint,error))throw X::Error(error);
    const auto path=std::filesystem::path(checkpoint);
    std::ifstream file((std::filesystem::is_directory(path)?path:path.parent_path())/"config.json");
    if(file){
        nlohmann::json config;
        try{file>>config;
            if(config.contains("quantization_config")&&config["quantization_config"].is_object()){
                const auto& q=config["quantization_config"];
                if(q.value("quant_method",std::string())=="modelopt"&&q.value("quant_algo",std::string())=="MIXED_PRECISION")
                    weightPrecision_="modelopt_mixed";
            }
        }catch(...){throw X::Error("Invalid CUDA checkpoint quantization config");}
    }
    std::filesystem::create_directories(cacheRoot_);
    X::ARGS inputs;for(uint64_t i=0;i<symbols.Size();++i)inputs.push_back(symbols.Get(i));
    auto arguments=inputs;TRTBuilder builder(Host());builder.SetCapturedStrictFP32(true);
    builder.SetCapturedWorkspaceBytes(builderWorkspaceBytes_);
    builder.SetCapturedWeightProfile(weightPrecision_);
    auto start=std::chrono::steady_clock::now();
    if(!builder.BuildCapturedGraph(graph,X::Value(),arguments,inputs,&weights_,enginePath_,{},error))
        throw X::Error(error);
    execution_=TRTBuilder::RetainCachedExecution(enginePath_,&weights_,error,weightPrecision_=="modelopt_mixed");
    if(!execution_)throw X::Error(error);
    externalWeightCount_=builder.ExternalBF16WeightInputs();externalWeightBytes_=builder.ExternalBF16WeightBytes();
    joinedProjectionInputs_=builder.JoinedProjectionInputs();
    nativeNVFP4WeightCount_=builder.NativeNVFP4WeightCount();
    nativeFP8WeightCount_=builder.NativeFP8WeightCount();
    compileMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    ready_=true;return true;
}
X::Value CUDACompiledGraph::Run(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);if(!ready_)throw X::Error("CUDA graph has not been built");
    if(externalOutput_)throw X::Error("Externally bound output requires run_device");
    TRTBuilder builder(Host());std::string error;auto start=std::chrono::steady_clock::now();
    auto result=builder.RunCapturedEngine(enginePath_,inputs,&weights_,X::Value(),false,error);
    if(!result.IsValid())throw X::Error(error);
    // Explicit host boundary for existing request-state orchestration. A GPU
    // resident session is a separate optimization, not hidden CPU execution.
    X::Tensor tensor(result);X::Value host=TensorHelper::CopyToCPU(tensor);
    lastMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return host;
}
X::Value CUDACompiledGraph::RunDevice(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);if(!ready_)throw X::Error("CUDA graph has not been built");
    if(externalOutput_){
        if(!inputs.IsList())throw X::Error("External CUDA output requires tensor input list");
        auto output=X::Tensor(deviceOutput_).Info();auto out=reinterpret_cast<uintptr_t>(output.data);
        for(uint64_t i=0;i<inputs.Size();++i)if(X::Tensor::IsTensor(inputs.Get(i))){
            auto input=X::Tensor(inputs.Get(i)).Info();if(input.device_type!=TensorHelper::CudaDevice||!input.data)continue;
            auto in=reinterpret_cast<uintptr_t>(input.data);
            // Remaining storage spans intentionally reject even ambiguous views
            // of the same backing allocation. Inputs and output stay separate.
            if(out>=in?out-in<input.byte_size:in-out<output.byte_size)
                throw X::Error("External CUDA output overlaps input storage");
        }
    }
    TRTBuilder builder(Host());std::string error;auto start=std::chrono::steady_clock::now();
    auto result=builder.RunCapturedEngine(enginePath_,inputs,&weights_,deviceOutput_,useCudaGraph_,error,{},serialGraphReuse_);
    if(!result.IsValid())throw X::Error(error);
    if(externalOutput_&&X::Tensor(result).Info().data!=X::Tensor(deviceOutput_).Info().data)
        throw X::Error("Configured CUDA output storage was not reused");
    deviceOutput_=result;
    lastMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    // Reused device storage: consume/copy its outputs before the next call.
    return result;
}
}
#endif
