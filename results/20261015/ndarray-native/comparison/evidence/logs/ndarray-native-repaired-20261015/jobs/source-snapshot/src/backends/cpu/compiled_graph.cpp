// SPDX-License-Identifier: Apache-2.0
#if defined(GARNET_CPU_ONLY)
#include "compiled_graph.h"
#include "openvino_builder.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <cstdlib>
namespace Garnet {
CPUCompiledGraph::CPUCompiledGraph() {
    static std::atomic<uint64_t> next{0};
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    m_cacheRoot=(std::filesystem::temp_directory_path()/
        ("garnet-cpu-memory-"+std::to_string(stamp)+"-"+std::to_string(++next))).string()+"/";
    m_cacheKey=m_cacheRoot+"graph";
}
CPUCompiledGraph::~CPUCompiledGraph() { Close(); }
void CPUCompiledGraph::CloseUnlocked() {
    OpenVINOBuilder::ReleaseCachedExecutions(m_cacheRoot);
    m_ready=false;
}
void CPUCompiledGraph::Close() { std::lock_guard<std::mutex> lock(m_mutex);CloseUnlocked(); }
bool CPUCompiledGraph::Ready() const { std::lock_guard<std::mutex> lock(m_mutex);return m_ready; }
double CPUCompiledGraph::CompileMs() const { std::lock_guard<std::mutex> lock(m_mutex);return m_compileMs; }
double CPUCompiledGraph::LastInferenceMs() const { std::lock_guard<std::mutex> lock(m_mutex);return m_lastInferenceMs; }
bool CPUCompiledGraph::Build(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(m_mutex);CloseUnlocked();
    if(graph.runtime()!=Host()->runtime||symbols.runtime()!=Host()->runtime||!symbols.IsList())
        throw X::Error("CPU graph and symbolic inputs must belong to this runtime");
    std::string error;
    if(!m_weights.Open(checkpoint,error))throw X::Error(error);
    X::ARGS inputs;for(uint64_t i=0;i<symbols.Size();++i)inputs.push_back(symbols.Get(i));
    auto arguments=inputs;OpenVINOBuilder builder;
    const auto start=std::chrono::steady_clock::now();
    const char* configured=std::getenv("GARNET_CPU_WEIGHT_PRECISION");
    const std::string precision=configured&&*configured?configured:"bf16";
    if(precision!="bf16"&&precision!="fp16"&&precision!="int4_fp16")throw X::Error("CPU weight precision must be bf16, fp16 or int4_fp16");
    if(!builder.BuildCapturedGraph(graph,X::Value(),arguments,inputs,&m_weights,m_cacheKey,precision,error,true))throw X::Error(error);
    if(!builder.PrepareCapturedEngine(m_cacheKey,&m_weights,error)){CloseUnlocked();throw X::Error(error);}
    m_compileMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    m_ready=true;return true;
}
X::Value CPUCompiledGraph::Run(X::Value inputs) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if(!m_ready)throw X::Error("CPU graph has not been built");
    OpenVINOBuilder builder;std::string error;const auto start=std::chrono::steady_clock::now();
    auto result=builder.RunCapturedEngine(m_cacheKey,inputs,&m_weights,X::Value(),false,1,error);
    m_lastInferenceMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    if(!result.IsValid())throw X::Error(error);
    return result;
}
}
#endif
