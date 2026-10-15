// SPDX-License-Identifier: Apache-2.0
#pragma once
#if defined(GARNET_CPU_ONLY)
#include "xlang3/xlang3.h"
#include "safetensors_index.h"
#include <mutex>
#include <string>
namespace Garnet {
// GPU-free public graph execution boundary. Requests own their input/state
// tensors; compiled weights stay in memory instead of duplicating SD storage.
class CPUCompiledGraph {
    SafeTensorsIndex m_weights;
    std::string m_cacheRoot,m_cacheKey;
    bool m_ready=false;
    double m_compileMs=0,m_lastInferenceMs=0;
    mutable std::mutex m_mutex;
    void CloseUnlocked();
public:
    CPUCompiledGraph();
    ~CPUCompiledGraph();
    bool Build(X::Value graph,X::Value symbols,const std::string& checkpoint);
    X::Value Run(X::Value inputs);
    void Close();
    bool Ready() const;
    double CompileMs() const;
    double LastInferenceMs() const;
    BEGIN_PACKAGE(CPUCompiledGraph)
        APISET().AddFunc<3>("build",&CPUCompiledGraph::Build);
        APISET().AddFunc<1>("run",&CPUCompiledGraph::Run);
        APISET().AddFunc<0>("close",&CPUCompiledGraph::Close);
        APISET().AddProp("ready",&CPUCompiledGraph::Ready);
        APISET().AddProp("compile_ms",&CPUCompiledGraph::CompileMs);
        APISET().AddProp("last_inference_ms",&CPUCompiledGraph::LastInferenceMs);
    END_PACKAGE
};
}
#endif
