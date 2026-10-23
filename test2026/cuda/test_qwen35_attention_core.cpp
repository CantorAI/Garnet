// SPDX-License-Identifier: Apache-2.0
// Standalone prototype validation: never run beside an engine benchmark.
#include "../../src/cuda/qwen35_attention_core.cuh"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef GARNET_TEST_ATTENTION_TRT
#include "attention_core_trt_probe.h"
#endif
using json=nlohmann::json;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
size_t count(const json& shape){size_t n=1;for(auto& d:shape){auto v=d.get<int>();if(v<=0)throw std::runtime_error("Invalid fixture dimension");n*=size_t(v);}return n;}
std::vector<float> read(const json& descriptor){
    std::vector<float> result(count(descriptor.at("shape")));
    std::ifstream in(descriptor.at("file").get<std::string>(),std::ios::binary|std::ios::ate);
    if(!in||in.tellg()!=std::streamoff(result.size()*sizeof(float)))throw std::runtime_error("Fixture size mismatch");
    in.seekg(0);in.read(reinterpret_cast<char*>(result.data()),result.size()*sizeof(float));
    if(!in)throw std::runtime_error("Fixture read failed");return result;
}
struct Buffer {
    static constexpr size_t guard=64;static constexpr float sentinel=1234567.f;
    float* raw=nullptr;size_t elements;
    explicit Buffer(size_t n):elements(n){
        check(cudaMalloc(reinterpret_cast<void**>(&raw),(n+2*guard)*sizeof(float)));
        std::vector<float> initial(n+2*guard,sentinel);
        check(cudaMemcpy(raw,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
    }
    ~Buffer(){if(raw)cudaFree(raw);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    float* data(){return raw+guard;}
    void upload(const std::vector<float>& values){
        if(values.size()!=elements)throw std::runtime_error("Upload size mismatch");
        check(cudaMemcpy(data(),values.data(),elements*sizeof(float),cudaMemcpyHostToDevice));
    }
    std::vector<float> download(){
        std::vector<float> values(elements);check(cudaMemcpy(values.data(),data(),elements*sizeof(float),cudaMemcpyDeviceToHost));return values;
    }
    void verifyGuard(){
        std::vector<float> ends(2*guard);
        check(cudaMemcpy(ends.data(),raw,guard*sizeof(float),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(ends.data()+guard,data()+elements,guard*sizeof(float),cudaMemcpyDeviceToHost));
        if(!std::all_of(ends.begin(),ends.end(),[](float v){return v==sentinel;}))throw std::runtime_error("Device buffer guard overwritten");
    }
};
bool bitEqual(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(float));}
float compare(const std::vector<float>& actual,const std::vector<float>& expected,bool exact){
    if(actual.size()!=expected.size())throw std::runtime_error("Output size mismatch");
    if(exact&&!bitEqual(actual,expected))throw std::runtime_error("Cache output is not bit-exact");
    float maxError=0;
    for(size_t i=0;i<actual.size();++i){
        float error=std::abs(actual[i]-expected[i]);maxError=std::max(maxError,error);
        if(!std::isfinite(actual[i])||error>2e-5f+2e-4f*std::abs(expected[i])){
            std::cerr<<"index="<<i<<" actual="<<actual[i]<<" expected="<<expected[i]<<" error="<<error<<"\n";
            throw std::runtime_error("FP32 attention reference mismatch");
        }
    }return maxError;
}
struct Graph {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    Graph(){check(cudaStreamCreate(&stream));}
    ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
}
int main(int argc,char** argv){
    try{
        bool trtMode=false;
#ifdef GARNET_TEST_ATTENTION_TRT
        trtMode=argc==3&&std::string(argv[2])=="trt";
#endif
        if(argc!=(trtMode?3:2))throw std::runtime_error("Expected fixtures.json [trt]");
        std::ifstream file(argv[1]);json cases;file>>cases;check(cudaSetDevice(0));
        int index=0;float overall=0;
        for(const auto& item:cases){
            const auto& cfg=item.at("config");
            Garnet::Qwen35AttentionCoreConfig c{cfg.at("batch"),cfg.at("tokens"),cfg.at("heads"),cfg.at("kv"),cfg.at("capacity"),cfg.at("dim")};
            std::vector<std::vector<float>> inputs,expected;
            std::vector<std::unique_ptr<Buffer>> deviceInputs,deviceOutputs;
            const void* in[8];void* out[3];
            for(int i=0;i<8;++i){inputs.push_back(read(item.at("inputs").at(i)));deviceInputs.emplace_back(new Buffer(inputs.back().size()));deviceInputs.back()->upload(inputs.back());in[i]=deviceInputs.back()->data();}
            for(int i=0;i<3;++i){expected.push_back(read(item.at("outputs").at(i)));deviceOutputs.emplace_back(new Buffer(expected.back().size()));out[i]=deviceOutputs.back()->data();}
            size_t workspaceBytes=Garnet::Qwen35AttentionCoreWorkspace(c);
            if(!workspaceBytes||workspaceBytes%sizeof(float))throw std::runtime_error("Invalid workspace size");
#ifdef GARNET_TEST_ATTENTION_TRT
            std::unique_ptr<AttentionCoreTRTProbe> trt;
            if(trtMode)trt.reset(new AttentionCoreTRTProbe(c));
#endif
            Buffer workspace(workspaceBytes/sizeof(float));Graph graph;
            auto enqueue=[&]{
#ifdef GARNET_TEST_ATTENTION_TRT
                if(trt)return trt->enqueue(in,out,graph.stream);
#endif
                return Garnet::RunQwen35AttentionCore(c,in,out,workspace.data(),graph.stream);
            };
            auto reset=[&]{for(auto& output:deviceOutputs)check(cudaMemsetAsync(output->data(),0xff,output->elements*sizeof(float),graph.stream));check(cudaMemsetAsync(workspace.data(),0xff,workspaceBytes,graph.stream));};
            reset();
            if(enqueue())throw std::runtime_error("Direct enqueue failed");
            check(cudaStreamSynchronize(graph.stream));std::vector<std::vector<float>> direct;
            for(int i=0;i<3;++i){direct.push_back(deviceOutputs[i]->download());overall=std::max(overall,compare(direct.back(),expected[i],i>0));}
            check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeGlobal));
            if(enqueue())throw std::runtime_error("Capture enqueue failed");
            check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.executable,graph.graph,0));
            for(int replay=0;replay<3;++replay){
                reset();check(cudaGraphLaunch(graph.executable,graph.stream));check(cudaStreamSynchronize(graph.stream));
                for(int i=0;i<3;++i){auto actual=deviceOutputs[i]->download();compare(actual,expected[i],i>0);if(!bitEqual(actual,direct[i]))throw std::runtime_error("Graph replay differs from direct result");}
            }
            auto invalid=c;invalid.dim=0;
            if(Garnet::Qwen35AttentionCoreWorkspace(invalid)||Garnet::RunQwen35AttentionCore(invalid,in,out,workspace.data(),graph.stream)==0)throw std::runtime_error("Invalid host shape accepted");
            for(int i=0;i<3;++i)if(!bitEqual(deviceOutputs[i]->download(),direct[i]))throw std::runtime_error("Rejected shape changed output");
            for(int i=0;i<8;++i){deviceInputs[i]->verifyGuard();if(!bitEqual(deviceInputs[i]->download(),inputs[i]))throw std::runtime_error("Input tensor mutated");}
            for(auto& output:deviceOutputs)output->verifyGuard();workspace.verifyGuard();
            std::cout<<json({{"case",index++},{"mode",trtMode?"trt":"direct"},{"batch",c.batch},{"tokens",c.tokens},{"dim",c.dim},{"workspace_bytes",workspaceBytes},{"graph_replays",3},{"status","pass"}}).dump()<<"\n";
        }
        std::cout<<(trtMode?"QWEN35_ATTENTION_TRT_COMPLETE cases ":"QWEN35_ATTENTION_CORE_COMPLETE cases ")<<cases.size()<<" max_abs_error "<<overall<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"ATTENTION_CORE_FAILURE "<<e.what()<<"\n";return 1;}
}
