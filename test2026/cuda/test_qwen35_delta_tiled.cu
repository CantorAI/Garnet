// SPDX-License-Identifier: Apache-2.0
// Standalone control/oracle/timing screen; never run beside a model cohort.
#include "../../src/cuda/qwen35_delta.cu"
#include "qwen35_delta_tiled_prototype.cuh"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{
    static constexpr int guard=64;static constexpr float sentinel=1234567.f;
    float* raw=nullptr;size_t n;
    explicit Buffer(size_t count):n(count){check(cudaMalloc(&raw,(n+2*guard)*4));std::vector<float> initial(n+2*guard,sentinel);check(cudaMemcpy(raw,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));}
    ~Buffer(){cudaFree(raw);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    float* data(){return raw+guard;}
    void upload(const std::vector<float>& v){if(v.size()!=n)throw std::runtime_error("Upload size");check(cudaMemcpy(data(),v.data(),n*4,cudaMemcpyHostToDevice));}
    std::vector<float> read(){std::vector<float> all(n+2*guard);check(cudaMemcpy(all.data(),raw,all.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<guard;++i)if(all[i]!=sentinel||all[guard+n+i]!=sentinel)throw std::runtime_error("Guard overwritten");
        return {all.begin()+guard,all.begin()+guard+n};}
};
// Independent scalar FP64 equations, fixed bounds shared by both native paths.
std::array<std::vector<double>,3> reference(Garnet::Qwen35DeltaConfig c,int b,int t,const std::vector<std::vector<float>>& x){
    const int ch=2*c.kh*128+c.vh*128;
    std::vector<double> conv(size_t(b)*t*ch),history(x[4].begin(),x[4].end()),state(x[5].begin(),x[5].end()),output(size_t(b)*t*c.vh*128);
    for(int row=0;row<b;++row)for(int channel=0;channel<ch;++channel){
        auto offset=(size_t(row)*ch+channel)*c.kernel;
        for(int token=0;token<t;++token){
            if(x[6][row*t+token]==0)continue;
            for(int k=0;k<c.kernel-1;++k)history[offset+k]=history[offset+k+1];
            history[offset+c.kernel-1]=x[0][(size_t(row)*t+token)*ch+channel];
            double value=0;for(int k=0;k<c.kernel;++k)value+=history[offset+k]*x[7][channel*c.kernel+k];
            conv[(size_t(row)*t+token)*ch+channel]=value/(1+std::exp(-value));
        }
    }
    for(int row=0;row<b;++row)for(int head=0;head<c.vh;++head)for(int token=0;token<t;++token){
        const int bt=row*t+token,kh=head/(c.vh/c.kh);if(x[6][bt]==0)continue;
        std::array<double,128> q,k,core;double qs=0,ks=0;
        for(int d=0;d<128;++d){q[d]=conv[size_t(bt)*ch+kh*128+d];k[d]=conv[size_t(bt)*ch+c.kh*128+kh*128+d];qs+=q[d]*q[d];ks+=k[d]*k[d];}
        for(int d=0;d<128;++d){q[d]/=std::sqrt(qs+1e-6)*std::sqrt(128.0);k[d]/=std::sqrt(ks+1e-6);}
        double av=double(x[3][bt*c.vh+head])+x[8][head];double softplus=std::max(av,0.0)+std::log1p(std::exp(-std::abs(av)));
        double decay=std::exp(-std::exp(double(x[9][head]))*softplus),gate=1/(1+std::exp(-double(x[2][bt*c.vh+head])));
        double sum=0;
        for(int column=0;column<128;++column){
            double memory=0;
            for(int d=0;d<128;++d){auto index=((size_t(row)*c.vh+head)*128+d)*128+column;state[index]*=decay;memory+=state[index]*k[d];}
            double delta=(conv[size_t(bt)*ch+2*c.kh*128+head*128+column]-memory)*gate,result=0;
            for(int d=0;d<128;++d){auto index=((size_t(row)*c.vh+head)*128+d)*128+column;state[index]+=k[d]*delta;result+=state[index]*q[d];}
            core[column]=result;sum+=result*result;
        }
        double factor=1/std::sqrt(sum/128+c.epsilon);
        for(int column=0;column<128;++column){auto index=(size_t(bt)*c.vh+head)*128+column;double z=x[1][index];output[index]=core[column]*factor*x[10][column]*(z/(1+std::exp(-z)));}
    }
    return {std::move(output),std::move(history),std::move(state)};
}
int main(int argc,char** argv){try{
    if(argc<2||argc>4)throw std::runtime_error("Expected output path, optional variant and --no-timing");
    const bool volatileShared=argc>=3&&std::string(argv[2])=="register128_volatile";
    const bool registers=volatileShared||(argc>=3&&std::string(argv[2])=="register128");
    if(argc>=3&&!registers&&std::string(argv[2])!="tiled128")throw std::runtime_error("Unknown delta variant");
    const bool timed=argc!=4;if(!timed&&std::string(argv[3])!="--no-timing")throw std::runtime_error("Unknown timing option");
    std::ofstream log(argv[1]);if(!log)throw std::runtime_error("Output path");
    std::mt19937 rng(61983);std::uniform_real_distribution<float> random(-0.2f,0.2f);
    cudaFuncAttributes original{},tiled{};check(cudaFuncGetAttributes(&original,Garnet::recurrence));
    if(volatileShared)check(cudaFuncGetAttributes(&tiled,Garnet::registerRecurrence128<true>));else if(registers)check(cudaFuncGetAttributes(&tiled,Garnet::registerRecurrence128<false>));else check(cudaFuncGetAttributes(&tiled,Garnet::tiledRecurrence128));
    log<<json({{"event","kernel_resources"},{"variant",volatileShared?"register128_volatile":registers?"register128":"tiled128"},{"original_registers",original.numRegs},{"original_local_bytes",original.localSizeBytes},{"tiled_registers",tiled.numRegs},{"tiled_local_bytes",tiled.localSizeBytes},{"tiled_shared_bytes",tiled.sharedSizeBytes}}).dump()<<std::endl;
    for(auto profile:std::vector<std::array<int,3>>{{1,4,16},{2,16,16},{4,4,16},{8,4,16},{1,64,16},{1,128,16},{1,4,32},{8,4,32},{1,64,32}}){
        int b=profile[0],t=profile[1],vh=profile[2];Garnet::Qwen35DeltaConfig c{16,vh,128,128,4,1e-6f};int ch=2*c.kh*128+vh*128;
        std::vector<size_t> sizes={size_t(b)*t*ch,size_t(b)*t*vh*128,size_t(b)*t*vh,size_t(b)*t*vh,size_t(b)*ch*4,size_t(b)*vh*128*128,size_t(b)*t,size_t(ch)*4,size_t(vh),size_t(vh),128};
        std::vector<std::vector<float>> x;std::vector<std::unique_ptr<Buffer>> inputs;
        for(size_t n:sizes){std::vector<float> values(n);for(auto& v:values)v=random(rng);x.push_back(std::move(values));}
        for(auto& v:x[5])v*=0.1f;for(auto& v:x[10])v+=1;
        for(int row=0;row<b;++row)for(int token=0;token<t;++token)x[6][row*t+token]=float(row%4!=3&&token<t-row%3&&token%3!=1);
        std::vector<const void*> in;for(auto& values:x){inputs.push_back(std::make_unique<Buffer>(values.size()));inputs.back()->upload(values);in.push_back(inputs.back()->data());}
        std::vector<size_t> outs={size_t(b)*t*vh*128,size_t(b)*ch*4,size_t(b)*vh*128*128};
        std::vector<std::unique_ptr<Buffer>> old,newer;std::vector<void*> a,z;
        for(auto n:outs){old.push_back(std::make_unique<Buffer>(n));newer.push_back(std::make_unique<Buffer>(n));a.push_back(old.back()->data());z.push_back(newer.back()->data());}
        Buffer workspace(registers?size_t(b)*t*ch:Garnet::TiledDeltaWorkspace(c,b,t)/4);
        auto candidateRun=[&](cudaStream_t stream){return registers?Garnet::RunRegisterDelta128(c,b,t,in.data(),z.data(),workspace.data(),stream,volatileShared):Garnet::RunTiledDelta128(c,b,t,in.data(),z.data(),workspace.data(),stream);};
        auto run=[&](bool candidate){int status=candidate?candidateRun(nullptr):Garnet::RunQwen35Delta(c,b,t,in.data(),a.data(),workspace.data(),nullptr);if(status)throw std::runtime_error("Kernel enqueue");};
        run(false);run(true);check(cudaDeviceSynchronize());auto expected=reference(c,b,t,x);double maxError=0;
        for(int field=0;field<3;++field){auto baseline=old[field]->read(),actual=newer[field]->read();
            if(std::memcmp(baseline.data(),actual.data(),actual.size()*4))throw std::runtime_error("Tiled/reference native bytes differ B"+std::to_string(b)+" T"+std::to_string(t)+" field"+std::to_string(field));
            for(size_t i=0;i<actual.size();++i){double error=std::abs(double(actual[i])-expected[field][i]);maxError=std::max(maxError,error);if(!std::isfinite(actual[i])||error>2e-5+2e-4*std::abs(expected[field][i]))throw std::runtime_error("Independent FP64 bound");}
        }
        cudaStream_t stream;check(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        if(candidateRun(stream))throw std::runtime_error("Capture enqueue");
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        for(int i=0;i<3;++i)check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        for(int field=0;field<3;++field){auto baseline=old[field]->read(),actual=newer[field]->read();if(std::memcmp(baseline.data(),actual.data(),actual.size()*4))throw std::runtime_error("Replay bytes differ");}
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        std::array<float,2> ms{};cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
        if(timed)for(int method=0;method<2;++method){for(int i=0;i<3;++i)run(method!=0);check(cudaEventRecord(start));for(int i=0;i<20;++i)run(method!=0);check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&ms[method],start,stop));ms[method]/=20;}
        check(cudaEventDestroy(start));check(cudaEventDestroy(stop));
        for(size_t i=0;i<x.size();++i){auto actual=inputs[i]->read();if(std::memcmp(actual.data(),x[i].data(),actual.size()*4))throw std::runtime_error("Input changed");}
        workspace.read();for(auto& value:old)value->read();for(auto& value:newer)value->read();
        log<<json({{"batch",b},{"tokens",t},{"key_heads",16},{"value_heads",vh},{"native_bytes_equal",true},{"independent_max_abs",maxError},{"fixed_atol",2e-5},{"fixed_rtol",2e-4},{"graph_replays",3},{"inputs_unchanged",true},{"guards_intact",true},{"original_warm_ms",timed?json(ms[0]):json(nullptr)},{"tiled_warm_ms",timed?json(ms[1]):json(nullptr)},{"workspace_bytes",workspace.n*4},{"scope",timed?"Standalone warm component timing, not end-to-end/model acceptance":"Correctness diagnostic only; timing disabled"}}).dump()<<std::endl;
        std::cout<<"TILED_DELTA_PASS B"<<b<<" T"<<t<<" VH"<<vh<<std::endl;
    }
    std::cout<<"QWEN35_TILED_DELTA_COMPLETE"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
