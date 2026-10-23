// SPDX-License-Identifier: Apache-2.0
// Standalone, no serving. FP64 oracle fixed before GPU execution.
#include "../../src/cuda/qwen35_indexed_head.cuh"
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>std::vector<T> read(const json& file,size_t n){
    std::ifstream in(file.get<std::string>(),std::ios::binary|std::ios::ate);
    require(in&&in.tellg()==std::streamoff(n*sizeof(T)),"Fixture byte size mismatch");
    std::vector<T> out(n);in.seekg(0);in.read((char*)out.data(),n*sizeof(T));require(bool(in),"Fixture read failed");return out;
}
struct Buffer {
    unsigned char* raw=nullptr;size_t bytes;static constexpr size_t guard=256;
    explicit Buffer(size_t n):bytes(n){check(cudaMalloc((void**)&raw,n+2*guard));check(cudaMemset(raw,0xA7,n+2*guard));}
    ~Buffer(){if(raw)cudaFree(raw);}
    Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
    void* data(){return raw+guard;}
    template<class T>T* ptr(){return (T*)data();}
    template<class T>void put(const std::vector<T>& v){require(v.size()*sizeof(T)==bytes,"Upload size");check(cudaMemcpy(data(),v.data(),bytes,cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get(){require(bytes%sizeof(T)==0,"Download size");std::vector<T> v(bytes/sizeof(T));check(cudaMemcpy(v.data(),data(),bytes,cudaMemcpyDeviceToHost));return v;}
    void checkGuard(){
        unsigned char ends[2*guard];check(cudaMemcpy(ends,raw,guard,cudaMemcpyDeviceToHost));check(cudaMemcpy(ends+guard,raw+guard+bytes,guard,cudaMemcpyDeviceToHost));
        for(auto x:ends)require(x==0xA7,"Buffer guard overwritten");
    }
};
template<class T>bool equal(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&!memcmp(a.data(),b.data(),a.size()*sizeof(T));}
float bf(uint16_t x){uint32_t bits=uint32_t(x)<<16;float value;memcpy(&value,&bits,4);return value;}
struct Graph {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Graph(){check(cudaStreamCreate(&stream));}
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
}
int main(int argc,char**argv){try{
    require(argc==2,"Expected fixtures.json");std::ifstream in(argv[1]);json cases;in>>cases;check(cudaSetDevice(0));int number=0;
    for(auto& item:cases){auto& cfg=item.at("config");Garnet::Qwen35IndexedHeadConfig c{cfg.at("rows"),cfg.at("hidden"),cfg.at("vocab"),cfg.at("candidates")};
        size_t count=size_t(c.rows)*c.candidates;auto& f=item.at("files");
        auto query=read<float>(f.at("queries"),size_t(c.rows)*c.hidden);auto table=read<uint16_t>(f.at("table"),size_t(c.vocab)*c.hidden);
        auto indices=read<int>(f.at("indices"),count);auto dots=read<double>(f.at("dots"),count);auto sumAbs=read<double>(f.at("sum_abs"),count);
        Buffer q(query.size()*4),w(table.size()*2),ix(count*4),scores(count*2),ids(c.rows*8),values(c.rows*4);
        q.put(query);w.put(table);ix.put(indices);Graph graph;
        auto run=[&](float* val){return Garnet::RunQwen35IndexedHead(c,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),val,graph.stream);};
        require(run(values.ptr<float>())==0,"Indexed head enqueue failed");check(cudaStreamSynchronize(graph.stream));
        auto direct=scores.get<uint16_t>();auto directIds=ids.get<long long>();auto directValues=values.get<float>();
        // FP32 FMA chains plus block reduction: conservative gamma_(ceil(H/256)+8), then round-to-nearest BF16.
        double n=(c.hidden+255)/256+8,u=std::ldexp(1.,-24),gamma=n*u/(1-n*u),maxError=0,maxFraction=0;
        for(int row=0;row<c.rows;++row){long long best=-1;float bestValue=-INFINITY;double oracleMax=-INFINITY,oracleChosen=-INFINITY,maxBound=0;
            for(int j=0;j<c.candidates;++j){size_t i=size_t(row)*c.candidates+j;int id=indices[i];float actual=bf(direct[i]);
                if(id<0||id>=c.vocab){require(actual==-INFINITY,"Invalid index score is not -Inf");continue;}
                double bound=gamma*sumAbs[i]+(std::abs(dots[i])+gamma*sumAbs[i])/256.+std::ldexp(1.,-133);
                double error=std::abs(double(actual)-dots[i]);require(std::isfinite(actual)&&error<=bound,"Independent FP64 dot/rounding bound failed");
                maxError=std::max(maxError,error);maxFraction=std::max(maxFraction,error/std::max(bound,1e-300));maxBound=std::max(maxBound,bound);oracleMax=std::max(oracleMax,dots[i]);
                if(best<0||actual>bestValue||(actual==bestValue&&id<best)){best=id;bestValue=actual;}
            }
            require(directIds[row]==best&&directValues[row]==bestValue,"Argmax/tie rule mismatch");
            if(best>=0){for(int j=0;j<c.candidates;++j){size_t i=size_t(row)*c.candidates+j;if(indices[i]==best)oracleChosen=std::max(oracleChosen,dots[i]);}
                require(oracleMax-oracleChosen<=2*maxBound,"Selected token exceeds independent oracle margin bound");}
        }
        check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeGlobal));require(run(values.ptr<float>())==0,"Capture enqueue");
        check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
        for(int replay=0;replay<3;++replay){check(cudaMemsetAsync(scores.data(),0,count*2,graph.stream));check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
            require(equal(scores.get<uint16_t>(),direct)&&equal(ids.get<long long>(),directIds)&&equal(values.get<float>(),directValues),"Graph replay differs");}
        require(run(nullptr)==0,"Optional values rejected");check(cudaStreamSynchronize(graph.stream));require(equal(ids.get<long long>(),directIds),"Optional values changed IDs");
        auto reject=[&](Garnet::Qwen35IndexedHeadConfig bad,const float* queryPtr,const void* tablePtr,const int* indexPtr,void* scorePtr,long long* idPtr,float* valuePtr){
            require(Garnet::RunQwen35IndexedHead(bad,queryPtr,tablePtr,indexPtr,scorePtr,idPtr,valuePtr,graph.stream)!=0,"Invalid call accepted");};
        auto bad=c;bad.rows=9;reject(bad,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        bad=c;bad.candidates=c.vocab+1;reject(bad,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,nullptr,w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,q.ptr<float>(),w.data(),ix.ptr<int>(),q.data(),ids.ptr<long long>(),values.ptr<float>());
        reject(c,q.ptr<float>(),w.data(),ix.ptr<int>(),scores.data(),ids.ptr<long long>(),scores.ptr<float>());
        check(cudaStreamSynchronize(graph.stream));require(equal(scores.get<uint16_t>(),direct)&&equal(ids.get<long long>(),directIds),"Rejected call mutated output");
        require(equal(q.get<float>(),query)&&equal(w.get<uint16_t>(),table)&&equal(ix.get<int>(),indices),"Input mutated");
        // All-NaN queries must yield no finite winner, while indices/table remain safe.
        std::vector<float> nanQuery(query.size(),std::numeric_limits<float>::quiet_NaN());q.put(nanQuery);require(run(values.ptr<float>())==0,"NaN enqueue");check(cudaStreamSynchronize(graph.stream));
        for(auto id:ids.get<long long>())require(id==-1,"Nonfinite candidate won");for(auto value:values.get<float>())require(value==-INFINITY,"Nonfinite result value");q.put(query);
        for(Buffer* p:{&q,&w,&ix,&scores,&ids,&values})p->checkGuard();
        require(run(values.ptr<float>())==0,"Timing warmup");check(cudaStreamSynchronize(graph.stream));cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));check(cudaEventRecord(start,graph.stream));
        for(int repeat=0;repeat<20;++repeat)require(run(values.ptr<float>())==0,"Timing enqueue");check(cudaEventRecord(stop,graph.stream));check(cudaEventSynchronize(stop));float elapsed;check(cudaEventElapsedTime(&elapsed,start,stop));cudaEventDestroy(start);cudaEventDestroy(stop);
        // Independent CPU stable sorting checks the candidate selector, including
        // duplicates/ties, nonfinite values, reordered source rows and scratch bounds.
        int sourceRows=c.rows*2;std::vector<uint16_t> logits(size_t(sourceRows)*c.vocab);std::vector<int> selected(c.rows),expected(count);
        for(int r=0;r<sourceRows;++r)for(int j=0;j<c.vocab;++j){
            float v=float((j*37+r*11)%251-125)/16;auto raw=__float2bfloat16_rn(v);memcpy(&logits[size_t(r)*c.vocab+j],&raw,2);
        }
        for(int r=0;r<sourceRows;++r){logits[size_t(r)*c.vocab]=0x7fc0;logits[size_t(r)*c.vocab+1]=0x7f80;logits[size_t(r)*c.vocab+2]=0xff80;}
        if(number==4)std::fill(logits.begin(),logits.end(),uint16_t(0x7fc0));
        for(int r=0;r<c.rows;++r){selected[r]=sourceRows-1-r*2;std::vector<int> order(c.vocab);std::iota(order.begin(),order.end(),0);
            auto value=[&](int id){float v=bf(logits[size_t(selected[r])*c.vocab+id]);return std::isfinite(v)?v:-std::numeric_limits<float>::infinity();};
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return value(a)>value(b);});
            for(int j=0;j<c.candidates;++j)expected[size_t(r)*c.candidates+j]=std::isfinite(value(order[j]))?order[j]:-1;
        }
        size_t scratchBytes=Garnet::Qwen35HeadCandidatesWorkspace(c.vocab);require(scratchBytes>0,"Sort workspace query");
        Buffer logitBuffer(logits.size()*2),candidates(count*4),scratch(scratchBytes);logitBuffer.put(logits);Graph selector;
        auto select=[&]{return Garnet::RunQwen35HeadCandidates(c.rows,sourceRows,c.vocab,c.candidates,selected.data(),logitBuffer.data(),candidates.ptr<int>(),scratch.data(),scratchBytes,selector.stream);};
        require(select()==0,"Selector enqueue failed");check(cudaStreamSynchronize(selector.stream));require(equal(candidates.get<int>(),expected),"Independent CPU selector mismatch");
        check(cudaStreamBeginCapture(selector.stream,cudaStreamCaptureModeGlobal));require(select()==0,"Selector capture");check(cudaStreamEndCapture(selector.stream,&selector.graph));check(cudaGraphInstantiate(&selector.exec,selector.graph,0));
        for(int r=0;r<3;++r){check(cudaMemsetAsync(candidates.data(),0,count*4,selector.stream));check(cudaGraphLaunch(selector.exec,selector.stream));check(cudaStreamSynchronize(selector.stream));require(equal(candidates.get<int>(),expected),"Selector graph replay mismatch");}
        int saved=selected.back();selected.back()=sourceRows;require(select()!=0,"Invalid source row accepted");selected.back()=saved;
        require(Garnet::RunQwen35HeadCandidates(c.rows,sourceRows,c.vocab,c.candidates,selected.data(),logitBuffer.data(),candidates.ptr<int>(),scratch.data(),scratchBytes-1,selector.stream)!=0,"Short scratch accepted");
        require(Garnet::RunQwen35HeadCandidates(c.rows,sourceRows,c.vocab,c.candidates,selected.data(),logitBuffer.data(),logitBuffer.ptr<int>(),scratch.data(),scratchBytes,selector.stream)!=0,"Selector alias accepted");
        check(cudaStreamSynchronize(selector.stream));require(equal(candidates.get<int>(),expected)&&equal(logitBuffer.get<uint16_t>(),logits),"Selector rejection/input mutation");
        for(Buffer* p:{&logitBuffer,&candidates,&scratch})p->checkGuard();
        std::cout<<json{{"case",number++},{"config",cfg},{"max_fp64_error",maxError},{"max_bound_fraction",maxFraction},{"three_graph_replays","bit_exact"},{"guards_inputs_invalid_alias_nan","pass"},{"candidate_selector_cpu_graph_safety","pass"},{"selector_workspace_bytes",scratchBytes},{"warm_subset_microseconds",elapsed*1000/20},{"timing_scope","cached synthetic subset only; not model speed evidence"}}.dump()<<"\n";
    }std::cout<<"QWEN35_INDEXED_HEAD_PASS "<<number<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
