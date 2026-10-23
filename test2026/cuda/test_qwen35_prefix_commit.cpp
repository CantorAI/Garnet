// SPDX-License-Identifier: Apache-2.0
// Standalone GPU guard/bit-pattern/graph tests; no engine/model timing.
#include "../../src/cuda/qwen35_prefix_commit.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void require(bool b){if(!b)throw std::runtime_error("Prefix commit assertion failed");}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{
    uint32_t* raw=nullptr;size_t size;static constexpr size_t guard=64;
    explicit Buffer(size_t n):size(n){check(cudaMalloc((void**)&raw,(n+2*guard)*4));check(cudaMemset(raw,0x5B,(n+2*guard)*4));}
    ~Buffer(){if(raw)cudaFree(raw);}float* data(){return (float*)(raw+guard);}
    template<class T>void put(const std::vector<T>& v){require(v.size()==size&&sizeof(T)==4);check(cudaMemcpy(data(),v.data(),size*4,cudaMemcpyHostToDevice));}
    std::vector<uint32_t> get(){std::vector<uint32_t> v(size);check(cudaMemcpy(v.data(),data(),size*4,cudaMemcpyDeviceToHost));return v;}
    void guards(){std::vector<uint32_t> ends(guard*2);check(cudaMemcpy(ends.data(),raw,guard*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(ends.data()+guard,raw+guard+size,guard*4,cudaMemcpyDeviceToHost));for(auto v:ends)require(v==0x5B5B5B5B);}
};
struct Graph{
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Graph(){check(cudaStreamCreate(&stream));}~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
}
int main(){try{check(cudaSetDevice(0));int cases=0;
    for(int b:{1,8})for(int t:{4,16}){
        int h=2560,kv=4,c=513,d=256;size_t hiddenN=size_t(b)*t*h,cacheN=size_t(b)*kv*c*d;
        std::vector<uint32_t> hidden(hiddenN),cache(cacheN),backup(cacheN);
        // Arbitrary bits include NaN payloads/signed zero; no arithmetic on data.
        for(size_t i=0;i<hiddenN;++i)hidden[i]=uint32_t(i*2654435761ULL);hidden[0]=0x80000000;
        for(size_t i=0;i<cacheN;++i){cache[i]=uint32_t(i*2246822519ULL);backup[i]=~cache[i];}
        backup[7]=0x7fa12345;cache[8]=0xffc12345;
        Buffer gpuHidden(hiddenN),gpuCache(cacheN),gpuBackup(cacheN),gpuActive(size_t(b)*t),gpuSpans(size_t(b)*2);gpuBackup.put(backup);
        for(int pattern=0;pattern<6;++pattern){
            std::vector<float> active(size_t(b)*t),spans(size_t(b)*2);auto expectedHidden=hidden,expectedCache=cache;
            for(int row=0;row<b;++row){int start=pattern==5?c-t:row*3,count=pattern==4?row%t:std::min(pattern,t);spans[row*2]=float(start);spans[row*2+1]=float(count);
                for(int j=0;j<t;++j)active[row*t+j]=float(j<count);
            }
            for(size_t i=0;i<hiddenN;++i)if(active[i/h]==0)expectedHidden[i]=0;
            for(int row=0;row<b;++row)for(int head=0;head<kv;++head)for(int slot=0;slot<c;++slot){
                bool accepted=slot>=spans[row*2]&&slot<spans[row*2]+spans[row*2+1];
                size_t offset=((size_t(row)*kv+head)*c+slot)*d;
                if(!accepted)std::copy_n(backup.begin()+offset,d,expectedCache.begin()+offset);
            }
            gpuHidden.put(hidden);gpuCache.put(cache);gpuActive.put(active);gpuSpans.put(spans);Graph graph;
            auto run=[&]{check(Garnet::MaskQwen35Chunk(gpuHidden.data(),gpuActive.data(),b,t,h,graph.stream));check(Garnet::RestoreQwen35KVSuffix(gpuCache.data(),gpuBackup.data(),gpuSpans.data(),b,kv,c,d,graph.stream));};
            run();check(cudaStreamSynchronize(graph.stream));require(gpuHidden.get()==expectedHidden&&gpuCache.get()==expectedCache);
            check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeGlobal));run();check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
            for(int repeat=0;repeat<3;++repeat){gpuHidden.put(hidden);gpuCache.put(cache);check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));require(gpuHidden.get()==expectedHidden&&gpuCache.get()==expectedCache);}
            require(Garnet::MaskQwen35Chunk(gpuHidden.data(),gpuHidden.data(),b,t,h,graph.stream)!=cudaSuccess);
            require(Garnet::RestoreQwen35KVSuffix(gpuCache.data(),gpuBackup.data(),gpuSpans.data(),9,kv,c,d,graph.stream)!=cudaSuccess);
            require(Garnet::RestoreQwen35KVSuffix(gpuCache.data(),gpuCache.data()+1,gpuSpans.data(),b,kv,c,d,graph.stream)!=cudaSuccess);
            require(gpuHidden.get()==expectedHidden&&gpuCache.get()==expectedCache&&gpuBackup.get()==backup);
            for(auto* p:{&gpuHidden,&gpuCache,&gpuBackup,&gpuActive,&gpuSpans})p->guards();++cases;
        }
    }std::cout<<"QWEN35_PREFIX_COMMIT_PASS cases "<<cases<<" bit_patterns guards rejection graph3\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
