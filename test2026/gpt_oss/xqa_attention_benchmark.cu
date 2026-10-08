// SPDX-License-Identifier: Apache-2.0
// Native eight-layer synthetic diagnostic. Both arms include production KV
// writes. Private XQA stays outside inference; no full-model speed claim.
#include "gpt_oss_xqa_attention.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Garnet;
namespace {
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
void check(cudaError_t status) { if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status)); }
struct Buffer {
    void* p=nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(&p,bytes)); }
    ~Buffer() { if(cudaFree(p)!=cudaSuccess)std::abort(); }
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
template<class T>void upload(Buffer& b,const std::vector<T>& v) {
    check(cudaMemcpy(b.p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));
}
float pattern(size_t index,bool values) {
    return (int(index%(values?97:89))-(values?48:44))/256.f;
}
__global__ void fillCache(__nv_bfloat16* keys,__nv_bfloat16* values,size_t count) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        keys[i]=__float2bfloat16((int(i%89)-44)/256.f);
        values[i]=__float2bfloat16((int(i%97)-48)/256.f);
    }
}
struct Captured {
    cudaGraph_t graph{};cudaGraphExec_t exec{};cudaStream_t stream;
    explicit Captured(cudaStream_t s):stream(s) {}
    ~Captured() {
        if(cudaStreamSynchronize(stream)!=cudaSuccess)std::abort();
        if(exec&&cudaGraphExecDestroy(exec)!=cudaSuccess)std::abort();
        if(graph&&cudaGraphDestroy(graph)!=cudaSuccess)std::abort();
    }
};
}
int main(int argc,char** argv) {try {
    require(argc==4,"usage: benchmark batch length compact0or1");
    const int batch=std::stoi(argv[1]),length=std::stoi(argv[2]),compact=std::stoi(argv[3]);
    require(batch>=2&&batch<=512&&length>=128&&length<=4096&&(compact==0||compact==1),"shape outside native screen");
    const std::string flash=std::getenv("GARNET_GPT_OSS_DECODE_FLASHINFER")?std::getenv("GARNET_GPT_OSS_DECODE_FLASHINFER"):"0";
    const std::string grouped=std::getenv("GARNET_GPT_OSS_DECODE_GQA_TILED")?std::getenv("GARNET_GPT_OSS_DECODE_GQA_TILED"):"0";
    const std::string splits=std::getenv("GARNET_GPT_OSS_DECODE_GQA_SPLITS")?std::getenv("GARNET_GPT_OSS_DECODE_GQA_SPLITS"):"8";
    require((flash=="0"||flash=="1")&&(grouped=="0"||grouped=="1")&&
        (splits=="4"||splits=="8"||splits=="16"),"unsupported original attention selector");
#ifndef GARNET_GPT_OSS_ENABLE_FLASHINFER_PREFILL
    require(flash=="0","FlashInfer requested but not built; refusing mislabeled comparison");
#endif
    constexpr int layers=8,heads=32,kvHeads=4,dim=64,packed=2560;
    const int logical=(length+15)/16,globalPages=batch*logical;
    // Nine ring pages preserve128 visible tokens plus an allowed16-token tile.
    const int ring=std::min(logical,9),windowPages=compact?batch*ring:globalPages;
    const size_t globalCount=size_t(compact?layers/2:layers)*globalPages*16*kvHeads*dim;
    const size_t windowCount=compact?size_t(layers/2)*windowPages*16*kvHeads*dim:1;
    Buffer keys(globalCount*2),values(globalCount*2),windowKeys(windowCount*2),windowValues(windowCount*2);
    fillCache<<<4096,256>>>((__nv_bfloat16*)keys.p,(__nv_bfloat16*)values.p,globalCount);check(cudaGetLastError());
    if(compact) {fillCache<<<4096,256>>>((__nv_bfloat16*)windowKeys.p,(__nv_bfloat16*)windowValues.p,windowCount);check(cudaGetLastError());}
    Buffer query(size_t(batch)*packed*4),sinks(heads*4),table(size_t(batch)*logical*4),windowTable(size_t(batch)*logical*4),
        lengths(batch*4),starts(batch*4),active(batch*4),original(size_t(batch)*heads*dim*4),candidate(size_t(batch)*heads*dim*4);
    std::vector<float> q(size_t(batch)*packed),sink(heads);
    for(size_t i=0;i<q.size();++i)q[i]=(int(i%113)-56)/128.f;
    for(int h=0;h<heads;++h)sink[h]=(h%11-5)*.25f;
    std::vector<int> globalTable(size_t(batch)*logical),ringTable(globalTable.size());
    for(int b=0;b<batch;++b)for(int p=0;p<logical;++p) {
        globalTable[size_t(b)*logical+p]=b*logical+p;
        ringTable[size_t(b)*logical+p]=b*ring+p%ring;
    }
    upload(query,q);upload(sinks,sink);upload(table,globalTable);upload(windowTable,ringTable);
    upload(lengths,std::vector<int>(batch,length));upload(starts,std::vector<int>(batch,length-1));upload(active,std::vector<int>(batch,1));
    GptOssOptions o;o.kind=1;o.qHeads=heads;o.kvHeads=kvHeads;o.headDim=dim;
    GptOssXqaContextState context;check(InitializeGptOssXqaContext(&context));
    const auto privateBytes=GptOssXqaAttentionWorkspace(batch,1,logical,o);require(privateBytes>0,"private workspace rejected");
    Buffer privateScratch(privateBytes),originalScratch(size_t(batch)*heads*64*130*4);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const void* in[]{query.p,keys.p,values.p,table.p,lengths.p,starts.p,active.p,sinks.p};
    auto select=[&](int layer) {
        o.window=layer%2?128:0;o.layer=compact?layer/2:layer;
        const bool useWindow=compact&&o.window;
        in[1]=useWindow?windowKeys.p:keys.p;in[2]=useWindow?windowValues.p:values.p;
        in[3]=useWindow?windowTable.p:table.p;
        return useWindow?windowPages:globalPages;
    };
    auto invoke=[&](int layer,bool xqa) {
        const int pages=select(layer);
        if(xqa) {
            check(TestGptOssWriteKV(in,batch,1,logical,pages,o,stream));
            check(RunGptOssXqaAttention(in,(float*)candidate.p,privateScratch.p,batch,1,logical,pages,o,context,stream));
        } else check(RunGptOssAttention(in,(float*)original.p,originalScratch.p,batch,1,logical,pages,o,stream));
    };
    size_t compared=0,fp64Checked=0;double maxPair=0,maxOracle=0;
    std::vector<float> a(size_t(batch)*heads*dim),c(a.size());
    auto validate=[&](int layer,bool reference) {
        const int pages=select(layer);
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(a.data(),original.p,a.size()*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(c.data(),candidate.p,c.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();++i) {
            require(std::isfinite(a[i])&&std::isfinite(c[i]),"nonfinite full output matrix");
            const double error=std::abs(double(c[i])-a[i]);maxPair=std::max(maxPair,error);++compared;
            require(error<=.006*(1+std::abs(double(a[i]))),"full original/private matrix bound");
        }
        if(!reference)return;
        const auto& selectedTable=compact&&o.window?ringTable:globalTable;
        for(int b:{0,batch/2,batch-1})for(int h=0;h<heads;++h) {
            double sum=std::exp(double(sink[h])),acc[dim]={};
            for(int pos=o.window?std::max(0,length-o.window):0;pos<length;++pos) {
                const int page=selectedTable[size_t(b)*logical+pos/16],kh=h/8;
                const size_t offset=((size_t(o.layer)*pages+page)*16+pos%16)*kvHeads*dim+kh*dim;
                double score=0;
                for(int d=0;d<dim;++d) {
                    const float key=pos==length-1?q[size_t(b)*packed+(heads+kh)*dim+d]:pattern(offset+d,false);
                    score+=double(q[size_t(b)*packed+h*dim+d])*key;
                }
                const double weight=std::exp(score/8.);sum+=weight;
                for(int d=0;d<dim;++d) {
                    const float value=pos==length-1?q[size_t(b)*packed+(heads+kvHeads+kh)*dim+d]:pattern(offset+d,true);
                    acc[d]+=weight*value;
                }
            }
            for(int d=0;d<dim;++d) {
                const double expected=__bfloat162float(__float2bfloat16(float(acc[d]/sum)));
                const size_t index=(size_t(b)*heads+h)*dim+d;
                for(float actual:{a[index],c[index]}) {
                    const double error=std::abs(double(actual)-expected);maxOracle=std::max(maxOracle,error);++fp64Checked;
                    require(error<=.006*(1+std::abs(expected)),"sampled independent FP64 bound");
                }
            }
        }
    };
    for(int layer=0;layer<layers;++layer) {invoke(layer,false);invoke(layer,true);validate(layer,layer<2||layer>=6);}
    cudaDeviceProp device;int deviceId;check(cudaGetDevice(&deviceId));check(cudaGetDeviceProperties(&device,deviceId));
    std::cout<<std::setprecision(10)<<"{\"protocol\":\"native-xqa-cost-v1\",\"device\":\""<<device.name
        <<"\",\"batch\":"<<batch<<",\"length\":"<<length<<",\"compact\":"<<compact
        <<",\"original_flashinfer\":"<<flash<<",\"original_gqa_tiled\":"<<grouped<<",\"original_gqa_splits\":"<<splits
        <<",\"layers\":8,\"full_layers\":4,\"window_layers\":4,\"kv_bytes\":"<<(globalCount+(compact?windowCount:0))*4
        <<",\"pair_values\":"<<compared<<",\"sampled_fp64_values\":"<<fp64Checked
        <<",\"max_pair_abs\":"<<maxPair<<",\"max_fp64_abs\":"<<maxOracle<<",\"bounds\":\".006*(1+abs)\",\"arms\":[";
    for(int arm=0;arm<2;++arm) {
        Captured graph(stream);check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int layer=0;layer<layers;++layer)invoke(layer,arm==1);
        check(cudaStreamEndCapture(stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0));
        for(int i=0;i<3;++i)check(cudaGraphLaunch(graph.exec,stream));check(cudaStreamSynchronize(stream));
        cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));std::vector<float> times;
        for(int trial=0;trial<9;++trial) {
            check(cudaEventRecord(begin,stream));for(int i=0;i<3;++i)check(cudaGraphLaunch(graph.exec,stream));
            check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));times.push_back(ms/3);
            // Last graph output must still obey both full-matrix and sampled
            // FP64 obligations. The peer arm is replayed outside timed events.
            invoke(layers-1,arm==0);validate(layers-1,true);
        }
        auto sorted=times;std::sort(sorted.begin(),sorted.end());
        if(arm)std::cout<<',';std::cout<<"{\"name\":\""<<(arm?"write+prepare+xqa+finalize":"production-write+attention")
            <<"\",\"warmups\":3,\"graph_replays_per_trial\":3,\"gpu_ms_trials\":[";
        for(size_t i=0;i<times.size();++i) {if(i)std::cout<<',';std::cout<<times[i];}
        std::cout<<"],\"gpu_ms_median\":"<<sorted[4]<<'}';check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    }
    std::cout<<"],\"post_timing_pair_values\":"<<compared<<",\"post_timing_sampled_fp64_values\":"<<fp64Checked
        <<",\"post_timing_max_pair_abs\":"<<maxPair<<",\"post_timing_max_fp64_abs\":"<<maxOracle
        <<",\"scope\":\"eight-layer native dense or ring synthetic; no TensorRT/NCCL/pretrained/full-goal claim\"}\n";
    check(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e) {std::cerr<<"XQA_COST_FAILED "<<e.what()<<'\n';return 1;}}
