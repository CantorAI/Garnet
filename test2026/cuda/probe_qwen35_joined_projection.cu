// SPDX-License-Identifier: Apache-2.0
// Unwired same-input projection fusion screen using unchanged production GEMM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace Garnet {
std::shared_ptr<void> CreateBF16ProjectionContext();
int RunBF16Projection(const std::shared_ptr<void>&,const void*,const void*,void*,int,int,int,cudaStream_t,void*,size_t) noexcept;
}
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct Buffer {
    void* owner=nullptr;void* p=nullptr;size_t bytes;
    explicit Buffer(size_t count):bytes(count){check(cudaMalloc(&owner,bytes+256));p=static_cast<char*>(owner)+128;check(cudaMemset(owner,0x6d,bytes+256));}
    ~Buffer(){if(owner)cudaFree(owner);}
    void guards()const{std::array<unsigned char,128> a{},b{};check(cudaMemcpy(a.data(),owner,128,cudaMemcpyDeviceToHost));check(cudaMemcpy(b.data(),static_cast<char*>(p)+bytes,128,cudaMemcpyDeviceToHost));for(auto v:a)if(v!=0x6d)throw std::runtime_error("Leading guard changed");for(auto v:b)if(v!=0x6d)throw std::runtime_error("Trailing guard changed");}
    template<class T>void put(const std::vector<T>& v){if(v.size()*sizeof(T)!=bytes)throw std::runtime_error("Upload size");check(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get()const{std::vector<T> v(bytes/sizeof(T));check(cudaMemcpy(v.data(),p,bytes,cudaMemcpyDeviceToHost));return v;}
};
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    Graph(const std::function<void()>& call,cudaStream_t stream){
        call();check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));call();
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
    }
    void run(cudaStream_t stream)const{check(cudaGraphLaunch(executable,stream));}
    ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};
__global__ void evict(uint32_t* data,size_t words){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)data[i]=uint32_t(i*2654435761ULL);}
uint16_t bits(__nv_bfloat16 value){uint16_t result;std::memcpy(&result,&value,2);return result;}
std::array<double,3> timings(const std::array<const Graph*,3>& graphs,Buffer& eviction,cudaStream_t stream,bool sweep){
    std::array<std::vector<float>,3> samples;cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    for(auto graph:graphs)for(int warm=0;warm<3;++warm)graph->run(stream);check(cudaStreamSynchronize(stream));
    for(int repeat=0;repeat<9;++repeat)for(int step=0;step<3;++step){
        int index=(repeat+step)%3;
        if(sweep)evict<<<2048,256,0,stream>>>((uint32_t*)eviction.p,eviction.bytes/4);
        check(cudaEventRecord(begin,stream));graphs[index]->run(stream);check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
        float ms=0;check(cudaEventElapsedTime(&ms,begin,end));samples[index].push_back(ms*1000);
    }
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));std::array<double,3> result{};
    for(int i=0;i<3;++i){std::sort(samples[i].begin(),samples[i].end());result[i]=samples[i][samples[i].size()/2];}return result;
}
json verify(const std::vector<__nv_bfloat16>& x,const std::vector<__nv_bfloat16>& w,const std::vector<__nv_bfloat16>& separate,
    const std::vector<__nv_bfloat16>& joined,int m,int k,int total,int offset,int width){
    size_t different=0;double maxFraction=0;int samples=0;
    for(int row=0;row<m;++row)for(int col=0;col<width;++col){
        auto a=separate[size_t(row)*width+col],b=joined[size_t(row)*total+offset+col];
        if(!std::isfinite(__bfloat162float(a))||!std::isfinite(__bfloat162float(b)))throw std::runtime_error("Nonfinite projection result");
        different+=bits(a)!=bits(b);
    }
    const double u=std::ldexp(1.,-24),gamma=k*u/(1-k*u);
    const int columns=std::min(width,64);
    for(int row=0;row<std::min(m,2);++row)for(int sample=0;sample<columns;++sample){
        int col=sample*(width-1)/(columns-1);double dot=0,absolute=0;
        for(int d=0;d<k;++d){double term=double(__bfloat162float(x[size_t(row)*k+d]))*__bfloat162float(w[size_t(offset+col)*k+d]);dot+=term;absolute+=std::abs(term);}
        const double bound=gamma*absolute+(std::abs(dot)+gamma*absolute)/256.+std::ldexp(1.,-133);
        for(double value:{double(__bfloat162float(separate[size_t(row)*width+col])),double(__bfloat162float(joined[size_t(row)*total+offset+col]))}){
            double error=std::abs(value-dot);if(error>bound)throw std::runtime_error("Original fixed FP64 forward-error bound failed");maxFraction=std::max(maxFraction,error/bound);
        }++samples;
    }
    return {{"bf16_values_different",different},{"fp64_samples_per_provider",samples},{"max_original_fixed_bound_fraction",maxFraction}};
}
}
int main(){try{
    check(cudaSetDevice(0));cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto context=Garnet::CreateBF16ProjectionContext();if(!context)throw std::runtime_error("Projection context");
    Buffer eviction(256ULL<<20);std::mt19937 generator(78231);std::normal_distribution<float> normal(0,.08f);
    std::cout<<json{{"event","joined_projection_metadata"},{"seed",78231},{"eviction_bytes",eviction.bytes},
        {"precision","BF16 input/weights/output, FP32 accumulation; unchanged production RunBF16Projection; workspace0"},
        {"scope","Synthetic graph-replay component screen; not model/quality/residency acceptance"}}.dump()<<std::endl;
    for(int hidden:{2048,2560}){
        const int vh=hidden==2048?16:32;const std::array<int,4> widths{4096+vh*128,vh*128,vh,vh};
        const int total=std::accumulate(widths.begin(),widths.end(),0);std::vector<__nv_bfloat16> weight(size_t(total)*hidden);
        for(auto& value:weight)value=__float2bfloat16_rn(normal(generator));Buffer w(weight.size()*2);w.put(weight);
        for(int m:{4,8,16,32,64,128,256,512}){
            std::vector<__nv_bfloat16> input(size_t(m)*hidden);for(auto& value:input)value=__float2bfloat16_rn(normal(generator));
            Buffer x(input.size()*2),out(size_t(m)*total*2);x.put(input);
            std::vector<std::unique_ptr<Buffer>> separate;for(int width:widths)separate.push_back(std::make_unique<Buffer>(size_t(m)*width*2));
            auto project=[&](const void* matrix,void* output,int n){if(Garnet::RunBF16Projection(context,x.p,matrix,output,m,hidden,n,stream,nullptr,0))throw std::runtime_error("Production GEMM failed");};
            auto baseline=[&]{int offset=0;for(int i=0;i<4;++i){project(static_cast<const __nv_bfloat16*>(w.p)+size_t(offset)*hidden,separate[i]->p,widths[i]);offset+=widths[i];}};
            auto joined=[&]{project(w.p,out.p,total);};
            auto split=[&]{joined();int offset=0;for(int i=0;i<4;++i){check(cudaMemcpy2DAsync(separate[i]->p,widths[i]*2,static_cast<const __nv_bfloat16*>(out.p)+offset,total*2,widths[i]*2,m,cudaMemcpyDeviceToDevice,stream));offset+=widths[i];}};
            Graph a(baseline,stream),b(joined,stream),c(split,stream);
            a.run(stream);check(cudaStreamSynchronize(stream));std::vector<std::vector<__nv_bfloat16>> reference;for(auto& buffer:separate)reference.push_back(buffer->get<__nv_bfloat16>());
            c.run(stream);check(cudaStreamSynchronize(stream));auto values=out.get<__nv_bfloat16>();json math=json::array();int offset=0;
            for(int i=0;i<4;++i){math.push_back(verify(input,weight,reference[i],values,m,hidden,total,offset,widths[i]));auto copied=separate[i]->get<__nv_bfloat16>();for(size_t j=0;j<copied.size();++j)if(bits(copied[j])!=bits(values[(j/widths[i])*total+offset+j%widths[i]]))throw std::runtime_error("Output split bits changed");offset+=widths[i];}
            auto warm=timings({&a,&b,&c},eviction,stream,false),swept=timings({&a,&b,&c},eviction,stream,true);
            x.guards();w.guards();out.guards();eviction.guards();for(auto& buffer:separate)buffer->guards();
            auto unchangedX=x.get<__nv_bfloat16>(),unchangedW=w.get<__nv_bfloat16>();if(std::memcmp(input.data(),unchangedX.data(),x.bytes)||std::memcmp(weight.data(),unchangedW.data(),w.bytes))throw std::runtime_error("Projection input mutated");
            std::cout<<json{{"event","joined_projection_case"},{"model",hidden==2048?"2b":"4b"},{"m",m},{"k",hidden},{"group_widths",widths},{"joined_n",total},
                {"warm_graph_median_us",warm},{"eviction_sweep_graph_median_us",swept},{"provider_order",{"four_separate","joined_only","joined_plus_exact_splits"}},
                {"single_weight_allocation_bytes",w.bytes},{"input_bytes_unchanged",true},{"output_guards_intact",true},{"output_splits_bit_exact",true},{"math",math},
                {"scope","Nine rotated graph repetitions per provider; eviction sweep is not validated L2 residency; no full-model prediction"}}.dump()<<std::endl;
        }
    }
    context.reset();check(cudaStreamDestroy(stream));std::cout<<"QWEN35_JOINED_PROJECTION_SCREEN_PASS"<<std::endl;return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}}
