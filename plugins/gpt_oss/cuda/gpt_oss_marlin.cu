// SPDX-License-Identifier: Apache-2.0
#define MARLIN_NAMESPACE_NAME GarnetMarlin
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
#include "gpt_oss_marlin.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace Garnet {
namespace {
bool debugMarlin() {
    static const bool enabled = std::getenv("GARNET_GPT_OSS_DEBUG_MARLIN") != nullptr;
    return enabled;
}
int marlinCtasPerSm() {
    static const int count = [] {
        const char* value = std::getenv("GARNET_GPT_OSS_MARLIN_CTAS_PER_SM");
        const int parsed = value ? std::atoi(value) : 2;
        return parsed == 1 || parsed == 4 ? parsed : 2;
    }();
    return count;
}
void reportMarlin(const char* message, int a = 0, int b = 0) {
    static std::atomic<int> count{0};
    if (debugMarlin() && count.fetch_add(1) < 128) std::fprintf(stderr, "GPT-OSS Marlin: %s (%d, %d)\n", message, a, b);
}
struct Geometry {
    int upK,upN,downK,downN;
    explicit Geometry(const GptOssOptions& o) : upK((o.hidden+63)/64*64),
        upN((2*o.intermediate+127)/128*128), downK((o.intermediate+127)/128*128),
        downN((o.hidden+63)/64*64) {}
};
bool supported(int tokens,const GptOssOptions& o) {
    return tokens>0 && tokens<=8 && o.hidden>0 && o.hidden<=16384 && o.hidden%32==0 &&
        o.intermediate>0 && o.intermediate<=65536 && o.intermediate%32==0 &&
        o.experts>0 && o.experts<=256 && o.topK>0 && o.topK<=8 && o.topK<=o.experts &&
        o.tpRank>=-1 && o.tpRank<2;
}
struct Layout {
    size_t bytes=0,selected,probabilities,routeLogits,a,up,activation,down,sorted,experts,padded,locks,tmp;
    size_t take(size_t size) { bytes=(bytes+15)&~size_t(15);size_t offset=bytes;bytes+=size;return offset; }
    Layout(int tokens,const GptOssOptions& o) {
        Geometry g(o); const size_t slots=size_t(tokens)*o.topK, n=std::max(g.upN,g.downN);
        selected=take(slots*4);probabilities=take(slots*4);
        routeLogits=take(size_t(tokens)*o.experts*4);a=take(size_t(tokens)*g.upK*2);
        up=take(slots*g.upN*2);activation=take(slots*g.downK*2);down=take(slots*g.downN*2);
        sorted=take(slots*8*4);experts=take(slots*4);padded=take(4);
        locks=take(size_t(o.experts)*(n/64)*16*4);
        // Matches upstream's FP32 partial-reduction bound, conservatively sized
        // for at most 512 SMs. The actual device is checked before execution.
        tmp=take(2*std::min(n*slots*8,size_t(512)*4*8*256)*4);
    }
};
template<class T> T* at(void* p,size_t offset) { return reinterpret_cast<T*>(static_cast<char*>(p)+offset); }
__device__ float rounded(float x) { return __bfloat162float(__float2bfloat16(x)); }
// Marlin's target tile mapping follows vLLM v0.31.0's Apache-licensed
// gptq_marlin_repack.cu, marlin_utils.py and marlin_utils_fp4.py.
__global__ void repackOriginal(const unsigned char* original,unsigned* packed,
    int experts,int rank,int originalK,int originalN,int paddedK,int paddedN) {
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t perExpert=size_t(paddedK)*paddedN/8;
    if(index>=size_t(experts)*perExpert) return;
    int localExpert=index/perExpert,expert=rank<0?localExpert:2*localExpert+rank;
    if(expert>=experts)return;size_t local=index%perExpert;
    int lane=(local%128)/4,warp=local%4,tile=local/128;
    int kTile=tile/(paddedN/64),nTile=tile%(paddedN/64);
    int firstN=nTile*64+warp*16+lane/4,firstK=kTile*16+(lane%4)*2;
    constexpr int offsets[4]={0,1,8,9},permutation[8]={0,2,4,6,1,3,5,7};
    unsigned result=0;
    #pragma unroll
    for(int p=0;p<8;++p) {
        int v=permutation[p],n=firstN+(v>=4?8:0),k=firstK+offsets[v%4];unsigned code=0;
        if(n<originalN && k<originalK) {
            unsigned byte=original[(size_t(expert)*originalN+n)*(originalK/2)+k/2];
            code=(byte>>((k%2)*4))&15;
        }
        result|=code<<(p*4);
    }
    packed[index]=result;
}
__global__ void repackScales(const unsigned char* original,unsigned char* packed,
    int experts,int rank,int originalK,int originalN,int paddedK,int paddedN) {
    size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t perExpert=size_t(paddedK/32)*paddedN;
    if(index>=size_t(experts)*perExpert) return;
    int localExpert=index/perExpert,expert=rank<0?localExpert:2*localExpert+rank;
    if(expert>=experts)return;
    int kGroup=(index%perExpert)/paddedN,n=(index%perExpert)%paddedN;
    n=(n&~3)|((n&1)<<1)|((n&2)>>1);
    n=(n/64)*64+(n%64)/8+8*(n%8);
    packed[index]=n<originalN && kGroup<originalK/32
        ?original[(size_t(expert)*originalN+n)*(originalK/32)+kGroup]:127;
}
__global__ void checkScales(const unsigned char* scales,size_t size,int* invalid) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    // Extreme/subnormal/reserved scales retain the exact original kernel path.
    if(i<size && (scales[i]<2 || scales[i]>249)) atomicExch(invalid,1);
}
__global__ void convertInput(const float* x,nv_bfloat16* a,int tokens,int width,int paddedWidth) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*paddedWidth)return;
    int row=i/paddedWidth,d=i%paddedWidth;a[i]=__float2bfloat16(d<width?x[row*width+d]:0);
}
__global__ void metadata(const int* selected,int* sorted,int* experts,int* padded,
    int slots,int expertCount,int rank) {
    __shared__ int counts[256],starts[256];int e=threadIdx.x;
    for(int i=e;i<slots*8;i+=256)sorted[i]=slots;
    int localCount=rank<0?expertCount:(expertCount+1-rank)/2;
    int globalExpert=rank<0?e:2*e+rank;
    int count=0;if(e<localCount)for(int s=0;s<slots;++s)count+=selected[s]==globalExpert;
    counts[e]=count;__syncthreads();
    if(!e) {int total=0;for(int i=0;i<localCount;++i){starts[i]=total;total+=(counts[i]+7)/8*8;}*padded=total;}
    __syncthreads();
    if(e<localCount && count) {
        int row=0;for(int s=0;s<slots;++s)if(selected[s]==globalExpert)sorted[starts[e]+row++]=s;
        for(int i=0;i<count;i+=8)experts[(starts[e]+i)/8]=e;
    }
}
__global__ void activation(const nv_bfloat16* up,const float* bias,const int* selected,
    nv_bfloat16* output,int slots,int upWidth,int paddedWidth,GptOssOptions o) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=slots*paddedWidth)return;
    int slot=index/paddedWidth,i=index%paddedWidth;
    if(i>=o.intermediate){output[index]=__float2bfloat16(0);return;}
    if(o.tpRank>=0 && selected[slot]%2!=o.tpRank){output[index]=__float2bfloat16(0);return;}
    int row=selected[slot]*2*o.intermediate+2*i;
    float gate=fminf(rounded(__bfloat162float(up[size_t(slot)*upWidth+2*i])+bias[row]),o.limit);
    float u=fminf(o.limit,fmaxf(-o.limit,rounded(__bfloat162float(up[size_t(slot)*upWidth+2*i+1])+bias[row+1])));
    float glu=rounded(gate*rounded(1.f/(1.f+expf(-rounded(1.702f*gate)))));
    output[index]=__float2bfloat16(glu*rounded(u+1));
}
__global__ void combine(const nv_bfloat16* down,const float* bias,const int* selected,
    const float* probabilities,float* y,int tokens,int width,GptOssOptions o) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=tokens*o.hidden)return;
    int token=index/o.hidden,d=index%o.hidden;float value=0;
    for(int k=0;k<o.topK;++k) {
        int slot=token*o.topK+k,expert=selected[slot];
        if(o.tpRank>=0 && expert%2!=o.tpRank)continue;
        value+=rounded(__bfloat162float(down[size_t(slot)*width+d])+bias[expert*o.hidden+d])*probabilities[slot];
    }
    y[index]=o.tpRank<0?rounded(value):value;
}
using UpKernel = void(*)(const int4*,const int4*,int4*,int4*,const int4*,const float*,
    const int4*,const float*,const int4*,const int32_t*,const int32_t*,const int32_t*,
    const float*,int,bool,int,int,int,int*,bool,bool,bool);
UpKernel upKernel() {return GarnetMarlin::Marlin<garnet_marlin_types::kBFloat16.id(),
    garnet_marlin_types::kFE2M1f.id(),garnet_marlin_types::kBFloat16.id(),
    garnet_marlin_types::kFE8M0fnu.id(),128,1,8,4,true,4,2,false>;}
UpKernel downKernel() {return GarnetMarlin::Marlin<garnet_marlin_types::kBFloat16.id(),
    garnet_marlin_types::kFE2M1f.id(),garnet_marlin_types::kBFloat16.id(),
    garnet_marlin_types::kFE8M0fnu.id(),128,1,4,8,true,4,2,false>;}
}
struct GptOssMarlin::State {
    GptOssOptions o;Geometry g;int device=-1,sms=0,blockedReason=0;bool ready=false,blocked=false;
    std::array<void*,4> weights{};std::array<const void*,4> sources{};
    explicit State(const GptOssOptions& options):o(options),g(options) {}
    ~State(){release();}
    void release() {
        if(device<0)return;int previous=0;cudaGetDevice(&previous);cudaSetDevice(device);
        for(auto& p:weights){if(p)cudaFree(p);p=nullptr;}
        cudaSetDevice(previous);ready=false;
    }
    cudaError_t prepare(const void* const* in,cudaStream_t stream) {
        std::array<const void*,4> pointers{in[3],in[4],in[6],in[7]};
        if(blocked && pointers==sources) {
            reportMarlin(blockedReason == 1 ? "device capability rejected" : "MXFP4 scales rejected", blockedReason, device);
            return cudaErrorNotSupported;
        }
        if(ready && pointers==sources)return cudaSuccess;
        release();blocked=false;sources=pointers;
        auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
        cudaDeviceProp prop{};status=cudaGetDeviceProperties(&prop,device);if(status!=cudaSuccess)return status;
        if(prop.major<8 || prop.multiProcessorCount>512){blocked=true;blockedReason=1;reportMarlin("device capability rejected", prop.major, prop.multiProcessorCount);return cudaErrorNotSupported;}
        sms=prop.multiProcessorCount;
        int* invalid=nullptr;status=cudaMalloc((void**)&invalid,4);if(status!=cudaSuccess)return status;
        status=cudaMemsetAsync(invalid,0,4,stream);
        const size_t upScales=size_t(o.experts)*2*o.intermediate*(o.hidden/32);
        const size_t downScales=size_t(o.experts)*o.hidden*(o.intermediate/32);
        if(status==cudaSuccess) {
            checkScales<<<(upScales+255)/256,256,0,stream>>>((const unsigned char*)in[4],upScales,invalid);
            status=cudaGetLastError();
        }
        if(status==cudaSuccess) {
            checkScales<<<(downScales+255)/256,256,0,stream>>>((const unsigned char*)in[7],downScales,invalid);
            status=cudaGetLastError();
        }
        int result=0;
        if(status==cudaSuccess)status=cudaMemcpyAsync(&result,invalid,4,cudaMemcpyDeviceToHost,stream);
        if(status==cudaSuccess)status=cudaStreamSynchronize(stream);
        cudaFree(invalid);if(status!=cudaSuccess)return status;
        if(result){blocked=true;blockedReason=2;reportMarlin("MXFP4 scales rejected", result, device);return cudaErrorNotSupported;}
        const int localExperts=o.tpRank<0?o.experts:(o.experts+1-o.tpRank)/2;
        const size_t sizes[]{size_t(localExperts)*g.upN*g.upK/2,size_t(localExperts)*g.upN*g.upK/32,
            size_t(localExperts)*g.downN*g.downK/2,size_t(localExperts)*g.downN*g.downK/32};
        for(int i=0;i<4;++i){status=cudaMalloc(&weights[i],sizes[i]);if(status!=cudaSuccess){release();return status;}}
        repackOriginal<<<(sizes[0]/4+255)/256,256,0,stream>>>((const unsigned char*)in[3],(unsigned*)weights[0],o.experts,o.tpRank,o.hidden,2*o.intermediate,g.upK,g.upN);
        repackScales<<<(sizes[1]+255)/256,256,0,stream>>>((const unsigned char*)in[4],(unsigned char*)weights[1],o.experts,o.tpRank,o.hidden,2*o.intermediate,g.upK,g.upN);
        repackOriginal<<<(sizes[2]/4+255)/256,256,0,stream>>>((const unsigned char*)in[6],(unsigned*)weights[2],o.experts,o.tpRank,o.intermediate,o.hidden,g.downK,g.downN);
        repackScales<<<(sizes[3]+255)/256,256,0,stream>>>((const unsigned char*)in[7],(unsigned char*)weights[3],o.experts,o.tpRank,o.intermediate,o.hidden,g.downK,g.downN);
        status=cudaGetLastError();if(status==cudaSuccess)status=cudaStreamSynchronize(stream);
        if(status==cudaSuccess)status=cudaFuncSetAttribute(upKernel(),cudaFuncAttributeMaxDynamicSharedMemorySize,27136);
        if(status==cudaSuccess)status=cudaFuncSetAttribute(downKernel(),cudaFuncAttributeMaxDynamicSharedMemorySize,35200);
        if(status!=cudaSuccess){release();return status;}ready=true;return cudaSuccess;
    }
};
GptOssMarlin::GptOssMarlin(const GptOssOptions& o):m_state(new State(o)) {}
GptOssMarlin::~GptOssMarlin()=default;
size_t GptOssMarlin::Workspace(int tokens,const GptOssOptions& o) {
    return supported(tokens,o)?Layout(tokens,o).bytes:0;
}
cudaError_t GptOssMarlin::Run(const void* const* in,float* y,void* workspace,int tokens,cudaStream_t stream) {
    auto& s=*m_state;if(!supported(tokens,s.o)){reportMarlin("shape rejected", tokens, s.o.hidden);return cudaErrorNotSupported;}
    reportMarlin("decode candidate", tokens, s.o.hidden);
    int device=-1;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return status;
    if(s.device>=0 && s.device!=device)return cudaErrorInvalidDevice;
    status=s.prepare(in,stream);if(status!=cudaSuccess)return status;
    const auto& o=s.o;const auto& g=s.g;const Layout l(tokens,o);int slots=tokens*o.topK;
    auto selected=at<int>(workspace,l.selected);auto probabilities=at<float>(workspace,l.probabilities);
    status=RunGptOssMoeRoute(in,selected,probabilities,at<float>(workspace,l.routeLogits),tokens,o,stream);
    if(status!=cudaSuccess)return status;
    convertInput<<<(tokens*g.upK+255)/256,256,0,stream>>>((const float*)in[0],at<nv_bfloat16>(workspace,l.a),tokens,o.hidden,g.upK);
    metadata<<<1,256,0,stream>>>(selected,at<int>(workspace,l.sorted),at<int>(workspace,l.experts),at<int>(workspace,l.padded),slots,o.experts,o.tpRank);
    status=cudaMemsetAsync(at<int>(workspace,l.locks),0,size_t(o.experts)*(std::max(g.upN,g.downN)/64)*16*4,stream);
    if(status!=cudaSuccess)return status;
    auto up=upKernel();up<<<s.sms*marlinCtasPerSm(),128,27136,stream>>>(at<int4>(workspace,l.a),(const int4*)s.weights[0],at<int4>(workspace,l.up),at<int4>(workspace,l.tmp),
        nullptr,nullptr,(const int4*)s.weights[1],nullptr,nullptr,at<int>(workspace,l.sorted),at<int>(workspace,l.experts),at<int>(workspace,l.padded),
        nullptr,o.topK,false,tokens,g.upN,g.upK,at<int>(workspace,l.locks),false,false,true);
    activation<<<(slots*g.downK+255)/256,256,0,stream>>>(at<nv_bfloat16>(workspace,l.up),(const float*)in[5],selected,at<nv_bfloat16>(workspace,l.activation),slots,g.upN,g.downK,o);
    auto down=downKernel();down<<<s.sms*marlinCtasPerSm(),128,35200,stream>>>(at<int4>(workspace,l.activation),(const int4*)s.weights[2],at<int4>(workspace,l.down),at<int4>(workspace,l.tmp),
        nullptr,nullptr,(const int4*)s.weights[3],nullptr,nullptr,at<int>(workspace,l.sorted),at<int>(workspace,l.experts),at<int>(workspace,l.padded),
        nullptr,1,false,slots,g.downN,g.downK,at<int>(workspace,l.locks),false,false,true);
    combine<<<(tokens*o.hidden+255)/256,256,0,stream>>>(at<nv_bfloat16>(workspace,l.down),(const float*)in[8],selected,probabilities,y,tokens,g.downN,o);
    return cudaGetLastError();
}
}
