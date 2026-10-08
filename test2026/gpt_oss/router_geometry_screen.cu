// SPDX-License-Identifier: Apache-2.0
// Private score-only geometry screen. No inference selector or production changes.
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#define CUDA_OK(call) do { auto e=(call); if(e!=cudaSuccess){ std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(1);}}while(0)
namespace {
uint32_t bits(float f){uint32_t n;std::memcpy(&n,&f,4);return n;}
float rounded(float f){auto n=bits(f);n=(n+0x7fffU+((n>>16)&1U))&0xffff0000U;std::memcpy(&f,&n,4);return f;}
template<int Groups>
__global__ void privateRouterGeometry(const float* x,const float* weights,const float* bias,
    float* logits,int tokens,Garnet::GptOssOptions o){
    static_assert(Groups==1 || Groups==2 || Groups==4);
#if __CUDA_ARCH__ >= 800
    namespace W=nvcuda::wmma;
    constexpr int Experts=16*Groups,Threads=32*Groups;
    const int first=blockIdx.x*16,firstExpert=blockIdx.y*Experts,warp=threadIdx.x/32;
    __shared__ __align__(32) __nv_bfloat16 a[16*32],b[Experts*32];
    __shared__ __align__(32) float result[Groups][16*16];
    W::fragment<W::matrix_a,16,16,16,__nv_bfloat16,W::row_major> af;
    W::fragment<W::matrix_b,16,16,16,__nv_bfloat16,W::col_major> bf;
    W::fragment<W::accumulator,16,16,16,float> acc;W::fill_fragment(acc,0.f);
    for(int start=0;start<o.hidden;start+=32){
        for(int i=threadIdx.x;i<16*32;i+=Threads){
            const int token=first+i/32,k=start+i%32;
            a[i]=__float2bfloat16(token<tokens && k<o.hidden?x[size_t(token)*o.hidden+k]:0.f);
        }
        for(int i=threadIdx.x;i<Experts*32;i+=Threads){
            const int expert=firstExpert+i/32,k=start+i%32;
            b[i]=__float2bfloat16(expert<o.experts && k<o.hidden?weights[size_t(expert)*o.hidden+k]:0.f);
        }
        __syncthreads();
        for(int offset=0;offset<32;offset+=16){
            W::load_matrix_sync(af,a+offset,32);
            W::load_matrix_sync(bf,b+warp*16*32+offset,32);
            W::mma_sync(acc,af,bf,acc);
        }
        __syncthreads();
    }
    W::store_matrix_sync(result[warp],acc,16,W::mem_row_major);__syncthreads();
    for(int i=threadIdx.x;i<16*Experts;i+=Threads){
        const int token=first+i/Experts,e=i%Experts,expert=firstExpert+e;
        if(token<tokens && expert<o.experts)
            logits[size_t(token)*o.experts+expert]=__bfloat162float(__float2bfloat16(result[e/16][(i/Experts)*16+e%16]+bias[expert]));
    }
#else
    // Host refuses this architecture. A fallback cannot establish TC correctness.
    asm("trap;");
#endif
}
struct Buffer{
    float* p=nullptr;size_t n;
    explicit Buffer(size_t count):n(count){CUDA_OK(cudaMalloc(&p,n*4));}
    void upload(const std::vector<float>& x,cudaStream_t s){if(x.size()!=n)std::exit(2);CUDA_OK(cudaMemcpyAsync(p,x.data(),n*4,cudaMemcpyHostToDevice,s));CUDA_OK(cudaStreamSynchronize(s));}
    std::vector<float> read(){std::vector<float> x(n);CUDA_OK(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
    ~Buffer(){CUDA_OK(cudaFree(p));}
};
void save(const std::filesystem::path& path,const std::vector<float>& x){
    if(std::filesystem::exists(path))std::exit(5);FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)std::exit(5);
    const bool failed=std::fwrite(x.data(),4,x.size(),f)!=x.size();const int closed=std::fclose(f);if(failed||closed)std::exit(5);
}
cudaError_t launch(int groups,const float* x,const float* w,const float* b,float* y,
    int tokens,const Garnet::GptOssOptions& o,cudaStream_t s){
    if(!x || !w || !b || !y || tokens<=0 || tokens>4096 || o.hidden<=0 || o.hidden>4096 || o.experts<=0 || o.experts>128)return cudaErrorInvalidValue;
    if(groups==0)return Garnet::TestGptOssBatchRouter(x,w,b,y,nullptr,nullptr,tokens,o,16,false,s);
    if(groups==4)privateRouterGeometry<4><<<dim3((tokens+15)/16,(o.experts+63)/64),128,0,s>>>(x,w,b,y,tokens,o);
    else if(groups==2)privateRouterGeometry<2><<<dim3((tokens+15)/16,(o.experts+31)/32),64,0,s>>>(x,w,b,y,tokens,o);
    else if(groups==1)privateRouterGeometry<1><<<dim3((tokens+15)/16,(o.experts+15)/16),32,0,s>>>(x,w,b,y,tokens,o);
    else return cudaErrorInvalidValue;
    return cudaGetLastError();
}
float xValue(int t,int k,int family,int changed){
    if(family==2)return 0.f;
    const int q=((t+changed)*13+k*7+3)%23-11;
    return family?rounded(std::ldexp(float(q)*1.03125f,(k%5)-6)):float(q)/64;
}
float wValue(int e,int k,int family){
    if(family==2)return 0.f;
    const int q=(e*17+k*3+5)%31-15;
    return family?rounded(std::ldexp(float(q)*.515625f,(k%7)-8)):float(q)/128;
}
std::vector<float> oracle(int rows,int hidden,int experts,int family,int changed){
    // Independent FP64 dots for EVERY distinct input row; periodic expansion is exact.
    std::vector<float> table(size_t(23)*experts),expected(size_t(rows)*experts);
    for(int t=0;t<23;++t)for(int e=0;e<experts;++e){
        double sum=family==2?.125f:float(e%7-3)/64;
        for(int k=0;k<hidden;++k)sum+=double(xValue(t,k,family,changed))*wValue(e,k,family);
        table[size_t(t)*experts+e]=rounded(float(sum));
    }
    for(size_t i=0;i<expected.size();++i)expected[i]=table[i%(23*experts)];
    return expected;
}
int parity(const std::filesystem::path& folder,cudaStream_t stream){
    const std::array<std::array<int,3>,15> cases{{{127,31,17},{128,32,63},{129,33,65},
        {144,2879,128},{224,2880,128},{256,2880,128},{448,2880,128},{512,2880,128},
        {513,2880,128},{1024,2880,128},{3584,2880,128},{4096,2880,128},
        {129,4096,127},{512,4096,128},{17,1,1}}};
    size_t matrices=0,values=0,accuracyFailures=0;
    auto checkAccuracy=[&](const std::vector<float>& actual,const std::vector<float>& expected,const std::string& stem){
        for(size_t i=0;i<actual.size();++i)if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>.006f*(1+std::abs(expected[i]))){
            if(accuracyFailures<16)std::fprintf(stderr,"ROUTER_FP64_BOUND_FAIL stem=%s index=%zu actual=%.9g expected=%.9g bound=%.9g\n",stem.c_str(),i,actual[i],expected[i],.006f*(1+std::abs(expected[i])));
            ++accuracyFailures;
        }
    };
    for(size_t c=0;c<cases.size();++c){
        const auto shape=cases[c];const int rows=shape[0],h=shape[1],e=shape[2];
        Garnet::GptOssOptions o;o.hidden=h;o.experts=e;o.topK=1;
        std::vector<float> x(size_t(rows)*h),w(size_t(e)*h),b(e);
        Buffer dx(x.size()),dw(w.size()),db(b.size()),dy(size_t(rows)*e);
        for(int bad:{-1,3,8})if(launch(bad,dx.p,dw.p,db.p,dy.p,rows,o,stream)!=cudaErrorInvalidValue)std::exit(4);
        const std::array<std::array<float*,4>,4> invalidPointers{{{nullptr,dw.p,db.p,dy.p},{dx.p,nullptr,db.p,dy.p},{dx.p,dw.p,nullptr,dy.p},{dx.p,dw.p,db.p,nullptr}}};
        for(auto pointers:invalidPointers)
            if(launch(1,pointers[0],pointers[1],pointers[2],pointers[3],rows,o,stream)!=cudaErrorInvalidValue)std::exit(4);
        for(int bad:{0,4097})if(launch(1,dx.p,dw.p,db.p,dy.p,bad,o,stream)!=cudaErrorInvalidValue)std::exit(4);
        for(int family=0;family<3;++family){
            for(int expert=0;expert<e;++expert){b[expert]=family==2?.125f:float(expert%7-3)/64;for(int k=0;k<h;++k)w[size_t(expert)*h+k]=wValue(expert,k,family);}
            dw.upload(w,stream);db.upload(b,stream);
            for(int changed=0;changed<2;++changed){
                for(int t=0;t<rows;++t)for(int k=0;k<h;++k)x[size_t(t)*h+k]=xValue(t,k,family,changed?131:0);
                dx.upload(x,stream);const auto expected=oracle(rows,h,e,family,changed?131:0);std::vector<float> original;
                for(int groups:{0,4,2,1}){
                    if(changed){
                        cudaGraph_t graph;cudaGraphExec_t exec;
                        CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));CUDA_OK(launch(groups,dx.p,dw.p,db.p,dy.p,rows,o,stream));
                        CUDA_OK(cudaStreamEndCapture(stream,&graph));CUDA_OK(cudaGraphInstantiate(&exec,graph,0));
                        // Change input AFTER graph construction, then replay the same executable.
                        for(int t=0;t<rows;++t)for(int k=0;k<h;++k)x[size_t(t)*h+k]=xValue(t,k,family,132);
                        dx.upload(x,stream);CUDA_OK(cudaGraphLaunch(exec,stream));CUDA_OK(cudaStreamSynchronize(stream));
                        const auto otherExpected=oracle(rows,h,e,family,132);auto other=dy.read();
                        const auto stem="case"+std::to_string(c)+"-family"+std::to_string(family)+"-groups"+std::to_string(groups);
                        save(folder/(stem+"-changed132.bin"),other);save(folder/(stem+"-oracle132.bin"),otherExpected);
                        checkAccuracy(other,otherExpected,stem+"-changed132");
                        // Restore131 without replacing the graph, checking repeated updates/replays.
                        for(int t=0;t<rows;++t)for(int k=0;k<h;++k)x[size_t(t)*h+k]=xValue(t,k,family,131);
                        dx.upload(x,stream);for(int r=0;r<3;++r)CUDA_OK(cudaGraphLaunch(exec,stream));CUDA_OK(cudaStreamSynchronize(stream));
                        CUDA_OK(cudaGraphExecDestroy(exec));CUDA_OK(cudaGraphDestroy(graph));++matrices;values+=other.size();
                    }else {CUDA_OK(launch(groups,dx.p,dw.p,db.p,dy.p,rows,o,stream));CUDA_OK(cudaStreamSynchronize(stream));}
                    const auto actual=dy.read();const auto stem="case"+std::to_string(c)+"-family"+std::to_string(family)+"-groups"+std::to_string(groups)+(changed?"-graph131":"-eager0");
                    save(folder/(stem+".bin"),actual);
                    if(groups==0){original=actual;save(folder/(stem+"-oracle.bin"),expected);}
                    else if(std::memcmp(actual.data(),original.data(),actual.size()*4)){std::fprintf(stderr,"ROUTER_GEOMETRY_BIT_FAIL stem=%s\n",stem.c_str());std::exit(3);}
                    checkAccuracy(actual,expected,stem);
                    std::printf("ROUTER_GEOMETRY_MATRIX_RETAINED case=%zu rows=%d hidden=%d experts=%d family=%d changed=%d groups=%d values=%zu\n",c,rows,h,e,family,changed,groups,actual.size());
                    ++matrices;values+=actual.size();
                }
            }
        }
    }
    std::printf("ROUTER_GEOMETRY_BIT_EQUIVALENCE_COMPLETE cases=15 matrices=%zu values=%zu accuracy_failures=%zu NO_INFERENCE_SELECTION\n",matrices,values,accuracyFailures);
    if(accuracyFailures){std::fprintf(stderr,"ROUTER_FP64_BOUND_FAILURE_FATAL count=%zu NO_QUALITY_OR_TIMING_ACCEPTANCE\n",accuracyFailures);return 3;}
    std::puts("ROUTER_GEOMETRY_PARITY_PASS FULL_FP64_BOUNDS EXACT_ORIGINAL NO_INFERENCE_SELECTION");return 0;
}
}
int main(int argc,char** argv){
    if(argc!=3 || std::string(argv[1])!="parity")return 2;
    const std::filesystem::path folder=argv[2];if(!std::filesystem::is_directory(folder)||!std::filesystem::is_empty(folder))return 2;
    cudaDeviceProp property;CUDA_OK(cudaGetDeviceProperties(&property,0));if(property.major<8)return 2;
    cudaStream_t stream;CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));const int result=parity(folder,stream);CUDA_OK(cudaStreamDestroy(stream));return result;
}
