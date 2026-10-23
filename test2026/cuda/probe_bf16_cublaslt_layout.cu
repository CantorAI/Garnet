// SPDX-License-Identifier: Apache-2.0
// Unwired cold-matrix diagnostic. Never run alongside a model/engine cohort.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace {
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void blas(cublasStatus_t x){if(x!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(x)));}
struct Buffer{void* p=nullptr;size_t bytes;explicit Buffer(size_t n):bytes(n){check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}
    template<class T>void put(const std::vector<T>& v){if(v.size()*sizeof(T)!=bytes)throw std::runtime_error("Upload size");check(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get(){std::vector<T> v(bytes/sizeof(T));check(cudaMemcpy(v.data(),p,bytes,cudaMemcpyDeviceToHost));return v;}
};
__global__ void evict(uint32_t* values,size_t words){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)values[i]=uint32_t(i*2654435761ULL);
}
float asFloat(__nv_bfloat16 x){return __bfloat162float(x);}
template<class F>float time(F call,Buffer& eviction,cudaStream_t stream,bool cold){
    for(int i=0;i<3;++i)blas(call());check(cudaStreamSynchronize(stream));cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));std::vector<float> times;
    for(int i=0;i<9;++i){
        if(cold)evict<<<2048,256,0,stream>>>((uint32_t*)eviction.p,eviction.bytes/4);
        check(cudaEventRecord(a,stream));blas(call());check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));times.push_back(ms*1000);
    }
    cudaEventDestroy(a);cudaEventDestroy(b);std::sort(times.begin(),times.end());return times[times.size()/2];
}
uint16_t raw(__nv_bfloat16 x){uint16_t bits;memcpy(&bits,&x,2);return bits;}
struct Layout{
    cublasLtMatmulDesc_t operation=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    Layout(int m,int k,int n,bool kn){
        blas(cublasLtMatmulDescCreate(&operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=kn?CUBLAS_OP_N:CUBLAS_OP_T;
        blas(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        blas(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,kn?n:k,kn?k:n,kn?n:k));
        blas(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k));blas(cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n));
    }
    ~Layout(){if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);if(operation)cublasLtMatmulDescDestroy(operation);}
};
}
int main(int argc,char** argv){try{
    std::string mode=argc==2?argv[1]:"full";
    if(argc>2||(mode!="full"&&mode!="quick"&&mode!="decode"&&mode!="prefill"))throw std::runtime_error("Expected quick/full/decode/prefill mode");
    bool quick=mode=="quick";check(cudaSetDevice(0));cudaStream_t stream;check(cudaStreamCreate(&stream));
    cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,0));
    std::cout<<json{{"event","diagnostic_metadata"},{"device",properties.name},{"sm",properties.major*10+properties.minor},{"l2_cache_bytes",properties.l2CacheSize},
        {"eviction_bytes",256ULL<<20},{"seed",61983},{"quick",quick},{"mode",mode},{"source_precision","BF16 inputs/output, FP32 accumulation, no output-type reduction"}}.dump()<<"\n";
    cublasHandle_t classic;blas(cublasCreate(&classic));blas(cublasSetStream(classic,stream));blas(cublasSetMathMode(classic,static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));blas(cublasSetWorkspace(classic,nullptr,0));
    cublasLtHandle_t lt;blas(cublasLtCreate(&lt));Buffer eviction(256ULL<<20),workspace(8ULL<<20);check(cudaMemset(eviction.p,0,eviction.bytes));std::mt19937 generator(61983);std::normal_distribution<float> normal(0,.08f);const float alpha=1,beta=0;
    std::vector<std::pair<int,int>> shapes=quick?std::vector<std::pair<int,int>>{{2048,6144}}:std::vector<std::pair<int,int>>{{2048,6144},{6144,2048},{2560,9216},{9216,2560}};
    std::vector<int> rows=quick?std::vector<int>{4,16}:mode=="decode"?std::vector<int>{4,8,16,32,64}:
        mode=="prefill"?std::vector<int>{64,128,256,512}:std::vector<int>{4,16,32};
    for(const auto& shape:shapes)for(int m:rows){
        const int k=shape.first,n=shape.second;
        std::vector<__nv_bfloat16> query(size_t(m)*k),weight(size_t(n)*k),transposed(weight.size());
        for(auto& x:query)x=__float2bfloat16_rn(normal(generator));for(auto& x:weight)x=__float2bfloat16_rn(normal(generator));
        for(int j=0;j<n;++j)for(int d=0;d<k;++d)transposed[size_t(d)*n+j]=weight[size_t(j)*k+d];
        Buffer q(query.size()*2),w(weight.size()*2),wt(weight.size()*2),out(size_t(m)*n*2);q.put(query);w.put(weight);wt.put(transposed);std::vector<__nv_bfloat16> baseline;
        auto verify=[&](const std::vector<__nv_bfloat16>& values){
            size_t different=0;for(size_t i=0;i<values.size();++i)if(raw(values[i])!=raw(baseline[i]))++different;
            double maxFraction=0;int samples=0;double u=std::ldexp(1.,-24),gamma=k*u/(1-k*u);
            for(int row=0;row<std::min(m,2);++row)for(int col=0;col<64;++col){
                int index=col*(n-1)/63;double dot=0,absolute=0;
                for(int d=0;d<k;++d){double term=double(asFloat(query[size_t(row)*k+d]))*asFloat(weight[size_t(index)*k+d]);dot+=term;absolute+=std::abs(term);}
                double bound=gamma*absolute+(std::abs(dot)+gamma*absolute)/256.+std::ldexp(1.,-133);
                double value=asFloat(values[size_t(row)*n+index]);double error=std::abs(value-dot);if(!std::isfinite(value)||error>bound)throw std::runtime_error("Independent FP64 FP32-accumulation/BF16-rounding bound failed");maxFraction=std::max(maxFraction,error/bound);++samples;
            }return json{{"bf16_values_different_from_classic_tn",different},{"fp64_samples",samples},{"max_fixed_bound_fraction",maxFraction}};
        };
        for(bool kn:{false,true}){
            void* matrix=kn?wt.p:w.p;Layout layout(m,k,n,kn);
            auto gemm=[&]{return cublasGemmEx(classic,kn?CUBLAS_OP_N:CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,matrix,CUDA_R_16BF,kn?n:k,q.p,CUDA_R_16BF,k,&beta,out.p,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);};
            blas(gemm());check(cudaStreamSynchronize(stream));auto values=out.get<__nv_bfloat16>();if(!kn)baseline=values;
            float cold=time(gemm,eviction,stream,true),warm=time(gemm,eviction,stream,false);
            std::cout<<json{{"m",m},{"k",k},{"n",n},{"layout",kn?"KN/NN":"NK/TN"},{"provider","GemmEx"},{"cold_median_us",cold},{"warm_median_us",warm},{"workspace_bytes",0},{"math",verify(values)},
                {"scope","Synthetic single matrix with256MiB eviction sweep; not validated L2 residency or model speed"}}.dump()<<"\n";
            for(size_t limit:{size_t(0),size_t(16384),size_t(8ULL<<20)}){
                cublasLtMatmulPreference_t preference;blas(cublasLtMatmulPreferenceCreate(&preference));blas(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
                uint32_t mask=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;blas(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&mask,sizeof(mask)));
                cublasLtMatmulHeuristicResult_t choices[24];int returned=0;blas(cublasLtMatmulAlgoGetHeuristic(lt,layout.operation,layout.a,layout.b,layout.c,layout.c,preference,24,choices,&returned));cublasLtMatmulPreferenceDestroy(preference);
                for(int i=0;i<returned;++i){auto& choice=choices[i];uint64_t flags=0;size_t written=0;uint32_t reduction=0;int id=-1;
                    if(choice.state!=CUBLAS_STATUS_SUCCESS||choice.workspaceSize>limit)continue;
                    blas(cublasLtMatmulAlgoCapGetAttribute(&choice.algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written));
                    blas(cublasLtMatmulAlgoConfigGetAttribute(&choice.algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction),&written));
                    if((flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F||
                        (flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK)!=CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF||
                        (reduction!=CUBLASLT_REDUCTION_SCHEME_NONE&&reduction!=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE))continue;
                    blas(cublasLtMatmulAlgoConfigGetAttribute(&choice.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written));
                    auto run=[&]{return cublasLtMatmul(lt,layout.operation,&alpha,matrix,layout.a,q.p,layout.b,&beta,out.p,layout.c,out.p,layout.c,&choice.algo,limit?workspace.p:nullptr,limit,stream);};
                    auto status=run();if(status!=CUBLAS_STATUS_SUCCESS){std::cout<<json{{"m",m},{"k",k},{"n",n},{"layout",kn?"KN/NN":"NK/TN"},{"provider","Lt"},{"algo_id",id},{"status",int(status)}}.dump()<<"\n";continue;}
                    check(cudaStreamSynchronize(stream));values=out.get<__nv_bfloat16>();auto math=verify(values);
                    cold=time(run,eviction,stream,true);warm=time(run,eviction,stream,false);
                    std::cout<<json{{"m",m},{"k",k},{"n",n},{"layout",kn?"KN/NN":"NK/TN"},{"provider","Lt"},{"algo_id",id},{"numerical_flags",flags},{"reduction_scheme",reduction},
                        {"workspace_limit_bytes",limit},{"workspace_required_bytes",choice.workspaceSize},{"cold_median_us",cold},{"warm_median_us",warm},{"math",math},
                        {"scope","Synthetic isolated matrix/eviction diagnostic, no full-model timing or quality acceptance"}}.dump()<<"\n";
                }
            }
        }
    }
    cublasLtDestroy(lt);cublasDestroy(classic);cudaStreamDestroy(stream);std::cout<<"BF16_CUBLASLT_LAYOUT_COMPLETE\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
