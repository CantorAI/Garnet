// SPDX-License-Identifier: Apache-2.0
// Unwired bandwidth screen: BF16 operands/output, explicit FP32 accumulation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS "+std::to_string(int(s)));}
__global__ void initialize(__nv_bfloat16* p,size_t count,unsigned seed){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned h=unsigned(i)*747796405u+seed;h=((h>>((h>>28)+4))^h)*277803737u;h=(h>>22)^h;
    p[i]=__float2bfloat16_rn((float(h%2001)-1000.f)/32768.f);
}
__global__ void evict(unsigned* p,size_t count,unsigned round){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)p[i]=p[i]+round+unsigned(i);
}
template<int Rows> __global__ void smallRows(const __nv_bfloat16* x,const __nv_bfloat16* weights,__nv_bfloat16* y,int inner,int columns){
    const int lane=threadIdx.x%32,column=blockIdx.x*(blockDim.x/32)+threadIdx.x/32;if(column>=columns)return;
    float sums[Rows]={};
    const auto* w=reinterpret_cast<const __nv_bfloat162*>(weights+size_t(column)*inner);
    for(int pair=lane;pair<inner/2;pair+=32){
        const float2 weight=__bfloat1622float2(w[pair]);
        #pragma unroll
        for(int row=0;row<Rows;++row){
            const float2 input=__bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x+row*inner)[pair]);
            sums[row]=__fmaf_rn(weight.x,input.x,sums[row]);sums[row]=__fmaf_rn(weight.y,input.y,sums[row]);
        }
    }
    #pragma unroll
    for(int row=0;row<Rows;++row){
        #pragma unroll
        for(int shift=16;shift>0;shift/=2)sums[row]=__fadd_rn(sums[row],__shfl_down_sync(0xffffffff,sums[row],shift));
        if(lane==0)y[size_t(row)*columns+column]=__float2bfloat16_rn(sums[row]);
    }
}
float value(__nv_bfloat16 b){unsigned short bits;std::memcpy(&bits,&b,2);unsigned raw=unsigned(bits)<<16;float f;std::memcpy(&f,&raw,4);return f;}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("Output JSONL path required");std::ofstream log(argv[1]);if(!log)throw std::runtime_error("Output path");
    check(cudaSetDevice(0));cublasHandle_t handle;blas(cublasCreate(&handle));
    blas(cublasSetMathMode(handle,static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    blas(cublasSetWorkspace(handle,nullptr,0));unsigned* eviction;const size_t bytes=256ULL<<20;check(cudaMalloc(&eviction,bytes));check(cudaMemset(eviction,0,bytes));
    cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,0));
    log<<json({{"event","setup"},{"seed",61983},{"eviction_bytes",bytes},{"reported_l2_bytes",device.l2CacheSize},{"precision","BF16 in/out; FP32 multiply-accumulate; no TF32"},{"scope","Unwired synthetic component screen; not proof of cache residency or full-model quality/speed"}}).dump()<<std::endl;
    for(auto shape:std::vector<std::array<int,3>>{{4,2048,6144},{4,6144,2048},{4,2048,512},{8,2048,6144},{4,2560,9216},{4,9216,2560},{1,2048,248320},{4,2048,248320}}){
        int rows=shape[0],inner=shape[1],columns=shape[2];__nv_bfloat16 *x,*w,*y;
        size_t nx=size_t(rows)*inner,nw=size_t(columns)*inner,ny=size_t(rows)*columns;
        check(cudaMalloc(&x,nx*2));check(cudaMalloc(&w,nw*2));check(cudaMalloc(&y,ny*2));
        initialize<<<(nx+255)/256,256>>>(x,nx,61983);initialize<<<(nw+255)/256,256>>>(w,nw,719);check(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> reference(ny),actual(ny),input(nx);check(cudaMemcpy(input.data(),x,nx*2,cudaMemcpyDeviceToHost));
        const float alpha=1,beta=0;
        auto run=[&](int method){
            if(method==0)blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows,inner,&alpha,w,CUDA_R_16BF,inner,x,CUDA_R_16BF,inner,&beta,y,CUDA_R_16BF,columns,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            else {int threads=method==1?128:256,blocks=(columns+threads/32-1)/(threads/32);
                if(rows==1)smallRows<1><<<blocks,threads>>>(x,w,y,inner,columns);
                else if(rows==4)smallRows<4><<<blocks,threads>>>(x,w,y,inner,columns);
                else if(rows==8)smallRows<8><<<blocks,threads>>>(x,w,y,inner,columns);
                else throw std::runtime_error("Profile rows invalid");check(cudaGetLastError());}
        };
        cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
        for(int method=0;method<3;++method){
            run(method);check(cudaDeviceSynchronize());check(cudaMemcpy(actual.data(),y,ny*2,cudaMemcpyDeviceToHost));if(method==0)reference=actual;
            size_t different=0;for(size_t i=0;i<ny;++i){if(!std::isfinite(value(actual[i])))throw std::runtime_error("Nonfinite projection");different+=value(actual[i])!=value(reference[i]);}
            double maxFraction=0;std::vector<__nv_bfloat16> weight(inner);
            for(int sample=0;sample<128;++sample){int row=sample%rows,col=int((uint64_t(sample)*104729+61983)%columns);
                check(cudaMemcpy(weight.data(),w+size_t(col)*inner,inner*2,cudaMemcpyDeviceToHost));double dot=0,absolute=0;
                for(int k=0;k<inner;++k){double product=double(value(input[row*inner+k]))*value(weight[k]);dot+=product;absolute+=std::abs(product);}
                double unit=std::ldexp(1.,-24),gamma=(inner*unit)/(1-inner*unit);
                double bound=gamma*absolute+(std::abs(dot)+gamma*absolute)/256.+std::ldexp(1.,-133);
                double error=std::abs(value(actual[size_t(row)*columns+col])-dot);if(error>bound)throw std::runtime_error("FP64 FP32-accumulation/BF16-rounding bound");maxFraction=std::max(maxFraction,error/bound);
            }
            std::vector<float> cold,warm;
            for(int trial=0;trial<20;++trial){
                evict<<<(bytes/4+255)/256,256>>>(eviction,bytes/4,trial+1);check(cudaGetLastError());
                check(cudaEventRecord(start));run(method);check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));float ms;check(cudaEventElapsedTime(&ms,start,stop));cold.push_back(ms*1000);
                check(cudaEventRecord(start));run(method);check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));check(cudaEventElapsedTime(&ms,start,stop));warm.push_back(ms*1000);
            }
            std::sort(cold.begin(),cold.end());std::sort(warm.begin(),warm.end());
            log<<json({{"rows",rows},{"inner",inner},{"columns",columns},{"method",method==0?"classic_tn":method==1?"warp_rows_128":"warp_rows_256"},{"cold_median_us",(cold[9]+cold[10])/2},{"warm_median_us",(warm[9]+warm[10])/2},{"different_bf16_from_classic",different},{"outputs",ny},{"fp64_samples",128},{"max_fixed_bound_fraction",maxFraction},{"scope","Standalone component only; fixed-order per-method cold/warm pairs, no inference claim"}}).dump()<<std::endl;
        }
        check(cudaEventDestroy(start));check(cudaEventDestroy(stop));check(cudaFree(x));check(cudaFree(w));check(cudaFree(y));std::cout<<"SMALL_ROWS_PASS "<<rows<<" "<<inner<<" "<<columns<<std::endl;
    }
    check(cudaFree(eviction));blas(cublasDestroy(handle));std::cout<<"BF16_SMALL_ROWS_COMPLETE"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
