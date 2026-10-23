// SPDX-License-Identifier: Apache-2.0
// Synthetic cuBLAS tactic/workspace screen, never full-model speed evidence.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(int(s)));}
__global__ void initialize(__nv_bfloat16* p,size_t n,uint32_t seed){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    uint32_t h=uint32_t(i)*747796405u+seed;h=((h>>((h>>28)+4))^h)*277803737u;h=(h>>22)^h;
    p[i]=__float2bfloat16_rn((float(h%2001)-1000.f)/32768.f);
}
int main(){
    try{
        check(cudaSetDevice(0));cublasHandle_t handle;blas(cublasCreate(&handle));
        blas(cublasSetMathMode(handle,static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
        void* scratch;check(cudaMalloc(&scratch,8ULL<<20));
        struct Shape{int m,k,n;};
        Shape shapes[]={{4,2048,6144},{4,6144,2048},{4,2048,4096},{4,2048,512},
            {32,2048,6144},{32,6144,2048},{1,2048,248320},{8,2048,248320},
            {4,2560,9216},{4,9216,2560},{32,2560,9216},{1,2560,248320}};
        for(auto s:shapes){
            __nv_bfloat16 *x,*w,*y;size_t nx=size_t(s.m)*s.k,nw=size_t(s.n)*s.k,ny=size_t(s.m)*s.n;
            check(cudaMalloc(&x,nx*2));check(cudaMalloc(&w,nw*2));check(cudaMalloc(&y,ny*2));
            initialize<<<(nx+255)/256,256>>>(x,nx,13);initialize<<<(nw+255)/256,256>>>(w,nw,719);
            check(cudaGetLastError());check(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> reference(ny),actual(ny);
            const size_t sizes[]={0,16384,65536,262144,1048576,8388608,8388608,1048576,262144,65536,16384,0};
            for(int order=0;order<12;++order){
                size_t workspaceBytes=sizes[order];
                blas(cublasSetWorkspace(handle,workspaceBytes?scratch:nullptr,workspaceBytes));
                const float alpha=1,beta=0;
                auto run=[&]{blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,s.n,s.m,s.k,&alpha,
                    w,CUDA_R_16BF,s.k,x,CUDA_R_16BF,s.k,&beta,y,CUDA_R_16BF,s.n,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));};
                for(int j=0;j<3;++j)run();check(cudaDeviceSynchronize());
                std::vector<float> samples;cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
                for(int j=0;j<20;++j){check(cudaEventRecord(start));run();check(cudaEventRecord(end));check(cudaEventSynchronize(end));
                    float elapsed;check(cudaEventElapsedTime(&elapsed,start,end));samples.push_back(elapsed*1000.f);}
                check(cudaEventDestroy(start));check(cudaEventDestroy(end));
                check(cudaMemcpy(actual.data(),y,ny*2,cudaMemcpyDeviceToHost));
                if(!order)reference=actual;size_t changed=0;float maxError=0;
                for(size_t i=0;i<ny;++i){float a=__bfloat162float(actual[i]),b=__bfloat162float(reference[i]);
                    if(!std::isfinite(a))throw std::runtime_error("Non-finite output");
                    changed+=a!=b;maxError=std::max(maxError,std::abs(a-b));}
                std::sort(samples.begin(),samples.end());
                std::cout<<"{\"rows\":"<<s.m<<",\"inner\":"<<s.k<<",\"columns\":"<<s.n
                    <<",\"order\":"<<order<<",\"workspace_bytes\":"<<workspaceBytes
                    <<",\"median_gpu_us\":"<<(samples[9]+samples[10])*.5f
                    <<",\"different_outputs\":"<<changed<<",\"output_count\":"<<ny
                    <<",\"max_abs_output_difference\":"<<maxError<<"}\n"<<std::flush;
            }
            check(cudaFree(x));check(cudaFree(w));check(cudaFree(y));
        }
        check(cudaFree(scratch));blas(cublasDestroy(handle));std::cout<<"BF16_WORKSPACE_SCREEN_COMPLETE\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
