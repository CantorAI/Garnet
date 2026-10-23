// SPDX-License-Identifier: Apache-2.0
#include "modelopt_cublaslt.h"
#include "modelopt_nvfp4.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>
void CUDA(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void BLAS(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS status=%d\n",int(s));throw std::runtime_error("FP4 cuBLASLt operation failed");}}
size_t Offset(int row,int col,int columns){
    return size_t(row/128)*128*columns+size_t(col/4)*512+(row%32)*16+((row%128)/32)*4+col%4;
}
int main(){
    try{
        constexpr int rows=256,k=256,blocks=k/16;
        cublasLtHandle_t handle=nullptr;BLAS(cublasLtCreate(&handle));
        cudaStream_t stream=nullptr;CUDA(cudaStreamCreate(&stream));
        for(int batch:{1,16}){
            std::vector<unsigned char> w(rows*k/2,0x22),x(batch*k/2,0x22);
            std::vector<unsigned char> ws(rows*blocks),xs(128*blocks,0);
            const unsigned char codes[]={56,64,72};
            const float values[]={1,2,4};
            for(int r=0;r<rows;++r)for(int b=0;b<blocks;++b)ws[Offset(r,b,blocks)]=codes[(r+b)%3];
            for(int r=0;r<batch;++r)for(int b=0;b<blocks;++b)xs[Offset(r,b,blocks)]=codes[(r+b+1)%3];
            void *dw=nullptr,*dx=nullptr,*dsw=nullptr,*dsx=nullptr,*dy=nullptr,*workspace=nullptr;
            CUDA(cudaMalloc(&dw,w.size()));CUDA(cudaMalloc(&dx,x.size()));CUDA(cudaMalloc(&dsw,ws.size()));CUDA(cudaMalloc(&dsx,xs.size()));
            CUDA(cudaMalloc(&dy,batch*rows*2));constexpr size_t workspaceBytes=4ULL<<20;CUDA(cudaMalloc(&workspace,workspaceBytes));
            CUDA(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dx,x.data(),x.size(),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(dsw,ws.data(),ws.size(),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(dsx,xs.data(),xs.size(),cudaMemcpyHostToDevice));
            BLAS(Garnet::RunModelOptNVFP4Matmul(handle,dw,dsw,dx,dsx,dy,batch,rows,k,.5f,workspace,workspaceBytes,stream));
            CUDA(cudaStreamSynchronize(stream));std::vector<__nv_bfloat16> output(batch*rows);
            CUDA(cudaMemcpy(output.data(),dy,output.size()*2,cudaMemcpyDeviceToHost));
            for(int b=0;b<batch;++b)for(int r=0;r<rows;++r){
                float expected=0;for(int block=0;block<blocks;++block)expected+=16*.5f*values[(r+block)%3]*values[(b+block+1)%3];
                expected=__bfloat162float(__float2bfloat16(expected));
                if(__bfloat162float(output[b*rows+r])!=expected)throw std::runtime_error("Native FP4 matmul differs from independent tiled-scale oracle");
            }
            printf("CUBLASLT_NATIVE_FP4_PASS: batch=%d, exact outputs=%d; no performance claim\n",batch,batch*rows);fflush(stdout);
            // Exercise the complete BF16 -> dynamic NVFP4 -> native FP4 path.
            std::vector<__nv_bfloat16> hidden(batch*k);
            const float activations[]={0,.5f,1.f,1.5f};
            for(int b=0;b<batch;++b)for(int c=0;c<k;++c)hidden[b*k+c]=__float2bfloat16(activations[(b+c)%4]);
            std::vector<unsigned char> rowScales(rows*blocks);
            for(int r=0;r<rows;++r)for(int block=0;block<blocks;++block)rowScales[r*blocks+block]=codes[(r+block)%3];
            std::fill(w.begin(),w.end(),0x42);
            void *deviceHidden=nullptr,*deviceRowScales=nullptr;
            CUDA(cudaMalloc(&deviceHidden,hidden.size()*2));CUDA(cudaMalloc(&deviceRowScales,rowScales.size()));
            CUDA(cudaMemcpy(deviceHidden,hidden.data(),hidden.size()*2,cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(deviceRowScales,rowScales.data(),rowScales.size(),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
            CUDA(Garnet::RepackModelOptNVFP4Scales((unsigned char*)deviceRowScales,(unsigned char*)dsw,rows,k,stream));
            constexpr float inputGlobal=1.f/448.f;
            CUDA(Garnet::QuantizeModelOptNVFP4Activation(deviceHidden,(unsigned char*)dx,(unsigned char*)dsx,inputGlobal,batch,k,stream));
            BLAS(Garnet::RunModelOptNVFP4Matmul(handle,dw,dsw,dx,dsx,dy,batch,rows,k,.5f*inputGlobal,workspace,workspaceBytes,stream));
            CUDA(cudaStreamSynchronize(stream));CUDA(cudaMemcpy(output.data(),dy,output.size()*2,cudaMemcpyDeviceToHost));
            for(int b=0;b<batch;++b)for(int r=0;r<rows;++r){
                float expected=0;for(int col=0;col<k;++col)expected+=__bfloat162float(hidden[b*k+col])*(col%2?2.f:1.f)*values[(r+col/16)%3]*.5f;
                expected=__bfloat162float(__float2bfloat16(expected));
                if(__bfloat162float(output[b*rows+r])!=expected)throw std::runtime_error("Dynamic NVFP4 pipeline differs from independent CPU oracle");
            }
            CUDA(cudaMemsetAsync(deviceHidden,0,hidden.size()*2,stream));
            CUDA(Garnet::QuantizeModelOptNVFP4Activation(deviceHidden,(unsigned char*)dx,(unsigned char*)dsx,inputGlobal,batch,k,stream));
            BLAS(Garnet::RunModelOptNVFP4Matmul(handle,dw,dsw,dx,dsx,dy,batch,rows,k,.5f*inputGlobal,workspace,workspaceBytes,stream));
            CUDA(cudaStreamSynchronize(stream));CUDA(cudaMemcpy(output.data(),dy,output.size()*2,cudaMemcpyDeviceToHost));
            for(auto value:output)if(__bfloat162float(value)!=0)throw std::runtime_error("All-zero activation block did not remain zero");
            CUDA(cudaMemcpy(xs.data(),dsx,xs.size(),cudaMemcpyDeviceToHost));
            for(auto value:xs)if(value!=0)throw std::runtime_error("Zero or padded activation scale was not zero-filled");
            CUDA(cudaFree(deviceHidden));CUDA(cudaFree(deviceRowScales));
            CUDA(cudaFree(dw));CUDA(cudaFree(dx));CUDA(cudaFree(dsw));CUDA(cudaFree(dsx));CUDA(cudaFree(dy));CUDA(cudaFree(workspace));
            printf("DYNAMIC_NVFP4_PIPELINE_PASS: batch=%d, exact outputs=%d; no performance claim\n",batch,batch*rows);fflush(stdout);
        }
        CUDA(cudaStreamDestroy(stream));BLAS(cublasLtDestroy(handle));return 0;
    }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
}
