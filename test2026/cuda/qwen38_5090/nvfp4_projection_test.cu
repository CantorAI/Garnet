// SPDX-License-Identifier: Apache-2.0
#include "modelopt_nvfp4.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
void Check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
// Independent arithmetic E4M3 decoder (no CUDA FP8 conversion in the oracle).
float Scale(unsigned char x) {
    const int exponent=(x>>3)&15, mantissa=x&7;
    float value=exponent ? std::ldexp(1.f+mantissa/8.f,exponent-7) : std::ldexp(float(mantissa),-9);
    return x&128 ? -value : value;
}
int main() {
    try {
        constexpr int batch=3, rows=17, columns=528;
        std::vector<__nv_bfloat16> input(batch*columns), output(batch*rows);
        std::vector<unsigned char> weights(rows*columns/2), scales(rows*columns/16);
        for(size_t i=0;i<input.size();++i) input[i]=__float2bfloat16(float(int(i%13)-6)/8);
        for(size_t i=0;i<weights.size();++i) weights[i]=static_cast<unsigned char>((i*73+19)%256);
        const unsigned char codes[]={0,1,7,8,24,48,56,64,80,96,112,126};
        for(size_t i=0;i<scales.size();++i) scales[i]=codes[i%12];
        void *x=nullptr,*w=nullptr,*s=nullptr,*y=nullptr;
        Check(cudaMalloc(&x,input.size()*2)); Check(cudaMalloc(&w,weights.size()));
        Check(cudaMalloc(&s,scales.size())); Check(cudaMalloc(&y,output.size()*2));
        Check(cudaMemcpy(x,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        Check(cudaMemcpy(w,weights.data(),weights.size(),cudaMemcpyHostToDevice));
        Check(cudaMemcpy(s,scales.data(),scales.size(),cudaMemcpyHostToDevice));
        constexpr float scale2=.125f;
        Check(Garnet::RunModelOptNVFP4Projection(x,(unsigned char*)w,(unsigned char*)s,scale2,y,batch,rows,columns,nullptr));
        Check(cudaMemcpy(output.data(),y,output.size()*2,cudaMemcpyDeviceToHost));
        const float values[]={0,.5,1,1.5,2,3,4,6};
        for(int b=0;b<batch;++b)for(int r=0;r<rows;++r) {
            double expected=0;
            for(int k=0;k<columns;++k) {
                const int code=(weights[r*columns/2+k/2]>>(4*(k%2)))&15;
                const double weight=values[code&7]*(code&8?-1:1)*Scale(scales[r*columns/16+k/16])*scale2;
                expected+=__bfloat162float(input[b*columns+k])*weight;
            }
            float rounded=__bfloat162float(__float2bfloat16(float(expected)));
            float actual=__bfloat162float(output[b*rows+r]);
            if(actual!=rounded) throw std::runtime_error("NVFP4 projection disagrees with independent CPU oracle");
        }
        if(Garnet::RunModelOptNVFP4Projection(x,(unsigned char*)w,(unsigned char*)s,scale2,y,1,rows,17,nullptr)!=cudaErrorInvalidValue)
            throw std::runtime_error("Unaligned input accepted");
        Check(cudaFree(x));Check(cudaFree(w));Check(cudaFree(s));Check(cudaFree(y));
        puts("NVFP4_PROJECTION_CORRECTNESS_PASS: 51 outputs, all nibble codes, varied block scales; no performance claim");
        return 0;
    } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what());return 1; }
}
