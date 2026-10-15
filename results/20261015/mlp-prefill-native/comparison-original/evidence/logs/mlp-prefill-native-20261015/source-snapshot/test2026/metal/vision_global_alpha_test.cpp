// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        Garnet::MetalContext gpu;
        const size_t k=7,n=5;
        std::vector<uint16_t> bits(k*n);std::vector<float> learned(k*n);
        for(size_t i=0;i<bits.size();++i) {
            learned[i]=float(int(i%17)-8)/32;uint32_t raw;std::memcpy(&raw,&learned[i],4);bits[i]=uint16_t(raw>>16);
        }
        auto vision=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,false,true);
        auto body=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        auto weight=gpu.Upload(learned,n,k);
        if(!vision.GlobalHalfReady())throw std::runtime_error("Exact vision fixture not admitted");
        for(size_t rows:{1,2,54,108,216,432})for(float scale:{1.f,-1.f,.5f,2.f}) {
            std::vector<float> values(rows*k);for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.013f)*scale;
            auto a=gpu.Upload(values,rows,k),reference=gpu.Allocate(rows,n),output=gpu.Allocate(rows,n);
            gpu.Begin();gpu.EncodeMatMul(a,weight,reference,true);gpu.SubmitAndWait();auto expected=gpu.Download(reference);
            gpu.Begin();gpu.EncodeBF16MatVec(a,vision,output);gpu.SubmitAndWait();auto actual=gpu.Download(output);
            if(gpu.LastVisionGlobalAlphaDispatches()!=1)throw std::runtime_error("Actual vision dispatch count missing");
            for(size_t r=0;r<rows;++r)for(size_t c=0;c<n;++c) {
                double exact=0;for(size_t i=0;i<k;++i)exact+=double(values[r*k+i])*learned[c*k+i];
                double bound=5e-6+5e-5*std::abs(exact);
                if(std::abs(double(actual[r*n+c])-exact)>bound||std::abs(double(expected[r*n+c])-exact)>bound||
                    std::abs(double(actual[r*n+c])-expected[r*n+c])>bound)
                    throw std::runtime_error("Vision common alpha exceeds unchanged FP64/FP32 envelope");
            }
            if(rows>128) {
                gpu.Begin();bool rejected=false;
                try{gpu.EncodeBF16MatVec(a,body,output);}catch(const std::invalid_argument&){rejected=true;}
                gpu.Abort();if(!rejected)throw std::runtime_error("Vision row limit leaked into decoder weights");
            }
            gpu.Begin();if(gpu.LastVisionGlobalAlphaDispatches())throw std::runtime_error("Vision Begin count retained");
            gpu.EncodeBF16MatVec(a,vision,output);gpu.Abort();
            if(gpu.LastVisionGlobalAlphaDispatches())throw std::runtime_error("Vision Abort count retained");
            bool unreadable=false;try{gpu.Download(output);}catch(const std::logic_error&){unreadable=true;}
            if(!unreadable)throw std::runtime_error("Aborted vision output remained readable");
            gpu.Begin();gpu.EncodeBF16MatVec(a,vision,output);gpu.SubmitAndWait();
        }
        const uint16_t mixed[]={0x3f80,0x3f00,0x2b80,0x2b00};
        auto rejectedScale=gpu.UploadBF16(mixed,sizeof(mixed),2,2,false,true);
        if(rejectedScale.GlobalHalfReady()||!rejectedScale.ExactHalfReady())
            throw std::runtime_error("Inexact vision common scale lost safe fallback representation");
        auto oversized=gpu.Allocate(32769,k),out=gpu.Allocate(32769,n);gpu.Begin();bool rejected=false;
        try{gpu.EncodeBF16MatVec(oversized,vision,out);}catch(const std::invalid_argument&){rejected=true;}
        gpu.Abort();if(!rejected)throw std::runtime_error("Out-of-profile vision rows accepted");
        std::cout<<"Vision global alpha tiny/full rows, FP64, decoder limits, fallback and abort PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
