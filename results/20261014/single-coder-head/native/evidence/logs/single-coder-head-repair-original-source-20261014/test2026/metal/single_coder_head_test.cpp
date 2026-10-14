// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        constexpr size_t k=896,n=151936;
        Garnet::MetalContext gpu;
        std::vector<uint16_t> bits(k*n);
        for(size_t i=0;i<bits.size();++i) {
            float value=float(int(i%17)-8)/128;uint32_t raw;
            std::memcpy(&raw,&value,4);bits[i]=uint16_t(raw>>16);
        }
        auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        std::vector<float> base(k);for(size_t i=0;i<k;++i)base[i]=std::sin(float(i)*.013f)*.3f;
        // Seventeen exact row patterns repeat across the full synthetic matrix.
        // Compute each independent FP64 reference, then check every output.
        std::array<double,17> reference{},sumabs{};
        for(size_t phase=0;phase<17;++phase)for(size_t i=0;i<k;++i) {
            double term=double(base[i])*(double(int((phase+i)%17)-8)/128);
            reference[phase]+=term;sumabs[phase]+=std::abs(term);
        }
        const double unit=std::ldexp(1.0,-24),rounds=std::ceil(double(k)/32)+32;
        const double gamma=rounds*unit/(1-rounds*unit);
        auto first=gpu.Allocate(1,n),both=gpu.Allocate(2,n);
        auto firstArg=gpu.Allocate(1,2),bothArg=gpu.Allocate(2,2);
        for(float scale:{1.f,-1.f,.5f,2.f}) {
            std::vector<float> single(k),two(2*k);
            for(size_t i=0;i<k;++i){single[i]=base[i]*scale;two[i]=single[i];two[k+i]=-single[i];}
            auto a=gpu.Upload(single,1,k),b=gpu.Upload(two,2,k);
            gpu.Begin();gpu.EncodeBF16MatVec(b,weight,both);gpu.EncodeArgmax(both,bothArg);gpu.SubmitAndWait();
            if(gpu.LastSingleCoderHeadDispatches())throw std::runtime_error("Batch two took single-row head route");
            auto control=gpu.Download(both),controlArg=gpu.Download(bothArg);
            gpu.Begin();gpu.EncodeBF16MatVec(a,weight,first);gpu.EncodeArgmax(first,firstArg);gpu.SubmitAndWait();
            if(gpu.LastSingleCoderHeadDispatches()!=1)throw std::runtime_error("Single-row actual head count missing");
            auto candidate=gpu.Download(first),candidateArg=gpu.Download(firstArg);
            if(std::memcmp(candidate.data(),control.data(),n*sizeof(float))||candidateArg[0]!=controlArg[0]||candidateArg[1]!=controlArg[1])
                throw std::runtime_error("Single coder head changed complete output bits or argmax");
            for(size_t c=0;c<n;++c) {
                size_t phase=(c*k)%17;double expected=reference[phase]*scale;
                double bound=gamma*sumabs[phase]*std::abs(scale)+1e-7;
                if(std::abs(double(candidate[c])-expected)>bound||std::abs(double(control[n+c])+expected)>bound)
                    throw std::runtime_error("Coder vocabulary projection exceeds unchanged FP64 gamma envelope");
            }
            gpu.Begin();if(gpu.LastSingleCoderHeadDispatches())throw std::runtime_error("Begin did not reset head count");
            gpu.EncodeBF16MatVec(a,weight,first);gpu.Abort();
            if(gpu.LastSingleCoderHeadDispatches())throw std::runtime_error("Abort did not reset head count");
            bool rejected=false;try{gpu.Download(first);}catch(const std::logic_error&){rejected=true;}
            if(!rejected)throw std::runtime_error("Aborted head output remained readable");
            gpu.Begin();gpu.EncodeBF16MatVec(a,weight,first);gpu.SubmitAndWait();
            auto recovered=gpu.Download(first);if(std::memcmp(recovered.data(),control.data(),n*sizeof(float)))
                throw std::runtime_error("Head did not recover after abort");
        }
        // A neighbouring body shape continues to use the original route.
        const uint16_t small[]={0x3f80,0x4000,0x4040,0x4080};
        auto w=gpu.UploadBF16(small,sizeof(small),2,2),a=gpu.Upload({1,2},1,2),o=gpu.Allocate(1,2);
        gpu.Begin();gpu.EncodeBF16MatVec(a,w,o);gpu.SubmitAndWait();
        if(gpu.LastSingleCoderHeadDispatches()||gpu.Download(o)!=std::vector<float>({5,11}))
            throw std::runtime_error("Non-vocabulary projection changed");
        std::cout<<"Single coder head full-bit/F64/argmax, batch-two fallback, reset and abort PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
