// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

static void same(const std::vector<float>& a,const std::vector<float>& b) {
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float)))
        throw std::runtime_error("Joined global MLP changed complete output bits");
}
int main() {
    try {
        for(auto shape:{std::make_tuple(24u,1024u,3072u),std::make_tuple(44u,1024u,3072u),
                       std::make_tuple(26u,896u,4864u),std::make_tuple(32u,1024u,3584u)}) {
            const auto [rows,inner,columns]=shape;
            Garnet::MetalContext gpu;
            std::vector<uint16_t> gateBits(inner*columns),upBits(inner*columns),larger(inner*columns);
            for(size_t i=0;i<gateBits.size();++i) {
                float g=float(int(i%17)-8)/32,u=float(int((i*7+3)%17)-8)/32;
                uint32_t gb,ub;std::memcpy(&gb,&g,4);std::memcpy(&ub,&u,4);
                gateBits[i]=uint16_t(gb>>16);upBits[i]=uint16_t(ub>>16);
                g*=2;std::memcpy(&gb,&g,4);larger[i]=uint16_t(gb>>16);
            }
            auto gate=gpu.UploadBF16(gateBits.data(),gateBits.size()*2,columns,inner);
            auto up=gpu.UploadBF16(upBits.data(),upBits.size()*2,columns,inner);
            auto changedScale=gpu.UploadBF16(larger.data(),larger.size()*2,columns,inner);
            if(!gate.GlobalHalfReady()||!up.GlobalHalfReady()||!changedScale.GlobalHalfReady())
                throw std::runtime_error("Synthetic exact common-scale fixture rejected");
            auto g=gpu.Allocate(rows,columns),u=gpu.Allocate(rows,columns),o=gpu.Allocate(rows,columns);
            std::vector<float> values(rows*inner);
            for(int request=0;request<3;++request) {
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.013f)*(request==1?-.5f:1.f);
                auto a=gpu.Upload(values,rows,inner);
                // After the first joined execution these separate MPS products
                // read retained compact half-weight subviews, including offsets.
                gpu.Begin();gpu.EncodeBF16MatVec(a,gate,g);gpu.EncodeBF16MatVec(a,up,u);
                gpu.EncodeSiluMul(g,u,o);gpu.SubmitAndWait();
                auto expectedG=gpu.Download(g),expectedU=gpu.Download(u),expectedO=gpu.Download(o);
                gpu.Begin();if(gpu.LastJoinedGlobalMlpBlocks())throw std::runtime_error("Begin count retained");
                if(!gpu.EncodeJoinedGlobalMlp(a,gate,up,g,u,o))throw std::runtime_error("Admitted shape fell back");
                bool pending=false;try{gpu.Download(o);}catch(const std::logic_error&){pending=true;}
                if(!pending)throw std::runtime_error("Pending joined activation was readable");
                gpu.SubmitAndWait();
                if(gpu.LastJoinedGlobalMlpBlocks()!=1)throw std::runtime_error("Actual joined count missing");
                same(expectedG,gpu.Download(g));same(expectedU,gpu.Download(u));same(expectedO,gpu.Download(o));
                gpu.Begin();
                if(gpu.EncodeJoinedGlobalMlp(a,gate,up,g,g,o))throw std::runtime_error("Aliased outputs accepted");
                if(gpu.EncodeJoinedGlobalMlp(a,gate,changedScale,g,u,o))throw std::runtime_error("Different common scales accepted");
                gpu.Abort();same(expectedG,gpu.Download(g));same(expectedU,gpu.Download(u));same(expectedO,gpu.Download(o));
                gpu.Begin();gpu.EncodeJoinedGlobalMlp(a,gate,up,g,u,o);gpu.Abort();
                if(gpu.LastJoinedGlobalMlpBlocks())throw std::runtime_error("Abort count retained");
                bool aborted=false;try{gpu.Download(o);}catch(const std::logic_error&){aborted=true;}
                if(!aborted)throw std::runtime_error("Aborted joined activation was readable");
                gpu.Begin();gpu.EncodeJoinedGlobalMlp(a,gate,up,g,u,o);gpu.SubmitAndWait();same(expectedO,gpu.Download(o));
                auto uninitialized=gpu.Allocate(rows,inner);gpu.Begin();bool rejected=false;
                try{gpu.EncodeJoinedGlobalMlp(uninitialized,gate,up,g,u,o);}catch(const std::logic_error&){rejected=true;}
                gpu.Abort();if(!rejected)throw std::runtime_error("Uninitialized joined input accepted");
            }
            {
                auto a=gpu.Upload(values,rows,inner),wrong=gpu.Allocate(rows,columns+1);
                gpu.Begin();bool badShape=false;
                try{gpu.EncodeJoinedGlobalMlp(a,gate,up,g,u,wrong);}catch(const std::invalid_argument&){badShape=true;}
                gpu.Abort();if(!badShape)throw std::runtime_error("Wrong activation shape accepted");
                Garnet::MetalContext foreignGpu;auto foreign=foreignGpu.Allocate(rows,columns);
                foreignGpu.Begin();foreignGpu.EncodeAdd(g,g,foreign);gpu.Begin();bool foreignPending=false;
                try{gpu.EncodeJoinedGlobalMlp(a,gate,up,foreign,u,o);}catch(const std::logic_error&){foreignPending=true;}
                gpu.Abort();foreignGpu.Abort();
                if(!foreignPending)throw std::runtime_error("Foreign pending output accepted");
            }
            // Decode must retain the unchanged learned BF16 representation.
            std::vector<float> one(values.begin(),values.begin()+inner);
            auto a=gpu.Upload(one,1,inner),decoded=gpu.Allocate(1,columns);
            gpu.Begin();if(gpu.EncodeJoinedGlobalMlp(a,gate,up,g,u,o))throw std::runtime_error("Decode took prefill route");
            gpu.EncodeBF16MatVec(a,gate,decoded);gpu.SubmitAndWait();
            auto result=gpu.Download(decoded);
            for(size_t c=0;c<columns;++c) {
                double reference=0,sumabs=0;
                for(size_t k=0;k<inner;++k) {uint32_t b=uint32_t(gateBits[c*inner+k])<<16;float w;std::memcpy(&w,&b,4);double v=double(one[k])*w;reference+=v;sumabs+=std::abs(v);}
                if(std::abs(result[c]-reference)>1e-6+2e-5*sumabs)throw std::runtime_error("Compact prefill changed raw BF16 decode");
            }
        }
        std::cout<<"Joined global MLP all outputs, compact reuse, fallback, abort and decode PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
