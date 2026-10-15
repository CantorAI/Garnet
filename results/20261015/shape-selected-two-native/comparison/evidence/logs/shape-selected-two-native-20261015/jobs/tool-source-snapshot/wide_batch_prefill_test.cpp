// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using Garnet::MetalContext;
int main() {
    try {
        MetalContext gpu;size_t checks=0;
        // Cover the exact-half MPS route and the raw BF16 fallback, including
        // the small projection widths used by Qwen3.5 linear attention.
        for(bool fallback:{false,true})for(size_t inner:{7,128,896,1024,3072})for(size_t columns:{8,37,128}) {
            std::vector<float> a(160*inner),w(columns*inner);
            std::vector<uint16_t> bits(w.size());
            for(size_t i=0;i<a.size();++i)a[i]=std::sin(float(i)*.037f)*.125f;
            for(size_t i=0;i<w.size();++i) {
                w[i]=float(int((i*7)%17)-8)/64;
                if(fallback&&i==1)w[i]=std::ldexp(1.0f,-126);
                uint32_t word;std::memcpy(&word,&w[i],4);bits[i]=uint16_t(word>>16);
            }
            auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner);
            if(weight.GlobalHalfReady()==fallback)throw std::runtime_error("Wrong tested projection route");
            auto input=gpu.Upload(a,160,inner),out=gpu.Allocate(160,columns);
            gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);
            bool pending=false;try{gpu.Download(out);}catch(const std::logic_error&){pending=true;}
            if(!pending)throw std::runtime_error("Pending projection was readable");
            gpu.SubmitAndWait();auto actual=gpu.Download(out);
            if(gpu.LastWideBatchPrefillDispatches()!=1)throw std::runtime_error("Wide projection was not counted");
            for(size_t row=0;row<160;++row)for(size_t col=0;col<columns;++col) {
                double expected=0;for(size_t k=0;k<inner;++k)expected+=double(a[row*inner+k])*w[col*inner+k];
                if(!std::isfinite(actual[row*columns+col])||std::abs(actual[row*columns+col]-expected)>5e-6+5e-5*std::abs(expected))
                    throw std::runtime_error("160-row projection exceeded original FP64 bound");
            }
            gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);gpu.Abort();
            if(gpu.LastWideBatchPrefillDispatches()!=0)throw std::runtime_error("Aborted projection count survived");
            bool aborted=false;try{gpu.Download(out);}catch(const std::logic_error&){aborted=true;}
            if(!aborted)throw std::runtime_error("Aborted projection was readable");
            gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);gpu.SubmitAndWait();
            for(size_t rows:{129,159,161}) {
                auto bad=gpu.Upload(std::vector<float>(rows*inner,1),rows,inner),dest=gpu.Allocate(rows,columns);
                gpu.Begin();bool rejected=false;try{gpu.EncodeBF16MatVec(bad,weight,dest);}catch(const std::invalid_argument&){rejected=true;}
                if(!rejected||gpu.LastWideBatchPrefillDispatches()!=0)throw std::runtime_error("Unvalidated row count admitted");
                gpu.Abort();
            }
            auto empty=gpu.Allocate(160,inner);gpu.Begin();bool rejected=false;
            try{gpu.EncodeBF16MatVec(empty,weight,out);}catch(const std::logic_error&){rejected=true;}
            if(!rejected||gpu.LastWideBatchPrefillDispatches()!=0)throw std::runtime_error("Uninitialized input admitted");
            gpu.Abort();++checks;
        }
        std::cout<<"Wide projection original-bound, readiness, abort, recovery and rejection checks="<<checks<<" PASS\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
