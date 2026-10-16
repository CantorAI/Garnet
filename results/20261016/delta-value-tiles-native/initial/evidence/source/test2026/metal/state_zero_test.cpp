// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
    try {
        Garnet::MetalContext gpu;
        size_t checks=0;
        for(size_t rows:{1,2,3})for(size_t columns:{1,3,17,65536}) {
            auto matrix=gpu.Allocate(rows,columns);
            for(size_t repeat=0;repeat<3;++repeat) {
                std::vector<float> values(rows*columns,float(repeat+1));
                gpu.Update(matrix,values);gpu.Zero(matrix);
                auto result=gpu.Download(matrix);
                for(float value:result) {
                    uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
                    if(bits!=0)throw std::runtime_error("Reset must produce positive zero");
                }
                // Consume cleared buffers in real GPU work, not only CPU readback.
                auto output=gpu.Allocate(rows,columns);
                gpu.Begin();gpu.EncodeAdd(matrix,matrix,output);gpu.SubmitAndWait();
                for(float value:gpu.Download(output))
                    if(value!=0)throw std::runtime_error("GPU consumed uncleared state");
                ++checks;
            }
            gpu.Begin();bool rejected=false;
            try{gpu.Zero(matrix);}catch(const std::logic_error&){rejected=true;}
            gpu.Abort();if(!rejected)throw std::runtime_error("Open command batch accepted");
        }
        bool rejected=false;
        try{gpu.Zero(Garnet::MetalMatrix());}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Empty matrix accepted");
        std::cout<<"state_zero checks "<<checks<<" PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
