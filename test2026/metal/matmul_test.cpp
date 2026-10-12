// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
    try {
        Garnet::MetalContext gpu;
        for(bool transpose: {false,true}) {
            const size_t m=3,k=7,n=5;
            std::vector<float> a(m*k),b(k*n);
            for(size_t i=0;i<a.size();++i) a[i]=(int(i%9)-4)*0.125f;
            for(size_t i=0;i<b.size();++i) b[i]=(int(i%11)-5)*0.25f;
            auto result=gpu.MatMul(a,b,m,k,n,transpose);
            for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j) {
                double expected=0;
                for(size_t p=0;p<k;++p)expected+=double(a[i*k+p])*b[transpose?j*k+p:p*n+j];
                if(!std::isfinite(result[i*n+j]) || std::abs(result[i*n+j]-expected)>1e-5)
                    throw std::runtime_error("MPS result disagrees with independent CPU reference");
            }
        }
        bool rejected=false;
        try {gpu.MatMul({}, {}, 0, 7, 5);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Invalid dimensions were accepted");
        std::cout<<"Metal/MPS matmul reference PASS device="<<gpu.DeviceName()<<"\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<e.what()<<"\n";return 1;
    }
}
