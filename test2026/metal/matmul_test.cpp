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
        auto input=gpu.Upload({1,2,3,4},2,2);
        auto weight=gpu.Upload({2,0,0,3},2,2);
        auto hidden=gpu.Allocate(2,2),output=gpu.Allocate(2,2);
        for(int request=0;request<2;++request) {
            gpu.Begin();
            gpu.EncodeMatMul(input,weight,hidden);
            gpu.EncodeMatMul(hidden,weight,output);
            bool pendingRejected=false;
            try {gpu.Download(output);}catch(const std::logic_error&){pendingRejected=true;}
            if(!pendingRejected)throw std::runtime_error("Pending GPU output was readable");
            gpu.SubmitAndWait();
            if(gpu.Download(output)!=std::vector<float>({4,18,12,36}))
                throw std::runtime_error("Persistent chained matmul reference mismatch");
        }
        gpu.Begin();gpu.EncodeMatMul(input,weight,output);gpu.Abort();
        bool abortedRejected=false;
        try {gpu.Download(output);}catch(const std::logic_error&){abortedRejected=true;}
        if(!abortedRejected)throw std::runtime_error("Aborted GPU output was readable");
        auto updated=gpu.Allocate(3,5);
        std::vector<float> padded(15);
        for(size_t i=0;i<padded.size();++i)padded[i]=float(i)-7;
        gpu.Update(updated,padded);
        if(gpu.Download(updated)!=padded)throw std::runtime_error("Padded row update mismatch");
        gpu.Update(updated,std::vector<float>(15,0.25f));
        if(gpu.Download(updated)!=std::vector<float>(15,0.25f))throw std::runtime_error("Repeated update retained stale values");
        bool wrongSize=false;
        try {gpu.Update(updated,{1});}catch(const std::invalid_argument&){wrongSize=true;}
        if(!wrongSize)throw std::runtime_error("Invalid update length accepted");
        gpu.Begin();gpu.EncodeMatMul(input,weight,hidden);
        bool batchUpdate=false;
        try {gpu.Update(input,{4,3,2,1});}catch(const std::logic_error&){batchUpdate=true;}
        if(!batchUpdate)throw std::runtime_error("Update during command batch accepted");
        gpu.SubmitAndWait();
        auto scale=gpu.Upload({0.25f,-0.5f},1,2);
        auto normalized=gpu.Allocate(2,2),activation=gpu.Allocate(2,2);
        gpu.Begin();
        gpu.EncodeRMSNorm(input,scale,normalized,1e-6f);
        gpu.EncodeSiluMul(normalized,input,activation);
        gpu.SubmitAndWait();
        auto actual=gpu.Download(activation);
        for(size_t row=0;row<2;++row)for(size_t col=0;col<2;++col) {
            double x=double(row*2+col+1);
            double first=row*2+1,second=row*2+2;
            double norm=x/std::sqrt((first*first+second*second)/2+1e-6)*(col==0?1.25:0.5);
            double wanted=norm/(1+std::exp(-norm))*x;
            if(std::abs(actual[row*2+col]-wanted)>2e-5)
                throw std::runtime_error("Qwen RMS/SiLU GPU reference mismatch");
        }
        std::cout<<"Metal/MPS matmul reference PASS device="<<gpu.DeviceName()<<"\n";
        return 0;
    }catch(const std::exception& e) {
        std::cerr<<e.what()<<"\n";return 1;
    }
}
