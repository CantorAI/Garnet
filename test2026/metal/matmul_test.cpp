// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <limits>
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
        for(size_t batch:{1,2,3,4,5,16,32})for(size_t width:{7,128,1024,1031}) {
            const size_t vocabulary=37;
            std::vector<float> values(batch*width),weights(vocabulary*width);
            std::vector<uint16_t> bits(weights.size());
            for(size_t i=0;i<values.size();++i)values[i]=float(int(i%13)-6)*0.125f;
            for(size_t i=0;i<weights.size();++i){weights[i]=float(int(i%11)-5)*0.25f;
                uint32_t raw;std::memcpy(&raw,&weights[i],4);bits[i]=uint16_t(raw>>16);}
            auto a=gpu.Upload(values,batch,width),out=gpu.Allocate(batch,vocabulary);
            auto w=gpu.UploadBF16(bits.data(),bits.size()*2,vocabulary,width);
            gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.SubmitAndWait();auto actual=gpu.Download(out);
            for(size_t row=0;row<batch;++row)for(size_t column=0;column<vocabulary;++column) {
                double expected=0;for(size_t k=0;k<width;++k)expected+=double(values[row*width+k])*weights[column*width+k];
                if(!std::isfinite(actual[row*vocabulary+column])||std::abs(actual[row*vocabulary+column]-expected)>1e-5)
                    throw std::runtime_error("BF16 vocabulary projection differs from FP64 reference");
            }
        }
        uint16_t nonfinite=0x7f80;bool bf16Rejected=false;
        try {gpu.UploadBF16(&nonfinite,2,1,1);}catch(const std::invalid_argument&){bf16Rejected=true;}
        if(!bf16Rejected)throw std::runtime_error("Nonfinite BF16 weight accepted");
        for(size_t width:{7,1031})for(float offset:{0.0f,1.0f}) {
            std::vector<float> values(2*width),weights(width);
            for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*0.37f);
            for(size_t i=0;i<weights.size();++i)weights[i]=std::cos(float(i)*0.11f);
            auto a=gpu.Upload(values,2,width),w=gpu.Upload(weights,1,width),out=gpu.Allocate(2,width);
            gpu.Begin();gpu.EncodeRMSNorm(a,w,out,1e-6f,offset);gpu.SubmitAndWait();auto actual=gpu.Download(out);
            for(size_t row=0;row<2;++row){double sum=0;for(size_t k=0;k<width;++k)sum+=double(values[row*width+k])*values[row*width+k];
                for(size_t k=0;k<width;++k){double expected=values[row*width+k]/std::sqrt(sum/width+1e-6)*(weights[k]+offset);
                    if(!std::isfinite(actual[row*width+k])||std::abs(actual[row*width+k]-expected)>2e-6)
                        throw std::runtime_error("Parallel RMS differs from independent FP64 reference");}}
        }
        for(size_t width:{7,1031}) {
            std::vector<float> values(2*width,-100);values[3]=8;values[width-1]=8;values[width+5]=9;
            auto a=gpu.Upload(values,2,width),out=gpu.Allocate(2,2);
            gpu.Begin();gpu.EncodeArgmax(a,out);gpu.SubmitAndWait();
            if(gpu.Download(out)!=std::vector<float>{3,0,5,0})throw std::runtime_error("GPU argmax tie/batch mismatch");
            values[0]=std::numeric_limits<float>::quiet_NaN();values[width]=std::numeric_limits<float>::infinity();gpu.Update(a,values);
            gpu.Begin();gpu.EncodeArgmax(a,out);gpu.SubmitAndWait();auto actual=gpu.Download(out);
            if(actual[1]!=1||actual[3]!=1)throw std::runtime_error("GPU argmax did not report nonfinite logits");
        }
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
        gpu.Begin();gpu.EncodeAdd(input,weight,output);gpu.Abort();
        bool customAbortRejected=false;
        try {gpu.Download(output);}catch(const std::logic_error&){customAbortRejected=true;}
        if(!customAbortRejected)throw std::runtime_error("Aborted custom compute output was readable");
        gpu.Begin();gpu.EncodeAdd(input,weight,output);gpu.SubmitAndWait();
        auto inputValues=gpu.Download(input),weightValues=gpu.Download(weight),resumed=gpu.Download(output);
        for(size_t index=0;index<resumed.size();++index)
            if(resumed[index]!=inputValues[index]+weightValues[index])throw std::runtime_error("Custom compute did not recover after abort");
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
