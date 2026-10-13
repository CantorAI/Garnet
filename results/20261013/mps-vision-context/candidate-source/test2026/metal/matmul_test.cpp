// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "exact_half_codec.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <limits>
int main() {
    try {
        Garnet::MetalContext gpu;
#ifdef GARNET_METAL_MPS_VISION_SCORES
        size_t visionChecks=0;
        for(size_t tokens:{4,24,216})for(size_t batch:{1,2})for(size_t heads:{2,12})for(size_t dim:{8,64}) {
            const size_t rows=batch*tokens,hidden=heads*dim,width=3*hidden;
            std::vector<float> values(rows*width);
            auto input=gpu.Allocate(rows,width),output=gpu.Allocate(rows,heads*tokens);
            for(size_t repeat=0;repeat<2;++repeat) {
                for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i+repeat*7919)*0.017f)*0.3f;
                gpu.Update(input,values);
                if(repeat==1){gpu.Begin();gpu.EncodeVision(7,{input,input,input},output,tokens,hidden,heads,4);gpu.Abort();}
                gpu.Begin();gpu.EncodeVision(7,{input,input,input},output,tokens,hidden,heads,4);gpu.SubmitAndWait();
                auto actual=gpu.Download(output);
#ifdef GARNET_METAL_MPS_VISION_CONTEXT
                auto context=gpu.Allocate(rows,hidden);
                if(repeat==1){gpu.Begin();gpu.EncodeVision(8,{output,input,input},context,tokens,hidden,heads,4);gpu.Abort();}
                // Re-encode scores and consume them within one command batch;
                // Q/K and P/V reuse the same scratch only after prior readers.
                gpu.Begin();gpu.EncodeVision(7,{input,input,input},output,tokens,hidden,heads,4);
                gpu.EncodeVision(8,{output,input,input},context,tokens,hidden,heads,4);gpu.SubmitAndWait();
                auto contextValues=gpu.Download(context);
                for(size_t row=0;row<rows;++row)for(size_t head=0;head<heads;++head)for(size_t d=0;d<dim;++d) {
                    double expected=0;const size_t base=row/tokens*tokens;
                    for(size_t key=0;key<tokens;++key)expected+=double(actual[row*heads*tokens+head*tokens+key])*values[(base+key)*width+2*hidden+head*dim+d];
                    const float value=contextValues[row*hidden+head*dim+d];
                    if(!std::isfinite(value)||std::abs(value-expected)>3e-6)
                        throw std::runtime_error("Batched vision value projection disagrees with independent FP64 reference");
                }
#endif
                std::vector<double> scores(tokens);
                for(size_t row=0;row<rows;++row)for(size_t head=0;head<heads;++head) {
                    double maximum=-std::numeric_limits<double>::infinity();
                    const size_t base=row/tokens*tokens;
                    for(size_t key=0;key<tokens;++key) {
                        double score=0;
                        for(size_t d=0;d<dim;++d)score+=double(values[row*width+head*dim+d])*values[(base+key)*width+hidden+head*dim+d];
                        scores[key]=score/std::sqrt(double(dim));maximum=std::max(maximum,scores[key]);
                    }
                    double total=0;for(auto& score:scores){score=std::exp(score-maximum);total+=score;}
                    double actualSum=0;
                    for(size_t key=0;key<tokens;++key) {
                        const float value=actual[row*heads*tokens+head*tokens+key];
                        if(!std::isfinite(value)||value<0||std::abs(value-scores[key]/total)>3e-6)
                            throw std::runtime_error("Batched vision scores disagree with independent FP64 softmax reference");
                        actualSum+=value;
                    }
                    if(std::abs(actualSum-1)>3e-6)throw std::runtime_error("Batched vision softmax is not normalized");
                }
                ++visionChecks;
            }
        }
        std::cout<<"Batched vision attention independent references PASS cases="<<visionChecks<<"\n";
#endif
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
#ifdef GARNET_METAL_EXACT_HALF_PREFILL
        {
            size_t checks=0;
            for(uint32_t word=0;word<65536;++word)for(int32_t scale=-64;scale<=64;++scale) {
                uint16_t actual=0;const bool encoded=Garnet::EncodeExactScaledHalf(uint16_t(word),scale,actual);
                if((word&0x7f80)==0x7f80){if(encoded)throw std::runtime_error("Half codec accepted nonfinite BF16");continue;}
                uint32_t raw=word<<16;float original;std::memcpy(&original,&raw,4);
                const __fp16 half=(__fp16)std::ldexp(original,scale);
                const float recovered=std::ldexp(float(half),-scale);
                const bool expected=std::isfinite(float(half))&&!std::memcmp(&original,&recovered,4);
                uint16_t reference;std::memcpy(&reference,&half,2);
                if(encoded!=expected||(encoded&&actual!=reference))throw std::runtime_error("Integer half codec disagrees with floating bit-roundtrip oracle");
                ++checks;
            }
            std::cout<<"Exact scaled half codec checks="<<checks<<"\n";
            for(uint32_t magnitude=0;magnitude<0x7f80;++magnitude) {
                int32_t actualScale=0;const bool actual=Garnet::ExactHalfRowScale(uint16_t(magnitude),actualScale);
                uint32_t raw=magnitude<<16;float value;std::memcpy(&value,&raw,4);
                int exponent=0;if(value)std::frexp(value,&exponent);
                const int32_t expectedScale=value?16-exponent:0;
                const bool expected=expectedScale>=-64&&expectedScale<=64;
                if(actual!=expected||(actual&&actualScale!=expectedScale))throw std::runtime_error("Integer row scaling disagrees with independent frexp reference");
            }
        }
        {
            std::vector<float> values={0.37f,-0.25f,2,0.25f,0.5f,1,2,3,4,5,-1,0.5f,0.75f,-2,0.125f};
            std::vector<float> weights={0,-0.0f,0.25f,0.5f,0.125f,std::ldexp(1.0f,-25),std::ldexp(3.0f,-25),std::ldexp(-4.0f,-25),0,0,0,0,0,0,0};
            std::vector<uint16_t> bits(weights.size());
            for(size_t i=0;i<bits.size();++i){uint32_t raw;std::memcpy(&raw,&weights[i],4);bits[i]=uint16_t(raw>>16);}
            auto w=gpu.UploadBF16(bits.data(),bits.size()*2,3,5);
            if(!w.ExactHalfReady())throw std::runtime_error("Lossless normal/zero/odd-row half storage unavailable");
            auto a=gpu.Upload(values,3,5),out=gpu.Allocate(3,3);
            gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.SubmitAndWait();const auto actual=gpu.Download(out);
            for(size_t row=0;row<3;++row)for(size_t col=0;col<3;++col) {
                double expected=0;for(size_t inner=0;inner<5;++inner)expected+=double(values[row*5+inner])*weights[col*5+inner];
                if(!std::isfinite(actual[row*3+col])||std::abs(actual[row*3+col]-expected)>1e-6)
                    throw std::runtime_error("Exact half rescaling disagrees with independent FP64 reference");
            }
            const uint16_t exceptional[]={0x3f80,0x0080};
            auto fallback=gpu.UploadBF16(exceptional,4,1,2);
            if(fallback.ExactHalfReady())throw std::runtime_error("Nonexact half encoding was accepted");
            auto fa=gpu.Upload({1,2,3,4,5,6},3,2),fo=gpu.Allocate(3,1);
            gpu.Begin();gpu.EncodeBF16MatVec(fa,fallback,fo);gpu.SubmitAndWait();
            if(gpu.Download(fo)!=std::vector<float>{1,3,5})throw std::runtime_error("Rejected half encoding did not retain raw BF16 fallback");
        }
#endif
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
        for(size_t rows:{1,2})for(size_t heads:{1,3})for(size_t kd:{7,128})for(size_t vd:{5,128})for(bool paused:{false,true}) {
            const size_t qw=heads*kd,vw=heads*vd,sw=heads*kd*vd;
            std::vector<float> q(rows*qw),k(rows*qw),v(rows*vw),s(rows*sw),g(rows*heads,-0.25f),beta(rows*heads,0.5f),flags(rows,1);
            for(size_t i=0;i<q.size();++i){q[i]=float(int(i%11)-5)*0.125f;k[i]=float(int(i%7)-3)*0.25f;}
            for(size_t i=0;i<v.size();++i)v[i]=float(int(i%13)-6)*0.0625f;
            for(size_t i=0;i<s.size();++i)s[i]=float(int(i%17)-8)*0.03125f;
            if(paused)flags[rows-1]=0;
            auto query=gpu.Upload(q,rows,qw),key=gpu.Upload(k,rows,qw),value=gpu.Upload(v,rows,vw),state=gpu.Upload(s,rows,sw);
            auto decay=gpu.Upload(g,rows,heads),update=gpu.Upload(beta,rows,heads),active=gpu.Upload(flags,rows,1);
            auto packed=gpu.Allocate(rows,vw+sw),result=gpu.Allocate(rows,vw),next=gpu.Allocate(rows,sw);
            gpu.Begin();gpu.EncodeGatedDelta(query,key,value,decay,update,state,active,packed,heads,kd,vd);gpu.SubmitAndWait();
            const auto reference=gpu.Download(packed);
            gpu.Begin();gpu.EncodeGatedDeltaSplit(query,key,value,decay,update,state,active,result,next,heads,kd,vd);
            for(const auto& output:{result,next}) {
                bool pending=false;try {gpu.Download(output);}catch(const std::logic_error&){pending=true;}
                if(!pending)throw std::runtime_error("Pending split delta output was readable");
            }
            gpu.SubmitAndWait();const auto actual=gpu.Download(result),actualState=gpu.Download(next);
            for(size_t row=0;row<rows;++row)for(size_t head=0;head<heads;++head) {
                const size_t base=row*(vw+sw)+head*(kd+1)*vd;
                if(std::memcmp(actual.data()+row*vw+head*vd,reference.data()+base,vd*sizeof(float))||
                    std::memcmp(actualState.data()+row*sw+head*kd*vd,reference.data()+base+vd,kd*vd*sizeof(float)))
                    throw std::runtime_error("Split delta differs from packed kernel bits");
            }
            if(rows==2&&heads==3&&kd==7&&vd==5&&paused) {
                gpu.Begin();gpu.EncodeGatedDeltaSplit(query,key,value,decay,update,state,active,result,next,heads,kd,vd);gpu.Abort();
                for(const auto& output:{result,next}) {
                    bool abandoned=false;try {gpu.Download(output);}catch(const std::logic_error&){abandoned=true;}
                    if(!abandoned)throw std::runtime_error("Aborted split delta output was readable");
                }
                gpu.Begin();gpu.EncodeGatedDeltaSplit(query,key,value,decay,update,state,active,result,next,heads,kd,vd);gpu.SubmitAndWait();
                if(gpu.Download(result)!=actual||gpu.Download(next)!=actualState)throw std::runtime_error("Split delta failed recovery after abort");
                gpu.Begin();bool inPlace=false;
                try {gpu.EncodeGatedDeltaSplit(query,key,value,decay,update,state,active,result,state,heads,kd,vd);}
                catch(const std::invalid_argument&){inPlace=true;}gpu.Abort();
                if(!inPlace)throw std::runtime_error("Split delta accepted in-place state update");
            }
        }
        for(size_t rows:{1,2})for(size_t heads:{1,3})for(size_t dim:{5,8,128})for(size_t chunk:{2,4,16})for(int pattern=0;pattern<4;++pattern) {
            const size_t capacity=32,width=heads*capacity*dim;
            std::vector<float> token(rows*chunk*heads*dim),cache(rows*width),slots(rows*chunk),active(rows*chunk,1),expected;
            for(size_t i=0;i<token.size();++i)token[i]=float(i)*0.125f+100;
            for(size_t i=0;i<cache.size();++i){uint32_t bits=i%2?0x80000000u:0x3f000000u+uint32_t(i);std::memcpy(&cache[i],&bits,4);}
            for(size_t row=0;row<rows;++row)for(size_t t=0;t<chunk;++t) {
                slots[row*chunk+t]=float(t+2);
                if(pattern==1&&t>=chunk/2)active[row*chunk+t]=0;
                if(pattern==2){slots[row*chunk+t]=float((t*3)%capacity);if(t==1)active[row*chunk+t]=0;}
                if(pattern==3)slots[row*chunk+t]=float((t%2)+5);
            }
            expected=cache;
            for(size_t row=0;row<rows;++row)for(size_t t=0;t<chunk;++t)if(active[row*chunk+t])
                for(size_t head=0;head<heads;++head)for(size_t col=0;col<dim;++col)
                    expected[row*width+(head*capacity+size_t(slots[row*chunk+t]))*dim+col]=token[(row*chunk+t)*heads*dim+head*dim+col];
            auto a=gpu.Upload(token,rows*chunk,heads*dim),c=gpu.Upload(cache,rows,width),s=gpu.Upload(slots,rows,chunk),flags=gpu.Upload(active,rows,chunk),out=gpu.Allocate(rows,width);
            gpu.Begin();gpu.EncodeKVChunk(a,c,s,flags,out,heads,dim,capacity,chunk,pattern<=1);gpu.SubmitAndWait();const auto actual=gpu.Download(out);
            if(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)))throw std::runtime_error("Chunk KV copy differs from independent latest-active-write reference");
        }
        auto input=gpu.Upload({1,2,3,4},2,2);
        // Check aligned vector copies and irregular scalar fallbacks against
        // independent row/group indexing, including padded matrix rows.
        for(size_t rows:{1,2,5})for(size_t groups:{1,3})
            for(size_t width:{7,12,20})for(size_t start:{0,1,4})for(size_t count:{1,3,4,8}) {
                if(start+count>width)continue;
                std::vector<float> values(rows*groups*width),expected(rows*groups*count);
                for(size_t index=0;index<values.size();++index) {
                    const uint32_t bits=index%2?0x80000000u:0x3f000000u+uint32_t(index);
                    std::memcpy(&values[index],&bits,sizeof(bits));
                }
                for(size_t row=0;row<rows;++row)for(size_t group=0;group<groups;++group)
                    for(size_t col=0;col<count;++col)
                        expected[(row*groups+group)*count+col]=values[(row*groups+group)*width+start+col];
                auto source=gpu.Upload(values,rows,groups*width),slice=gpu.Allocate(rows,groups*count);
                gpu.Begin();gpu.EncodeSliceGroups(source,slice,groups,width,start,count);gpu.SubmitAndWait();
                const auto actual=gpu.Download(slice);
                if(std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(float)))
                    throw std::runtime_error("Grouped slice differs from independent bitwise reference");
            }
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
