// Focused native admission, alias/ownership and exact output tests.
#include "metal_context.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace Garnet;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void test(size_t rows,size_t inner,size_t columns) {
    MetalContext gpu;std::vector<uint16_t> raw0(inner*columns),raw1(raw0.size());
    for(size_t i=0;i<raw0.size();++i){raw0[i]=(i%3?0:0x8000)|0x3d00;raw1[i]=(i%5?0:0x8000)|0x3c80;}
    auto w0=gpu.UploadBF16(raw0.data(),raw0.size()*2,columns,inner),w1=gpu.UploadBF16(raw1.data(),raw1.size()*2,columns,inner);
    std::vector<float> values(rows*inner);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%17)-8)*.0625f;
    auto input=gpu.Upload(values,rows,inner),x=gpu.Allocate(rows,columns),y=gpu.Allocate(rows,columns);
    gpu.Begin();gpu.EncodeBF16MatVec(input,w0,x);gpu.EncodeBF16MatVec(input,w1,y);gpu.SubmitAndWait();
    auto ref0=gpu.Download(x),ref1=gpu.Download(y);
    for(int repeat=0;repeat<3;++repeat) {
        gpu.Begin();require(gpu.EncodeJoinedBF16Prefill(input,w0,w1,x,y),"Supported pair rejected");
        require(gpu.LastJoinedPrefillPairs()==1,"Dispatch counter incorrect");gpu.SubmitAndWait();
        auto a=gpu.Download(x),b=gpu.Download(y);require(std::memcmp(a.data(),ref0.data(),a.size()*4)==0&&std::memcmp(b.data(),ref1.data(),b.size()*4)==0,"Joined outputs changed bits");
    }
    // Exercise both offset views, then repack a reversed pair from those views.
    gpu.Begin();gpu.EncodeBF16MatVec(input,w0,x);gpu.EncodeBF16MatVec(input,w1,y);gpu.SubmitAndWait();
    auto solo0=gpu.Download(x),solo1=gpu.Download(y);
    require(std::memcmp(solo0.data(),ref0.data(),solo0.size()*4)==0&&std::memcmp(solo1.data(),ref1.data(),solo1.size()*4)==0,"Individual projections changed after joining");
    gpu.Begin();require(gpu.EncodeJoinedBF16Prefill(input,w1,w0,y,x),"Reversed pair rejected");gpu.SubmitAndWait();
    solo0=gpu.Download(x);solo1=gpu.Download(y);
    require(std::memcmp(solo0.data(),ref0.data(),solo0.size()*4)==0&&std::memcmp(solo1.data(),ref1.data(),solo1.size()*4)==0,"Reversed joined outputs changed bits");
    gpu.Begin();gpu.EncodeBF16MatVec(input,w0,x);gpu.EncodeBF16MatVec(input,w1,y);gpu.SubmitAndWait();
    solo0=gpu.Download(x);solo1=gpu.Download(y);
    require(std::memcmp(solo0.data(),ref0.data(),solo0.size()*4)==0&&std::memcmp(solo1.data(),ref1.data(),solo1.size()*4)==0,"Individual projections changed after reversed joining");
    auto untouched=gpu.Upload(std::vector<float>(rows*columns,99.5f),rows,columns);
    gpu.Begin();require(!gpu.EncodeJoinedBF16Prefill(input,w0,w1,untouched,untouched),"Aliased pair admitted");require(gpu.LastJoinedPrefillPairs()==0,"Alias rejection changed counter");gpu.Abort();
    for(float value:gpu.Download(untouched))require(value==99.5f,"Alias rejection changed output");
    auto bad=gpu.Upload(std::vector<float>(rows*(columns-1),99.5f),rows,columns-1);bool rejected=false;
    gpu.Begin();try{gpu.EncodeJoinedBF16Prefill(input,w0,w1,untouched,bad);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected&&gpu.LastJoinedPrefillPairs()==0,"Invalid shape not rejected before dispatch");gpu.Abort();
    for(float value:gpu.Download(untouched))require(value==99.5f,"Shape rejection changed output");
    auto empty=gpu.Allocate(rows,inner);rejected=false;gpu.Begin();try{gpu.EncodeJoinedBF16Prefill(empty,w0,w1,untouched,x);}catch(const std::logic_error&){rejected=true;}
    require(rejected&&gpu.LastJoinedPrefillPairs()==0,"Uninitialized input admitted");gpu.Abort();
    MetalContext foreign;auto other=foreign.Allocate(rows,columns);foreign.Begin();foreign.EncodeBF16MatVec(input,w0,other);
    rejected=false;gpu.Begin();try{gpu.EncodeJoinedBF16Prefill(input,w0,w1,untouched,other);}catch(const std::logic_error&){rejected=true;}
    require(rejected&&gpu.LastJoinedPrefillPairs()==0,"Foreign pending output admitted");gpu.Abort();foreign.SubmitAndWait();
    auto completed=foreign.Download(other);require(std::memcmp(completed.data(),ref0.data(),completed.size()*4)==0,"Foreign batch was modified");
    auto unsupportedInput=gpu.Upload(std::vector<float>(inner*24,1),24,inner),unsupportedOutput=gpu.Upload(std::vector<float>(columns*24,99.5f),24,columns);
    gpu.Begin();require(!gpu.EncodeJoinedBF16Prefill(unsupportedInput,w0,w1,unsupportedOutput,unsupportedOutput),"Unsupported rows admitted");gpu.Abort();
    for(float value:gpu.Download(unsupportedOutput))require(value==99.5f,"Unsupported pair changed output");
    std::cout<<"joined_prefill PASS rows="<<rows<<" inner="<<inner<<" columns="<<columns<<"\n";
}
int main(){try{test(44,1024,3072);test(26,896,4864);return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
