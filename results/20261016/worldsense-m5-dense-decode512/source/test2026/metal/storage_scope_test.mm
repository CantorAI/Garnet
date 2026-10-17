#include "metal_context.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(){try {using Garnet::MetalContext;using Garnet::MetalMatrix;MetalContext gpu;auto base=MetalContext::MemoryStats().allocated;
@autoreleasepool {
 MetalMatrix alias;int value=0;MetalContext::StorageScope([&]{auto a=gpu.Allocate(1024,1024);gpu.Zero(a);alias=a;value=7;});need(value==7,"Callback value lost");need(MetalContext::MemoryStats().allocated>=base+4194304,"Live alias released");alias=MetalMatrix();need(MetalContext::MemoryStats().allocated==base,"Closed alias stayed autoreleased");
 bool thrown=false;try{MetalContext::StorageScope([&]{auto a=gpu.Allocate(1024,1024);gpu.Zero(a);throw std::runtime_error("storage failure");});}catch(const std::runtime_error&e){thrown=std::string(e.what())=="storage failure";}need(thrown,"Exception type/payload changed");need(MetalContext::MemoryStats().allocated==base,"Exceptional pool retained allocation");
 MetalContext::StorageScope([&]{MetalContext::StorageScope([&]{value=9;});});need(value==9,"Nested callback failed");
 MetalMatrix survivor;MetalContext::StorageScope([&]{auto a=gpu.Upload(std::vector<float>(1024,1),1,1024);survivor=gpu.Allocate(1,1024);gpu.Begin();gpu.EncodeAdd(a,a,survivor);gpu.SubmitAndWait();});auto result=gpu.Download(survivor);need(std::all_of(result.begin(),result.end(),[](float x){return x==2;}),"GPU output owner invalid after drain");
 bool aborted=false;try{MetalContext::StorageScope([&]{auto a=gpu.Upload(std::vector<float>(1024,3),1,1024);auto out=gpu.Allocate(1,1024);gpu.Begin();gpu.EncodeAdd(a,a,out);throw std::runtime_error("pending");});}catch(const std::runtime_error&){gpu.Abort();aborted=true;}need(aborted,"Pending exception lost");
 auto a=gpu.Upload(std::vector<float>(1024,4),1,1024),out=gpu.Allocate(1,1024);gpu.Begin();gpu.EncodeAdd(a,a,out);gpu.SubmitAndWait();result=gpu.Download(out);need(std::all_of(result.begin(),result.end(),[](float x){return x==8;}),"GPU abort recovery failed");
}
need(MetalContext::MemoryStats().allocated==base,"Outer scope leaked resource");std::cout<<"Storage boundary scope values/nesting/alias lifetime/exception drain/GPU owner/abort recovery PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
