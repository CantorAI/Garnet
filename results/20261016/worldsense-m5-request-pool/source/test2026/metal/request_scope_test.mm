#include "metal_context.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <atomic>
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{Garnet::MetalContext gpu;
#ifdef GARNET_METAL_REQUEST_AUTORELEASE_POOL
 constexpr uint64_t enabled=1;
#else
 constexpr uint64_t enabled=0;
#endif
 const auto base=Garnet::MetalContext::MemoryStats().allocated;
 @autoreleasepool {
  Garnet::MetalMatrix alias;auto before=Garnet::MetalContext::RequestScopeCount();
  auto value=Garnet::MetalContext::RequestScope([&]{auto a=gpu.Allocate(1024,1024);gpu.Zero(a);alias=a;return std::string("value preserved");});
  need(value=="value preserved"&&Garnet::MetalContext::RequestScopeCount()-before==enabled,"Normal value/counter mismatch");
  auto retained=Garnet::MetalContext::MemoryStats().allocated;need(retained>=base+4*1024*1024,"Pool released retained native owner");
  alias=Garnet::MetalMatrix();auto afterAlias=Garnet::MetalContext::MemoryStats().allocated;
  need(enabled?afterAlias==base:afterAlias>=retained,"Wrong pooled/unpooled allocation lifetime");
  bool thrown=false;before=Garnet::MetalContext::RequestScopeCount();
  try{Garnet::MetalContext::RequestScope([&]()->std::string{auto a=gpu.Allocate(1024,1024);gpu.Zero(a);throw std::runtime_error("expected scope exception");});}
  catch(const std::runtime_error&e){need(std::string(e.what())=="expected scope exception","Exception payload changed");thrown=true;}
  need(thrown&&Garnet::MetalContext::RequestScopeCount()-before==enabled,"Exception/counter mismatch");
  if(enabled)need(Garnet::MetalContext::MemoryStats().allocated==base,"Exceptional scope did not drain");
  before=Garnet::MetalContext::RequestScopeCount();
  value=Garnet::MetalContext::RequestScope([&]{return Garnet::MetalContext::RequestScope([]{return std::string("nested");});});
  need(value=="nested"&&Garnet::MetalContext::RequestScopeCount()-before==2*enabled,"Nested scope mismatch");
  Garnet::MetalMatrix survivor;
  value=Garnet::MetalContext::RequestScope([&]{auto a=gpu.Upload(std::vector<float>(1024,1),1,1024);survivor=gpu.Allocate(1,1024);gpu.Begin();gpu.EncodeAdd(a,a,survivor);gpu.SubmitAndWait();return std::string("recovered");});
  auto result=gpu.Download(survivor);need(value=="recovered"&&std::all_of(result.begin(),result.end(),[](float v){return v==2;}),"Surviving GPU output/recovery failed");
  bool aborted=false;
  try{Garnet::MetalContext::RequestScope([&]()->std::string{auto a=gpu.Upload(std::vector<float>(1024,3),1,1024);auto out=gpu.Allocate(1,1024);gpu.Begin();gpu.EncodeAdd(a,a,out);throw std::runtime_error("abort pending");});}
  catch(const std::runtime_error&){gpu.Abort();aborted=true;}
  need(aborted,"Pending scope exception disappeared");
  auto a=gpu.Upload(std::vector<float>(1024,4),1,1024),out=gpu.Allocate(1,1024);gpu.Begin();gpu.EncodeAdd(a,a,out);gpu.SubmitAndWait();result=gpu.Download(out);need(std::all_of(result.begin(),result.end(),[](float v){return v==8;}),"Abort recovery failed");
 }
 need(Garnet::MetalContext::MemoryStats().allocated==base,"Outer pool did not restore baseline resource accounting");
 std::atomic<int> entered{0},bad{0};
 auto worker=[&]{auto prior=Garnet::MetalContext::RequestScopeCount();auto response=Garnet::MetalContext::RequestScope([&]{entered.fetch_add(1);while(entered.load()<2)std::this_thread::yield();return std::string("thread");});if(response!="thread"||Garnet::MetalContext::RequestScopeCount()-prior!=enabled)bad.fetch_add(1);};
 std::thread first(worker),second(worker);first.join();second.join();need(bad.load()==0,"Overlapping calls contaminate per-thread invocation counts");
 std::cout<<"Request scope normal/exception/nested values, counters, retained owners, exact allocation release, GPU output and pending-abort recovery PASS enabled="<<enabled<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
