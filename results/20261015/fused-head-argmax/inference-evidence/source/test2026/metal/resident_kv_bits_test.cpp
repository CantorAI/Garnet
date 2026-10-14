// Private single-slot writes preserve untouched IEEE-754 words and abort rules.
#include "metal_context.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace Garnet;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
 size_t cases=0;
 for(size_t batch:{1,2})for(size_t heads:{1,2})for(size_t dim:{3,128})for(size_t capacity:{7,256}) {
  MetalContext gpu;size_t width=heads*capacity*dim;
  std::vector<float> original(batch*width),tokens(batch*heads*dim),slots(batch),active(batch,1);
  const uint32_t patterns[]={0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x3f800000,0xbf800000};
  for(size_t i=0;i<original.size();++i){uint32_t bits=patterns[i%10];std::memcpy(&original[i],&bits,4);}
  for(size_t i=0;i<tokens.size();++i){uint32_t bits=patterns[(i+3)%10];std::memcpy(&tokens[i],&bits,4);}
  auto token=gpu.Upload(tokens,batch,heads*dim),cache=gpu.Upload(original,batch,width);
  for(size_t iteration=0;iteration<3;++iteration){
   for(size_t row=0;row<batch;++row){slots[row]=float(iteration==0?0:iteration==1?capacity-1:2);active[row]=iteration==2&&row==batch-1?0:1;}
   auto positions=gpu.Upload(slots,batch,1),flags=gpu.Upload(active,batch,1);
   auto expected=gpu.Download(cache);
   for(size_t row=0;row<batch;++row)if(active[row])for(size_t h=0;h<heads;++h)
    std::memcpy(expected.data()+row*width+(h*capacity+size_t(slots[row]))*dim,tokens.data()+(row*heads+h)*dim,dim*4);
   gpu.Begin();
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
   gpu.EncodeKVUpdate(token,cache,positions,flags,cache,heads,dim,capacity);gpu.SubmitAndWait();
   auto actual=gpu.Download(cache);require(actual.size()==expected.size()&&std::memcmp(actual.data(),expected.data(),actual.size()*4)==0,"Cache words changed");
#else
   bool rejected=false;try{gpu.EncodeKVUpdate(token,cache,positions,flags,cache,heads,dim,capacity);}catch(const std::invalid_argument&){rejected=true;}
   gpu.Abort();require(rejected,"Legacy cache alias accepted");gpu.Update(cache,original);
#endif
  }
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
  auto positions=gpu.Upload(slots,batch,1),flags=gpu.Upload(active,batch,1);
  gpu.Begin();gpu.EncodeKVUpdate(token,cache,positions,flags,cache,heads,dim,capacity);gpu.Abort();
  bool unreadable=false;try{gpu.Download(cache);}catch(const std::logic_error&){unreadable=true;}
  require(unreadable,"Aborted cache remained readable");gpu.Update(cache,original);
  gpu.Begin();gpu.EncodeKVUpdate(token,cache,positions,flags,cache,heads,dim,capacity);gpu.SubmitAndWait();
  auto actual=gpu.Download(cache);auto expected=original;
  for(size_t row=0;row<batch;++row)if(active[row])for(size_t h=0;h<heads;++h)
   std::memcpy(expected.data()+row*width+(h*capacity+size_t(slots[row]))*dim,tokens.data()+(row*heads+h)*dim,dim*4);
  require(std::memcmp(actual.data(),expected.data(),actual.size()*4)==0,"Cache abort recovery differs");
#endif
  ++cases;
 }
 std::printf("resident_kv_bits PASS profiles=%zu signed_zero_subnormal_nan_payloads=true\n",cases);return 0;
 }catch(const std::exception& error){std::fprintf(stderr,"resident_kv_bits FAILED: %s\n",error.what());return 1;}}
