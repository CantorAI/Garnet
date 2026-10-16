// Correctness/route/ownership tests only; WorldSense supplies performance data.
#include "metal_context.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(){try{
 Garnet::MetalContext gpu;size_t checks=0;
 for(auto shape:std::vector<std::pair<size_t,size_t>>{{7,17},{7,65},{63,127},{65,128},{1024,896},{2560,9216},{9216,2560},{2048,6144}}){
  auto k=shape.first,n=shape.second;std::vector<uint16_t> weights(k*n);for(size_t i=0;i<weights.size();++i){float value=float(int(i%17)-8)/128;uint32_t bits;std::memcpy(&bits,&value,4);weights[i]=uint16_t(bits>>16);}auto w=gpu.UploadBF16(weights.data(),weights.size()*2,n,k);
  for(size_t b:{size_t(1),size_t(2),size_t(3),size_t(16),size_t(32),size_t(64)}){
   std::vector<float> values(b*k);for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.013f)*.3f;auto a=gpu.Upload(values,b,k),x=gpu.Allocate(b,n),y=gpu.Allocate(b,n);
   if(b<=2)need(gpu.PrepareBF16BodyShape(a,w,x),"Decode specialization setup failed");
   gpu.SetCoalescedPrefill(false);gpu.Begin();gpu.EncodeBF16MatVec(a,w,x);gpu.SubmitAndWait();auto expected=gpu.Download(x);need(gpu.LastCoalescedPrefillDispatches()==0,"Reference encoded candidate path");
   gpu.SetCoalescedPrefill(true);gpu.Begin();need(gpu.LastCoalescedPrefillDispatches()==0,"Begin did not clear route count");gpu.EncodeBF16MatVec(a,w,y);bool pending=false;try{gpu.Download(y);}catch(const std::logic_error&){pending=true;}need(pending,"Pending coalesced output became readable");gpu.SubmitAndWait();auto actual=gpu.Download(y);need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"Coalesced prefill changed FP32 bits");need(gpu.LastCoalescedPrefillDispatches()==size_t(b>=16&&n>=64),"Actual wide route/fallback count mismatch");
   for(size_t row=0;row<b;++row)for(size_t col=0;col<n;++col){double reference=0,absolute=0;for(size_t j=0;j<k;++j){uint32_t raw=uint32_t(weights[col*k+j])<<16;float value;std::memcpy(&value,&raw,4);double term=double(values[row*k+j])*value;reference+=term;absolute+=std::abs(term);}double unit=std::ldexp(1.,-23),rounds=std::ceil(double(k)/32)+32,gamma=rounds*unit/(1-rounds*unit);need(std::isfinite(actual[row*n+col])&&std::abs(double(actual[row*n+col])-reference)<=gamma*absolute+1e-7,"Coalesced output exceeds unchanged FP64 envelope");}
   gpu.Begin();gpu.EncodeBF16MatVec(a,w,y);bool rejected=false;try{gpu.SetCoalescedPrefill(false);}catch(const std::logic_error&){rejected=true;}need(rejected,"Policy changed during pending command batch");gpu.Abort();need(gpu.LastCoalescedPrefillDispatches()==0,"Abort did not clear route count");rejected=false;try{gpu.Download(y);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted output remained readable");gpu.Begin();gpu.EncodeBF16MatVec(a,w,y);gpu.SubmitAndWait();need(gpu.Download(y)==actual,"Candidate did not recover after abort");++checks;
  }
 }
 std::cout<<"Coalesced wide-prefill complete FP32 bits/FP64, tails, B1/B2 decode and small-row fallback, route counts, ownership and abort recovery PASS "<<checks<<" cases\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
