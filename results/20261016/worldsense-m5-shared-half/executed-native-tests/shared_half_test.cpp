// Correctness and storage accounting only; no inference performance measurements.
#include "metal_context.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){try{
 Garnet::MetalContext gpu;size_t cases=0;
 for(auto shape:std::vector<std::pair<size_t,size_t>>{{7,17},{1024,896},{2560,512},{2560,4096},{9216,2560},{2048,6144}}){
  auto k=shape.first,n=shape.second;std::vector<uint16_t> bits(k*n);
  for(size_t row=0;row<n;++row)for(size_t col=0;col<k;++col){float f=std::ldexp(float(int(col%17)-8)/128,int(row%7)-3);uint32_t b;std::memcpy(&b,&f,4);bits[row*k+col]=uint16_t(b>>16);}
  auto shared=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);auto original=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,true);
  need(shared.SharedExactHalfReady(),"Expected immutable shared layout was not admitted");
  need(shared.RetainedStorageBytes()<=bits.size()*2+n*260,"Shared layout retained a second full weight buffer");
  need(original.RetainedStorageBytes()==bits.size()*2&&!original.SharedExactHalfReady(),"Original control storage changed");
  for(size_t b:{size_t(1),size_t(2)}){
   std::vector<float> values(b*k);for(size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.013f)*.3f;
   auto a=gpu.Upload(values,b,k),x=gpu.Allocate(b,n),y=gpu.Allocate(b,n);
   for(bool specialized:{false,true}){
    if(specialized)need(gpu.PrepareBF16BodyShape(a,shared,x)&&gpu.PrepareBF16BodyShape(a,original,y),"Body specialization unavailable");
    gpu.Begin();gpu.EncodeBF16MatVec(a,shared,x);gpu.EncodeBF16MatVec(a,original,y);gpu.SubmitAndWait();auto actual=gpu.Download(x),expected=gpu.Download(y);
    need(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"Shared decode differs in FP32 bits");
    if(specialized)need(gpu.LastBodyShapeDispatches()==2,"Specialized routes were not used");
    for(size_t row=0;row<b;++row)for(size_t col=0;col<n;++col){double ref=0,absolute=0;for(size_t j=0;j<k;++j){uint32_t raw=uint32_t(bits[col*k+j])<<16;float f;std::memcpy(&f,&raw,4);double term=double(values[row*k+j])*f;ref+=term;absolute+=std::abs(term);}double unit=std::ldexp(1.,-23),rounds=std::ceil(double(k)/32)+32,gamma=rounds*unit/(1-rounds*unit);need(std::isfinite(actual[row*n+col])&&std::abs(double(actual[row*n+col])-ref)<=gamma*absolute+1e-7,"Shared decode exceeds original FP64 envelope");}
    ++cases;
   }
  }
 }
 // A tiny exact value becomes a half subnormal under its row scale.
 const uint16_t edge[]={0x3f80,0x2c00,0x8000,0,0x3e80,0xbf80,0xac00,0,0x8000,0xbe80};
 auto w=gpu.UploadBF16(edge,sizeof(edge),2,5);need(w.SharedExactHalfReady(),"Half-subnormal fixture was not admitted");
 auto native=gpu.UploadBF16(edge,sizeof(edge),2,5,true);auto a=gpu.Upload({.37f,-.5f,.25f,.125f,-.3f},1,5),x=gpu.Allocate(1,2),y=gpu.Allocate(1,2);
 gpu.Begin();gpu.EncodeBF16MatVec(a,w,x);gpu.EncodeBF16MatVec(a,native,y);gpu.SubmitAndWait();auto vx=gpu.Download(x),vy=gpu.Download(y);need(!std::memcmp(vx.data(),vy.data(),vx.size()*4),"Subnormal/zero reconstruction changed bits");
 const uint16_t unrepresentable[]={0x3f80,0x0080};auto fallback=gpu.UploadBF16(unrepresentable,4,1,2);need(!fallback.SharedExactHalfReady()&&!fallback.ExactHalfReady(),"Nonrepresentable layout was admitted");
 Garnet::MetalContext other;auto z=other.Allocate(1,2);gpu.Begin();other.Begin();gpu.EncodeBF16MatVec(a,w,x);other.EncodeBF16MatVec(a,w,z);other.SubmitAndWait();gpu.SubmitAndWait();need(gpu.Download(x)==other.Download(z),"Immutable shared weights failed across GPU contexts");
 gpu.Begin();gpu.EncodeBF16MatVec(a,w,x);gpu.Abort();bool rejected=false;try{gpu.Download(x);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted output remained readable");gpu.Begin();gpu.EncodeBF16MatVec(a,w,x);gpu.SubmitAndWait();need(gpu.Download(x)==vx,"Shared weights failed recovery");
 std::cout<<"Shared exact-half decode FP32-bit/FP64, real widths, one-buffer storage, subnormals, fallback, cross-context and abort recovery PASS "<<cases<<" numeric cases\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
