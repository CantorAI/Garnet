#include "metal_context.h"
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <iostream>
using namespace Garnet;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){try{
 MetalContext gpu;auto initial=MetalContext::PrimaryStats();
 {
 std::vector<uint16_t> bits;for(uint32_t i=0;i<65536;++i)if(((i>>7)&255)!=255)bits.push_back(uint16_t(i));bits.resize(512*128,0x8000);
 auto w=gpu.UploadBF16(bits.data(),bits.size()*2,512,128);auto shared=w;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(w.PrimaryReady(),"Finite-pattern matrix not packed");
 auto now=MetalContext::PrimaryStats();need(now.activeWeights==initial.activeWeights+1&&now.originalBytes==initial.originalBytes+bits.size()*2,"Owner accounting mismatch");
 auto info=MetalContext::WeightBufferInfo(w);need(info.size()==2&&info[0].representation=="primary_bf16_payload"&&info[1].representation=="primary_bf16_metadata","Packed weight retained a raw duplicate");
 auto actual=gpu.PrimaryReadWords(w,0,bits.size());for(size_t i=0;i<bits.size();++i)need(actual[i]==uint32_t(bits[i])<<16,"Finite GPU bits mismatch");
 w=MetalBF16Weight();need(MetalContext::PrimaryStats().activeWeights==now.activeWeights,"Shared owner was freed");
 need(gpu.PrimaryReadWords(shared,7,100)[0]==uint32_t(bits[7])<<16,"Nonzero read offset mismatch");
 bool rejected=false;try{gpu.PrimaryReadWords(shared,bits.size()-1,2);}catch(...){rejected=true;}need(rejected,"Invalid read range accepted");
#else
 need(!w.PrimaryReady(),"Disabled packed admission");
#endif
 }
 need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights&&MetalContext::PrimaryStats().allocatedBytes==initial.allocatedBytes,"Released owners still charged");
 // Alternating wide-exponent/raw and compact blocks, plus unsupported dimensions.
 std::vector<uint16_t> mix(260*256);
 for(size_t r=0;r<260;++r)for(size_t k=0;k<256;++k){const size_t block=(r/4)*2+k/128;const unsigned e=block%4==0?(k%2?96:130):120+unsigned(k%4);mix[r*256+k]=uint16_t((e<<7)|((r*17+k*5)&127)|((k&1)<<15));}
 {
 auto w=gpu.UploadBF16(mix.data(),mix.size()*2,260,256);
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(w.PrimaryReady(),"Mixed matrix not packed");auto actual=gpu.PrimaryReadWords(w,0,mix.size());for(size_t i=0;i<mix.size();++i)need(actual[i]==uint32_t(mix[i])<<16,"Mixed raw/compact GPU bits mismatch");
#endif
 }
 for(size_t batch:{1,2}) {
  const size_t k=2560,n=260;std::vector<uint16_t> bits(k*n);for(size_t i=0;i<bits.size();++i)bits[i]=uint16_t((120u+unsigned(i%4))<<7|unsigned(i%127)|unsigned((i&1)<<15));
  auto packed=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);auto rawControl=gpu.UploadBF16(bits.data(),bits.size()*2,n,k,true);
  std::vector<float> values(batch*k);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%29)-14)*.017f;
  auto input=gpu.Upload(values,batch,k),out=gpu.Allocate(batch,n),reference=gpu.Allocate(batch,n);auto count=gpu.BodyShapePipelineCount();
  need(gpu.PrepareBF16BodyShape(input,packed,out)&&gpu.PrepareBF16BodyShape(input,rawControl,reference),"Same-shape PSO preparation failed");
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
  need(gpu.BodyShapePipelineCount()==count+2,"Raw/packed pipeline keys collided");
#else
  need(gpu.BodyShapePipelineCount()==count+1,"Raw duplicate pipeline entry");
#endif
  gpu.Begin();gpu.EncodeBF16MatVec(input,rawControl,reference);gpu.SubmitAndWait();auto wanted=gpu.Download(reference);
  gpu.Begin();gpu.EncodeBF16MatVec(input,packed,out);bool rejected=false;try{gpu.Download(out);}catch(...){rejected=true;}need(rejected,"Pending packed output readable");
  {MetalContext other;other.Begin();rejected=false;try{other.EncodeBF16MatVec(input,packed,out);}catch(...){rejected=true;}other.Abort();need(rejected,"Foreign pending writer accepted");}
  gpu.Abort();rejected=false;try{gpu.Download(out);}catch(...){rejected=true;}need(rejected,"Aborted packed output readable");
  gpu.Begin();gpu.EncodeBF16MatVec(input,packed,out);gpu.SubmitAndWait();auto actual=gpu.Download(out);need(actual.size()==wanted.size()&&!std::memcmp(actual.data(),wanted.data(),wanted.size()*4),"Packed tail260/Abort recovery differs from same-shape raw control");
 }
 std::vector<uint16_t> raw(257*129,0x3f80);auto fallback=gpu.UploadBF16(raw.data(),raw.size()*2,257,129);need(!fallback.PrimaryReady(),"Unsupported dimensions packed");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
 for(int stage:{1,2}){auto before=MetalContext::PrimaryStats();MetalContext::PrimaryTestFailAfter(stage);bool rejected=false;try{gpu.UploadBF16(mix.data(),mix.size()*2,260,256);}catch(...){rejected=true;}auto after=MetalContext::PrimaryStats();need(rejected&&after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Partial primary allocation failed cleanup");}
 auto recovered=gpu.UploadBF16(mix.data(),mix.size()*2,260,256);need(recovered.PrimaryReady(),"Primary allocation did not recover");
#endif
 std::cout<<"Primary storage finite65536-filtered/mixed/address/owner/dedup/no-master/fallback/allocation-recovery PASS"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
