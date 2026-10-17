#include "metal_context.h"
#include "primary_32_contract.h"
#include "primary_block_codec.h"
#include <vector>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <iostream>
using namespace Garnet;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}

#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
static void checkPrefillTiles(MetalContext& gpu,const MetalBF16Weight& weight,const std::vector<uint16_t>& bits,size_t rows,size_t inner) {
 const auto before=MetalContext::PrimaryStats();
 for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,32},{64,64}}) {
  auto actual=gpu.PrimaryReadPrefillTileWords(weight,0,rows,geometry.first,geometry.second);
  need(actual.size()==bits.size(),"Prefill tile full word count mismatch");
  for(size_t i=0;i<bits.size();++i)need(actual[i]==uint32_t(bits[i])<<16,"Prefill tile changed original FP32 bits");
  const size_t first=((rows-1)/geometry.first)*geometry.first,count=rows-first;
  auto tail=gpu.PrimaryReadPrefillTileWords(weight,first,count,geometry.first,geometry.second);
  need(tail.size()==count*inner,"Prefill tile tail word count mismatch");
  for(size_t i=0;i<tail.size();++i)need(tail[i]==uint32_t(bits[first*inner+i])<<16,"Prefill tile nonzero-row/tail bits mismatch");
 }
 const auto after=MetalContext::PrimaryStats();need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Prefill diagnostic retained a primary owner");
}
#endif

int main(){try{
 // Pure integer boundary checks require no multi-GiB allocation.
 need(Primary32::ShapeFits(260,256,260ULL*256*2),"Valid primary32 shape rejected");
 need(Primary32::ShapeFits(8388608,128,Primary32::MaxRawBytes),"Exact2GiB bound rejected");
 need(!Primary32::ShapeFits(8388612,128,Primary32::MaxRawBytes+1024),"Over2GiB admitted");
 need(!Primary32::ShapeFits(UINT64_MAX,128,1024)&&!Primary32::ShapeFits(260,UINT64_MAX,1024),"Overflow dimensions admitted");
 need(!Primary32::ShapeFits(uint64_t(UINT32_MAX)+1,128,1024)&&!Primary32::ShapeFits(260,uint64_t(UINT32_MAX)+1,1024),"Overuint32 dimensions admitted");
 need(!Primary32::ShapeFits(0,128,0)&&!Primary32::ShapeFits(260,0,0)&&!Primary32::ShapeFits(256,128,65536)&&!Primary32::ShapeFits(257,128,65792)&&!Primary32::ShapeFits(260,129,67080),"Unsupported shape admitted");
 need(!Primary32::ShapeFits(260,256,260ULL*256*2-1),"Mismatched byte count admitted");
 need(Primary32::NarrowMetadata(UINT32_MAX)==UINT32_MAX,"Maximum uint32 metadata changed");
 {bool bad=false;try{Primary32::NarrowMetadata(uint64_t(UINT32_MAX)+1);}catch(...){bad=true;}need(bad,"Metadata truncation accepted");}
 {std::array<uint16_t,PrimaryBlocks::Words> words;words.fill(0x3f80);
  auto edge=PrimaryBlocks::Encode(words,Primary32::MaxRawBytes-256);
  need(Primary32::NarrowMetadata(edge.metadata)==edge.metadata,"Representable high offset changed");
  auto tooHigh=PrimaryBlocks::Encode(words,Primary32::MaxRawBytes);bool bad=false;try{Primary32::NarrowMetadata(tooHigh.metadata);}catch(...){bad=true;}need(bad,"Unrepresentable encoded offset accepted");}
 MetalContext gpu;auto initial=MetalContext::PrimaryStats();
 {
 std::vector<uint16_t> bits;for(uint32_t i=0;i<65536;++i)if(((i>>7)&255)!=255)bits.push_back(uint16_t(i));bits.resize(512*128,0x8000);
 auto w=gpu.UploadBF16(bits.data(),bits.size()*2,512,128);auto shared=w;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(w.PrimaryReady(),"Finite-pattern matrix not packed");
 auto now=MetalContext::PrimaryStats();need(now.activeWeights==initial.activeWeights+1&&now.originalBytes==initial.originalBytes+bits.size()*2,"Owner accounting mismatch");
 auto info=MetalContext::WeightBufferInfo(w);need(info.size()==2&&info[0].representation=="primary_bf16_payload"&&info[1].representation=="primary_bf16_metadata","Packed weight retained a raw duplicate");
 auto vectorActual=gpu.PrimaryReadVectorWords(w,0,bits.size());auto actual=gpu.PrimaryReadWords(w,0,bits.size());need(actual==vectorActual,"Finite scalar/vector GPU words differ");for(size_t i=0;i<bits.size();++i)need(actual[i]==uint32_t(bits[i])<<16,"Finite GPU bits mismatch");
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
 checkPrefillTiles(gpu,w,bits,512,128);
#endif
 w=MetalBF16Weight();need(MetalContext::PrimaryStats().activeWeights==now.activeWeights,"Shared owner was freed");
 need(gpu.PrimaryReadWords(shared,7,100)[0]==uint32_t(bits[7])<<16,"Nonzero read offset mismatch");
 for(size_t start:{size_t(7),size_t(125),size_t(507),bits.size()-100}){auto v=gpu.PrimaryReadVectorWords(shared,start,100);for(size_t j=0;j<100;++j)need(v[j]==uint32_t(bits[start+j])<<16,"Vector partial/block/tail offset mismatch");}
 for(int which:{0,1,2}){bool bad=false;try{gpu.PrimaryReadVectorWords(shared,which==0?bits.size()-1:0,which==0?2:which==1?0:8*1024*1024+1);}catch(...){bad=true;}need(bad,"Vector reader invalid bound accepted");}
 gpu.Begin();{bool bad=false;try{gpu.PrimaryReadVectorWords(shared,0,16);}catch(...){bad=true;}need(bad,"Vector reader admitted an open batch");}gpu.Abort();

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
 need(w.PrimaryReady(),"Mixed matrix not packed");auto vectorActual=gpu.PrimaryReadVectorWords(w,0,mix.size());auto actual=gpu.PrimaryReadWords(w,0,mix.size());need(actual==vectorActual,"Mixed scalar/vector GPU words differ");for(size_t i=0;i<mix.size();++i)need(actual[i]==uint32_t(mix[i])<<16,"Mixed raw/compact GPU bits mismatch");
#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
 checkPrefillTiles(gpu,w,mix,260,256);
 for(int which=0;which<6;++which) {
  bool bad=false;
  try{gpu.PrimaryReadPrefillTileWords(w,which==0?1:which==2?320:0,which==1?0:which==3?4:64,which==4?16:64,which==5?16:32);}catch(...){bad=true;}
  need(bad,"Prefill tile malformed range/geometry accepted");
 }
 gpu.Begin();{bool bad=false;try{gpu.PrimaryReadPrefillTileWords(w,0,64,64,32);}catch(...){bad=true;}need(bad,"Prefill tile admitted an open batch");}gpu.Abort();
 need(gpu.PrimaryReadPrefillTileWords(w,256,4,64,32)[0]==uint32_t(mix[256*256])<<16,"Prefill tile recovery after Abort failed");
#endif
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

#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL
 {
  // Valid shape/range with >8Miwords isolates the readback cap from bounds.
  std::vector<uint16_t> large(65540*128,0x3f80);auto weight=gpu.UploadBF16(large.data(),large.size()*2,65540,128);
  bool bad=false;try{gpu.PrimaryReadPrefillTileWords(weight,0,65540,64,32);}catch(...){bad=true;}need(bad,"Prefill tile readback cap not enforced");
 }
 std::cout<<"Vector prefill tile all3 geometries/full finite/mixed words/nonzero rows/tail260/closed-batch/cap/recovery PASS"<<std::endl;
#endif
 std::vector<uint16_t> raw(257*129,0x3f80);auto fallback=gpu.UploadBF16(raw.data(),raw.size()*2,257,129);need(!fallback.PrimaryReady(),"Unsupported dimensions packed");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
 for(int stage:{1,2}){auto before=MetalContext::PrimaryStats();MetalContext::PrimaryTestFailAfter(stage);bool rejected=false;try{gpu.UploadBF16(mix.data(),mix.size()*2,260,256);}catch(...){rejected=true;}auto after=MetalContext::PrimaryStats();need(rejected&&after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Partial primary allocation failed cleanup");}
 auto recovered=gpu.UploadBF16(mix.data(),mix.size()*2,260,256);need(recovered.PrimaryReady(),"Primary allocation did not recover");
#endif
 std::cout<<"Primary storage finite65536-filtered/mixed/address/owner/dedup/no-master/fallback/allocation-recovery PASS"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
