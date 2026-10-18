// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <limits>
using namespace Garnet;
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static float f32(uint32_t v){float out;std::memcpy(&out,&v,4);return out;}
template<class F>static void rejects(F f,const char* why){bool failed=false;try{f();}catch(...){failed=true;}need(failed,why);}
int main(int argc,char** argv){try{
 bool enabled=false;bool direct=false;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_TEXT
 enabled=true;direct=true;
#endif
 need(argc==2,"Output path required");std::ofstream binary(argv[1],std::ios::binary);need(bool(binary),"Output open failed");
 auto initial=MetalContext::SoleStats();need(initial.activeWeights==0&&initial.scratchBytes==0,"Leaked initial weights/scratch");
 {
 MetalContext gpu;std::ofstream shader(std::string(argv[1])+".metal");shader<<gpu.SoleShaderProgramSource();
 size_t cases=0,records=0;double worst=0;
 for(size_t columns:{2048,1024,151936}){
 size_t inner=2048;size_t count=inner*columns;
 std::vector<uint16_t> raw(count);
 for(size_t i=0;i<count;++i)raw[i]=uint16_t(((i*17+i/inner)%2?0x8000:0)|((123+(i*13+i/inner)%8)<<7)|((i*29+i/inner)%128));
 // Force raw admission in every role; packed shape tests remain original.
 auto weight=gpu.UploadBF16(raw.data(),raw.size()*2,columns,inner,false,false,false);
 auto info=MetalContext::WeightBufferInfo(weight);need(info.size()==1&&info[0].lengthBytes==count*2,"Duplicate/incorrect weight storage");
 need(info[0].representation==(enabled?"sole_bf16_interleaved":"bf16"),"Layout representation mismatch");
 need(!weight.PrimaryReady()&&!weight.InterleavedBodyReady()&&!weight.ExactHalfReady(),"Unexpected auxiliary representation");
 for(size_t start=0;start<count;start+=1048576){size_t n=std::min(size_t(1048576),count-start);auto host=gpu.RawBF16ReadWords(weight,start,n);
  need(host.size()==n,"HOST read length");for(size_t j=0;j<n;++j)need(host[j]==uint32_t(raw[start+j])<<16,"HOST logical bits differ");
  if(enabled){auto device=gpu.SoleInterleavedReadWords(weight,start,n);need(device==host,"GPU logical bits differ");}
 }
 if(enabled){rejects([&]{gpu.SoleInterleavedReadWords(weight,count,1);},"Range accepted");rejects([&]{gpu.SoleInterleavedReadWords(weight,0,0);},"Zero count accepted");}
 if(direct){
 for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,64},{64,32}}){
  size_t tileN=geometry.first,tileK=geometry.second,chunk=(8*1024*1024/inner/tileN)*tileN;
  for(size_t first=0;first<columns;first+=chunk){size_t nr=std::min(chunk,columns-first);auto actual=gpu.SoleInterleavedReadPrefillTileWords(weight,first,nr,tileN,tileK);need(actual.size()==nr*inner,"Native tile word count");for(size_t j=0;j<actual.size();++j)need(actual[j]==uint32_t(raw[first*inner+j])<<16,"Direct prefill tile original bits differ");}
 }
 rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,1,32,32,32);},"Unaligned tile row accepted");
 rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,0,0,32,32);},"Zero tile rows accepted");
 rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,0,32,32,16);},"Wrong tile geometry accepted");
 need(MetalContext::SoleStats().scratchBytes==0,"Direct tile readback allocated restore scratch");
 }
 std::vector<float> saved;
 for(size_t rows:{1,2,3,16,32,64,128}){
 auto a=gpu.Allocate(rows,inner),out=gpu.Allocate(rows,columns);bool prepared=gpu.PrepareBF16BodyShape(a,weight,out);need(prepared==(rows<=2&&columns<100000),"Sole shape-cache admission mismatch");
 for(size_t pass=0;pass<2;++pass){
  // Sparse full-width inputs keep an independent complete FP64 oracle bounded.
  std::vector<float> input(rows*inner,0);
  for(size_t row=0;row<rows;++row)for(size_t j=0;j<33;++j){size_t k=(j*131+row*17+pass*43)%inner;input[row*inner+k]=f32((127u<<23)|uint32_t((j*7919+row*131+pass*65537)&0x7fffff));}
  gpu.Update(a,input);auto before=MetalContext::SoleStats();gpu.Begin();
  if(enabled)rejects([&]{gpu.SoleInterleavedReadWords(weight,0,1);},"Active-batch readback accepted");
  if(direct)rejects([&]{gpu.SoleInterleavedReadPrefillTileWords(weight,0,32,32,32);},"Active-batch tile readback accepted");
  gpu.EncodeBF16MatVec(a,weight,out);gpu.SubmitAndWait();auto actual=gpu.Download(out);auto after=MetalContext::SoleStats();
  need(after.decodeEncoded-before.decodeEncoded==size_t(enabled&&rows<=2),"Wrong sole decode count");
  need(after.restoreEncoded-before.restoreEncoded==size_t(enabled&&rows>2&&!direct),"Wrong sole restore count");
  need(after.prefillEncoded-before.prefillEncoded==size_t(enabled&&rows>2&&direct),"Wrong direct prefill count");
  if(direct)need(after.scratchBytes==0&&after.restoreEncoded==0,"Direct prefill used restore/scratch");
  need(gpu.Download(a)==input,"Projection input mutated");
  for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col){double ref=0,l1=0;
   for(size_t j=0;j<33;++j){size_t k=(j*131+row*17+pass*43)%inner;double product=double(input[row*inner+k])*f32(uint32_t(raw[col*inner+k])<<16);ref+=product;l1+=std::abs(product);}
   double bound=rows<=2?1e-4+2e-6*l1:5e-6+5e-5*std::abs(ref);double error=std::abs(double(actual[row*columns+col])-ref);
   need(std::isfinite(actual[row*columns+col])&&error<=bound,"Full FP64 original bound failed");worst=std::max(worst,error/bound);
  }
  uint64_t shape[4]={rows,inner,columns,pass};binary.write(reinterpret_cast<char*>(shape),sizeof(shape));binary.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);++records;saved=actual;
 }
 ++cases;
 }
 // Interleaved and row-major of identical shape must never share a decode PSO.
 auto bypass=gpu.UploadBF16(raw.data(),raw.size()*2,columns,inner,true,false,false);
 need(MetalContext::WeightBufferInfo(bypass)[0].representation=="bf16","floatPrefill converted");
 auto a=gpu.Upload(std::vector<float>(inner,1.f),1,inner),one=gpu.Allocate(1,columns),two=gpu.Allocate(1,columns);
 need(gpu.PrepareBF16BodyShape(a,bypass,one)==(columns<100000)&&gpu.PrepareBF16BodyShape(a,weight,two)==(columns<100000),"Separate cache/head admission failed");
 gpu.Begin();gpu.EncodeBF16MatVec(a,bypass,one);gpu.EncodeBF16MatVec(a,weight,two);gpu.SubmitAndWait();need(gpu.Download(one)==gpu.Download(two),"Raw/sole cache collision");
 // Two distinct weights, one scratch, a partial submission, and released owners.
 auto raw2=raw;for(auto& v:raw2)v^=0x8000;auto second=gpu.UploadBF16(raw2.data(),raw2.size()*2,columns,inner,false,false,false);
 auto p=gpu.Upload(std::vector<float>(3*inner,1.f),3,inner),x=gpu.Allocate(3,columns),y=gpu.Allocate(3,columns),z=gpu.Allocate(3,columns);
 gpu.Begin();gpu.EncodeBF16MatVec(p,weight,x);gpu.Flush();gpu.EncodeBF16MatVec(p,second,y);gpu.EncodeBF16MatVec(p,weight,z);second=MetalBF16Weight();gpu.SubmitAndWait();
 auto xv=gpu.Download(x),yv=gpu.Download(y);need(xv==gpu.Download(z),"Scratch overwrite/Flush hazard");for(size_t i=0;i<xv.size();++i)need(xv[i]==-yv[i],"Different weight scratch contamination");
 gpu.Begin();gpu.EncodeBF16MatVec(p,weight,x);gpu.Flush();gpu.EncodeBF16MatVec(p,weight,z);gpu.Abort();rejects([&]{gpu.Download(x);},"Aborted output readable");
 gpu.Begin();gpu.EncodeBF16MatVec(p,weight,x);gpu.SubmitAndWait();need(gpu.Download(x)==xv,"Abort restore recovery differs");
 {MetalContext peer;auto pa=peer.Upload(std::vector<float>(3*inner,1.f),3,inner),po=peer.Allocate(3,columns);peer.Begin();peer.EncodeBF16MatVec(pa,weight,po);peer.SubmitAndWait();need(peer.Download(po)==xv,"Shared-owner peer context differs");}
 auto bad=raw;bad[0]=0x7f80;rejects([&]{gpu.UploadBF16(bad.data(),bad.size()*2,columns,inner,false,false,false);},"Nonfinite admitted");
 if(enabled){
#if defined(GARNET_METAL_SOLE_INTERLEAVED_MLP) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
  for(int stage:{1,2}){auto stats=MetalContext::SoleStats();MetalContext::SoleTestFailAfter(stage);rejects([&]{gpu.UploadBF16(raw.data(),raw.size()*2,columns,inner,false,false,false);},"Allocation fault not raised");need(MetalContext::SoleStats().activeWeights==stats.activeWeights&&MetalContext::SoleStats().allocatedBytes==stats.allocatedBytes,"Weight failure charged/leaked");}
  MetalContext::SoleTestUnavailable(true);auto fallback=gpu.UploadBF16(raw.data(),raw.size()*2,columns,inner,false,false,false);MetalContext::SoleTestUnavailable(false);need(MetalContext::WeightBufferInfo(fallback)[0].representation=="bf16","Unsupported path converted");
  {MetalContext fresh;auto fa=fresh.Upload(std::vector<float>(3*inner,1.f),3,inner),fo=fresh.Allocate(3,columns);auto stats=MetalContext::SoleStats();MetalContext::SoleTestFailAfter(3);fresh.Begin();rejects([&]{fresh.EncodeBF16MatVec(fa,weight,fo);},"Prefill preparation/dispatch fault not raised");fresh.Abort();need(MetalContext::SoleStats().scratchBytes==stats.scratchBytes,"Scratch failure charged/leaked");fresh.Begin();fresh.EncodeBF16MatVec(fa,weight,fo);fresh.SubmitAndWait();need(fresh.Download(fo)==xv,"Prefill fault recovery differs");}
#endif
 }
 }
 auto small=gpu.UploadBF16(std::vector<uint16_t>(256*128,0x3f80).data(),256*128*2,256,128,false,false,false);need(MetalContext::WeightBufferInfo(small)[0].representation=="bf16","Unsupported shape converted");
 need(cases==21&&records==42,"Missing expanded text projection coverage");std::cout<<"sole enabled="<<enabled<<" cases="<<cases<<" records="<<records<<" worst_original_FP64_bound_fraction="<<worst<<"\n";
 }
 auto final=MetalContext::SoleStats();need(final.activeWeights==0&&final.allocatedBytes==0&&final.scratchBytes==0,"Sole weight/scratch lifetime leak");
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
