// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <array>
#include <algorithm>
using namespace Garnet;
static void need(bool v,const char* m){if(!v)throw std::runtime_error(m);}
template<class F>static void rejects(F f,const char* m){bool yes=false;try{f();}catch(...){yes=true;}need(yes,m);}
static float f32(uint32_t w){float v;std::memcpy(&v,&w,4);return v;}
int main(int argc,char**argv){try{
 bool enabled=false;
#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
 enabled=true;
#endif
 need(argc==2,"Output file required");std::ofstream binary(argv[1],std::ios::binary);need(bool(binary),"Cannot open output");
 const size_t K=2048,N=6144,blocks=N*K/16;size_t records=0;double worst=0;auto initial=MetalContext::SoleStats();need(initial.activeWeights==0&&initial.scratchBytes==0,"Initial owner leak");
 {
 MetalContext gpu;std::ofstream shader(std::string(argv[1])+".metal");shader<<gpu.DualProjectionShaderProgramSource();
 auto execute=[&](MetalContext& ctx,const MetalMatrix& a,const MetalBF16Weight& g,const MetalBF16Weight& u,const MetalMatrix& go,const MetalMatrix& uo){auto before=MetalContext::DualProjectionStats();bool paired=ctx.EncodeSoleDualProjectionDecode(a,g,u,go,uo);need(paired==enabled,"Pair admission mismatch");if(!paired){ctx.EncodeBF16MatVec(a,g,go);ctx.EncodeBF16MatVec(a,u,uo);}auto after=MetalContext::DualProjectionStats();need(after.encodedPairs-before.encodedPairs==uint64_t(enabled)&&after.representedProjections-before.representedProjections==2*uint64_t(enabled),"Pair counter mismatch");};
 for(size_t pair=0;pair<2;++pair){
 std::vector<uint16_t> rawG(N*K),rawU(N*K);
 for(size_t i=0;i<N*K;++i){rawG[i]=uint16_t((((i*17+i/K+pair)%2)?0x8000:0)|((123+(i*13+i/K+pair)%8)<<7)|((i*29+i/K+pair*7)%128));rawU[i]=uint16_t((((i*11+i/K+pair*3)%3)?0x8000:0)|((122+(i*7+i/K+pair)%9)<<7)|((i*19+i/K+pair*31)%128));}
 auto wg=gpu.UploadBF16(rawG.data(),rawG.size()*2,N,K,false,false,false),wu=gpu.UploadBF16(rawU.data(),rawU.size()*2,N,K,false,false,false);
 need(!wg.PrimaryReady()&&!wu.PrimaryReady(),"Forced raw pair packed");
 if(enabled){
  auto before=MetalContext::DualProjectionStats();
  for(size_t first=0;first<blocks;first+=262144){size_t count=std::min(size_t(262144),blocks-first);auto got=gpu.SoleDualProjectionReadWords(wg,wu,first,count);need(got.size()==count*32,"Pair word count");for(size_t i=0;i<count;++i){size_t block=first+i,col=(block/(K/4))*4,k=(block%(K/4))*4;for(size_t j=0;j<4;++j)for(size_t c=0;c<4;++c){need(got[i*32+j*4+c]==uint32_t(rawG[(col+c)*K+k+j])<<16,"Pair gate original word mismatch");need(got[i*32+16+j*4+c]==uint32_t(rawU[(col+c)*K+k+j])<<16,"Pair up original word mismatch");}}}
  auto after=MetalContext::DualProjectionStats();need(after.encodedPairs==before.encodedPairs&&after.representedProjections==before.representedProjections,"Probe changed serving counters");
  rejects([&]{gpu.SoleDualProjectionReadWords(wg,wu,0,0);},"Zero packet accepted");rejects([&]{gpu.SoleDualProjectionReadWords(wg,wu,blocks,1);},"Past-end packet accepted");rejects([&]{gpu.SoleDualProjectionReadWords(wg,wu,0,262145);},"Oversize packet accepted");
 }
 for(size_t rows:{1,2}){
 auto input=gpu.Allocate(rows,K),gref=gpu.Allocate(rows,N),uref=gpu.Allocate(rows,N),gout=gpu.Allocate(rows,N),uout=gpu.Allocate(rows,N);
 gpu.PrepareBF16BodyShape(input,wg,gref);gpu.PrepareBF16BodyShape(input,wu,uref);
 for(size_t pass=0;pass<(pair?4:5);++pass){
  const bool dense=pass==4;std::vector<float> x(rows*K,dense?1.f:0.f);
  if(!dense)for(size_t row=0;row<rows;++row)for(size_t j=0;j<33;++j){size_t k=(j*131+row*17+pair*43)%K;float value=f32((127u<<23)|uint32_t((j*7919+row*131+pair*65537)&0x7fffff));if(j%2)value=-value;if(pass==1)value=-value;else if(pass==2)value*=.5f;else if(pass==3)value*=2.f;x[row*K+k]=value;}
  gpu.Update(input,x);gpu.Begin();gpu.EncodeBF16MatVec(input,wg,gref);gpu.EncodeBF16MatVec(input,wu,uref);gpu.SubmitAndWait();auto expectedG=gpu.Download(gref),expectedU=gpu.Download(uref);
  gpu.Begin();if(enabled)rejects([&]{gpu.SoleDualProjectionReadWords(wg,wu,0,1);},"Active pair readback accepted");execute(gpu,input,wg,wu,gout,uout);gpu.SubmitAndWait();auto actualG=gpu.Download(gout),actualU=gpu.Download(uout);need(actualG==expectedG&&actualU==expectedU,"Complete original projection bits differ");need(gpu.Download(input)==x,"Input changed");
  for(size_t projection=0;projection<2;++projection){const auto& actual=projection?actualU:actualG;const auto& raw=projection?rawU:rawG;
   for(size_t row=0;row<rows;++row)for(size_t col=0;col<N;++col){double ref=0,l1=0;
    if(dense){for(size_t k=0;k<K;++k){double product=double(x[row*K+k])*f32(uint32_t(raw[col*K+k])<<16);ref+=product;l1+=std::abs(product);}}
    else for(size_t j=0;j<33;++j){size_t k=(j*131+row*17+pair*43)%K;double product=double(x[row*K+k])*f32(uint32_t(raw[col*K+k])<<16);ref+=product;l1+=std::abs(product);}
    double bound=1e-4+2e-6*l1,error=std::abs(double(actual[row*N+col])-ref);need(std::isfinite(actual[row*N+col])&&error<=bound,"Original FP64 bound failed");worst=std::max(worst,error/bound);
   }
   uint64_t shape[6]={rows,K,N,pass,projection,pair};binary.write(reinterpret_cast<char*>(shape),sizeof(shape));binary.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);++records;
  }
 }
 if(enabled){
  rejects([&]{gpu.EncodeSoleDualProjectionDecode(input,wg,wu,gout,uout);},"Missing batch accepted");
  auto untouched=gpu.Download(gout);auto before=MetalContext::DualProjectionStats();gpu.Begin();rejects([&]{gpu.EncodeSoleDualProjectionDecode(input,wg,wu,gout,gout);},"Output alias accepted");gpu.Abort();need(gpu.Download(gout)==untouched&&MetalContext::DualProjectionStats().encodedPairs==before.encodedPairs,"Alias mutated output/counter");
  auto uninitialized=gpu.Allocate(rows,K);gpu.Begin();rejects([&]{gpu.EncodeSoleDualProjectionDecode(uninitialized,wg,wu,gout,uout);},"Uninitialized input accepted");gpu.Abort();
  {MetalContext peer;auto pending=peer.Allocate(rows,K);peer.Begin();peer.EncodeAdd(input,input,pending);gpu.Begin();rejects([&]{gpu.EncodeSoleDualProjectionDecode(pending,wg,wu,gout,uout);},"Foreign pending input accepted");gpu.Abort();peer.Abort();}
  {MetalContext peer;auto pi=peer.Upload(std::vector<float>(rows*K,1.f),rows,K);peer.Begin();peer.EncodeBF16MatVec(pi,wg,gout);gpu.Begin();rejects([&]{gpu.EncodeSoleDualProjectionDecode(input,wg,wu,gout,uout);},"Foreign pending output accepted");gpu.Abort();peer.Abort();}
 }
 auto stableInput=std::vector<float>(rows*K,1.f);gpu.Update(input,stableInput);gpu.Begin();execute(gpu,input,wg,wu,gout,uout);gpu.Flush();auto otherG=gpu.Allocate(rows,N),otherU=gpu.Allocate(rows,N);execute(gpu,input,wg,wu,otherG,otherU);gpu.SubmitAndWait();auto referenceG=gpu.Download(gout),referenceU=gpu.Download(uout);need(referenceG==gpu.Download(otherG)&&referenceU==gpu.Download(otherU),"Partial Flush pair differs");
 gpu.Begin();execute(gpu,input,wg,wu,gout,uout);gpu.Flush();execute(gpu,input,wg,wu,otherG,otherU);gpu.Abort();rejects([&]{gpu.Download(gout);},"Aborted gate readable");rejects([&]{gpu.Download(uout);},"Aborted up readable");rejects([&]{gpu.Download(otherG);},"Aborted later gate readable");
 gpu.Begin();execute(gpu,input,wg,wu,gout,uout);gpu.SubmitAndWait();need(gpu.Download(gout)==referenceG&&gpu.Download(uout)==referenceU,"Abort recovery differs");
 {auto localG=gpu.UploadBF16(rawG.data(),rawG.size()*2,N,K,false,false,false),localU=gpu.UploadBF16(rawU.data(),rawU.size()*2,N,K,false,false,false);gpu.Begin();execute(gpu,input,localG,localU,gout,uout);localG=MetalBF16Weight();localU=MetalBF16Weight();gpu.SubmitAndWait();need(gpu.Download(gout)==referenceG&&gpu.Download(uout)==referenceU,"Released weight owners differ");}
 {MetalContext peer;auto pi=peer.Upload(stableInput,rows,K),pg=peer.Allocate(rows,N),pu=peer.Allocate(rows,N);peer.Begin();execute(peer,pi,wg,wu,pg,pu);peer.SubmitAndWait();need(peer.Download(pg)==referenceG&&peer.Download(pu)==referenceU,"Peer shared owner differs");}
#if defined(GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
 auto before=MetalContext::DualProjectionStats();MetalContext::DualProjectionTestUnavailable(true);gpu.Begin();need(!gpu.EncodeSoleDualProjectionDecode(input,wg,wu,gout,uout),"Unavailable pair admitted");gpu.EncodeBF16MatVec(input,wg,gout);gpu.EncodeBF16MatVec(input,wu,uout);gpu.SubmitAndWait();MetalContext::DualProjectionTestUnavailable(false);need(MetalContext::DualProjectionStats().encodedPairs==before.encodedPairs&&gpu.Download(gout)==referenceG&&gpu.Download(uout)==referenceU,"Unavailable fallback mutated route/output");
 MetalContext::DualProjectionTestFailNext(true);gpu.Begin();rejects([&]{gpu.EncodeSoleDualProjectionDecode(input,wg,wu,gout,uout);},"Predispatch fault not raised");gpu.Abort();need(MetalContext::DualProjectionStats().encodedPairs==before.encodedPairs,"Fault changed counters");gpu.Begin();execute(gpu,input,wg,wu,gout,uout);gpu.SubmitAndWait();need(gpu.Download(gout)==referenceG&&gpu.Download(uout)==referenceU,"Fault recovery differs");
#endif
 }
 auto three=gpu.Upload(std::vector<float>(3*K,1.f),3,K),og=gpu.Allocate(3,N),ou=gpu.Allocate(3,N);gpu.Begin();auto before=MetalContext::DualProjectionStats();need(!gpu.EncodeSoleDualProjectionDecode(three,wg,wu,og,ou),"Prefill pair admitted");gpu.Abort();need(MetalContext::DualProjectionStats().encodedPairs==before.encodedPairs,"Unsupported pair changed counters");rejects([&]{gpu.Download(og);},"Unsupported output initialized");
 auto bypass=gpu.UploadBF16(rawG.data(),rawG.size()*2,N,K,true,false,false);auto one=gpu.Upload(std::vector<float>(K,1.f),1,K),o1=gpu.Allocate(1,N),o2=gpu.Allocate(1,N);gpu.Begin();need(!gpu.EncodeSoleDualProjectionDecode(one,bypass,wu,o1,o2),"Row-major auxiliary pair admitted");gpu.Abort();
 }
 need(records==36,"Missing complete pair records");need(MetalContext::SoleStats().scratchBytes==0,"Pair created restore scratch");std::cout<<"dual_enabled="<<enabled<<" complete_records="<<records<<" worst_original_FP64_bound_fraction="<<worst<<"\n";
 }
 auto final=MetalContext::SoleStats();need(final.activeWeights==0&&final.allocatedBytes==0&&final.scratchBytes==0,"Pair weight lifetime leak");auto dual=MetalContext::DualProjectionStats();need(dual.representedProjections==2*dual.encodedPairs&&bool(dual.encodedPairs)==enabled,"Final pair counters differ");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
