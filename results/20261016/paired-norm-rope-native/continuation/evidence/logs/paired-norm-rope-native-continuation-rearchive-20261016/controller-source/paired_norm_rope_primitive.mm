// Synthetic FP32 bit identity and ownership prerequisites; not learned/model performance.
#import <Foundation/Foundation.h>
#include "metal_context.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static std::vector<float> values(size_t n,uint32_t seed){std::vector<float> result(n);for(auto& v:result){seed=seed*1664525u+1013904223u;v=float(int((seed>>8)%65521)-32760)/8192.f;}return result;}
int main(){@autoreleasepool{try{
 Garnet::MetalContext gpu,other;size_t cases=0,checked=0;
 struct Shape{size_t rows,q,k,dim,rotary;bool normalize;float offset;};
 std::array<Shape,15> shapes{{{1,16,8,128,128,true,0},{2,16,8,128,128,true,0},{24,16,8,128,128,true,0},{44,16,8,128,128,true,0},{1,14,2,64,64,false,0},{2,14,2,64,64,false,0},{26,14,2,64,64,false,0},{52,14,2,64,64,false,0},{1,8,2,256,64,true,1},{2,8,2,256,64,true,1},{80,8,2,256,64,true,1},{160,8,2,256,64,true,1},{3,5,3,72,48,true,.25f},{7,3,1,32,16,false,-.25f},{5,2,2,96,64,true,-.25f}}};
 for(auto s:shapes){
  auto q=gpu.Upload(values(s.rows*s.q*s.dim,11),s.rows,s.q*s.dim),k=gpu.Upload(values(s.rows*s.k*s.dim,73),s.rows,s.k*s.dim);
  auto qw=gpu.Upload(values(s.dim,971),1,s.dim),kw=gpu.Upload(values(s.dim,1231),1,s.dim);
  std::vector<float> positions(s.rows*3),frequencies(s.rotary);for(size_t row=0;row<s.rows;++row)for(size_t axis=0;axis<3;++axis)positions[row*3+axis]=float((row*17+axis*11)%193)-51.5f;
  for(size_t i=0;i<s.rotary/2;++i){frequencies[i]=float(std::pow(1000000.,-2.*double(i)/double(s.rotary)));frequencies[s.rotary/2+i]=float(i%3);}
  auto p=gpu.Upload(positions,s.rows,3),f=gpu.Upload(frequencies,2,s.rotary/2),qo=gpu.Allocate(s.rows,s.q*s.dim),ko=gpu.Allocate(s.rows,s.k*s.dim);
  for(uint32_t variant=0;variant<4;++variant){
   gpu.Update(q,values(s.rows*s.q*s.dim,11+variant*997));gpu.Update(k,values(s.rows*s.k*s.dim,73+variant*991));
   gpu.Begin();gpu.EncodeNormRope(q,qw,p,f,qo,s.q,s.dim,s.rotary,1e-6f,s.offset,s.normalize);gpu.EncodeNormRope(k,kw,p,f,ko,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize);gpu.SubmitAndWait();std::array<std::vector<float>,2> expected{gpu.Download(qo),gpu.Download(ko)};
   gpu.Begin();need(gpu.LastPairedNormRopeDispatches()==0,"Counter not reset");need(gpu.EncodePairedNormRope(q,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize)&&gpu.LastPairedNormRopeDispatches()==1,"Paired encode failed");
   for(auto output:{qo,ko}){bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending paired output readable");}gpu.SubmitAndWait();
   std::array<std::vector<float>,2> actual{gpu.Download(qo),gpu.Download(ko)};for(size_t side=0;side<2;++side){need(actual[side].size()==expected[side].size()&&!std::memcmp(actual[side].data(),expected[side].data(),actual[side].size()*sizeof(float)),"Original norm-RoPE bits differ");for(float value:actual[side])need(std::isfinite(value),"Nonfinite output");checked+=actual[side].size();}++cases;
  }
  gpu.Begin();need(gpu.EncodePairedNormRope(q,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize),"Abort precondition failed");gpu.Abort();need(gpu.LastPairedNormRopeDispatches()==0,"Abort retained counter");
  for(auto output:{qo,ko}){bool rejected=false;try{gpu.Download(output);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted output readable");}
  gpu.Begin();need(gpu.EncodePairedNormRope(q,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize),"Recovery failed");gpu.SubmitAndWait();auto preserved=gpu.Download(qo);
  gpu.Begin();need(!gpu.EncodePairedNormRope(q,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary-1,1e-6f,s.offset,s.normalize)&&gpu.LastPairedNormRopeDispatches()==0,"Odd rotary fallback encoded");gpu.Abort();need(gpu.Download(qo)==preserved,"Fallback changed output");
  gpu.Begin();bool rejected=false;try{gpu.EncodePairedNormRope(q,k,qw,kw,p,f,q,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize);}catch(const std::invalid_argument&){rejected=true;}need(rejected&&gpu.LastPairedNormRopeDispatches()==0,"Input/output alias accepted");gpu.Abort();
  auto missing=gpu.Allocate(s.rows,s.q*s.dim);gpu.Begin();rejected=false;try{gpu.EncodePairedNormRope(missing,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize);}catch(const std::logic_error&){rejected=true;}need(rejected,"Uninitialized query accepted");gpu.Abort();
  gpu.Begin();need(gpu.EncodePairedNormRope(q,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize),"Foreign writer precondition failed");other.Begin();rejected=false;try{other.EncodePairedNormRope(q,k,qw,kw,p,f,qo,ko,s.q,s.k,s.dim,s.rotary,1e-6f,s.offset,s.normalize);}catch(const std::logic_error&){rejected=true;}need(rejected&&other.LastPairedNormRopeDispatches()==0,"Foreign writer accepted");other.Abort();gpu.SubmitAndWait();
 }
 need(cases==60,"Coverage differs");printf("Paired norm-RoPE60/original-bits/pending/foreign/alias/abort-recovery/fallback PASS checked_values=%zu\n",checked);return 0;
}catch(const std::exception& error){fprintf(stderr,"paired_norm_rope_primitive_failed: %s\n",error.what());return 1;}}}
