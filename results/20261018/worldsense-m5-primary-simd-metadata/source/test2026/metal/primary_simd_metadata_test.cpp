// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
using namespace Garnet;
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F>static void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}need(rejected,"Malformed or unavailable metadata readback accepted");}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: primary_simd_metadata_test output.bin");MetalContext gpu;std::ofstream file(argv[1],std::ios::binary);need(bool(file),"Cannot create complete metadata word records");
 auto initial=MetalContext::PrimaryStats();
#ifndef GARNET_METAL_PRIMARY_SIMD_METADATA
 rejects([&]{gpu.PrimaryReadSIMDMetadataWords(MetalBF16Weight(),0,1,false);});
#else
 for(size_t inner:std::vector<size_t>{128,256,512})for(size_t pattern=0;pattern<3;++pattern){
  const size_t rows=260,n=rows*inner;std::vector<uint16_t> bits(n);
  for(size_t i=0;i<n;++i){size_t row=i/inner,k=i%inner,block=(row/4)*(inner/128)+k/128;bits[i]=pattern==0?uint16_t(i):uint16_t(0x3f00u+((i*13)%128));if(pattern==2&&block%3==1)bits[i]=uint16_t(i*19);if((bits[i]&0x7f80u)==0x7f80u)bits[i]=0x8000;}
  {auto invalid=bits;invalid[0]=0x7f80;rejects([&]{gpu.UploadBF16(invalid.data(),invalid.size()*2,rows,inner,false,false,true);});}
  auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,rows,inner,false,false,true);need(weight.PrimaryReady(),"Metadata fixture not primary packed");auto alias=weight;weight=MetalBF16Weight();auto before=MetalContext::PrimaryStats();
  for(bool prefill:{false,true})for(auto range:std::vector<std::pair<size_t,size_t>>{{0,n},{1,1},{127,3},{inner-1,7},{3*inner+125,517},{n-1,1}}){
   auto actual=gpu.PrimaryReadSIMDMetadataWords(alias,range.first,range.second,prefill);need(actual.size()==range.second,"Metadata range length changed");for(size_t j=0;j<actual.size();++j)need(actual[j]==uint32_t(bits[range.first+j])<<16,"Metadata sharing changed original word");
   uint64_t header[6]={rows,inner,pattern,uint64_t(prefill),range.first,range.second};file.write(reinterpret_cast<const char*>(header),sizeof(header));file.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);
  }
  auto after=MetalContext::PrimaryStats();need(before.activeWeights==after.activeWeights&&before.allocatedBytes==after.allocatedBytes&&before.decodeEncoded==after.decodeEncoded&&before.prefillEncoded==after.prefillEncoded&&before.headEncoded==after.headEncoded,"Diagnostic altered primary owners or serving counters");
  rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,0,0,false);});rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,n,1,true);});rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,std::numeric_limits<size_t>::max(),1,false);});rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,0,8*1024*1024+1,true);});
  gpu.Begin();rejects([&]{gpu.PrimaryReadSIMDMetadataWords(alias,0,1,false);});gpu.Abort();auto recovered=gpu.PrimaryReadSIMDMetadataWords(alias,n-1,1,true);need(recovered[0]==uint32_t(bits.back())<<16,"Abort recovery metadata word changed");
  {MetalContext shared;auto words=shared.PrimaryReadSIMDMetadataWords(alias,0,1,false);need(words[0]==uint32_t(bits[0])<<16,"Same-device shared weight readback changed");}
  alias=MetalBF16Weight();auto released=MetalContext::PrimaryStats();need(released.activeWeights==initial.activeWeights&&released.allocatedBytes==initial.allocatedBytes,"Metadata fixture owner retained");
 }
 rejects([&]{gpu.PrimaryReadSIMDMetadataWords(MetalBF16Weight(),0,1,false);});
#endif
 need(bool(file),"Cannot write complete metadata records");std::cout<<"Original-bit metadata helpers, bounds, owner, Abort and disabled guards PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
