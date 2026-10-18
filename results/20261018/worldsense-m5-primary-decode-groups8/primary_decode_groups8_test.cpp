// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "primary_decode_groups8_shader.h"
#include "primary_isolation_shader.h"
#include "primary_vector_prefill_shader.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
using namespace Garnet;using json=nlohmann::json;
static void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: primary_decode_groups8_test words.bin");MetalContext gpu;
#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS8
 constexpr bool enabled=true;
#else
 constexpr bool enabled=false;
#endif
 const auto programs=gpu.ShaderProgramSources();need(programs.size()==2&&gpu.PrimaryCoreProgramSource().empty(),"Original program contract or disabled CORE getter changed");
 auto transformed=PrimaryDecodeGroups8Program();const std::string from="ulong column=ulong(group)*32+simd*4;",to="ulong column=ulong(group)*16+simd*4;";size_t pos=0,count=0;
 while((pos=transformed.find(from,pos))!=std::string::npos){transformed.replace(pos,from.size(),to);pos+=to.size();++count;}
 need(count==2&&transformed==kPrimaryBlockShader,"Selected packed shader inverse changed non-origin bytes");
 if(enabled){need(programs[0].find("kernel void primary_bf16_decode(")==std::string::npos,"Packed decode remains in raw MAIN");need(programs[1]==std::string(kPrimaryIsolatedPreamble)+PrimaryDecodeGroups8Program()+kPrimaryVectorPrefillShader,"Actual FULL secondary differs from exact selected fragment");}
 {std::ofstream f("groups8_selected_primary_program.metal",std::ios::binary);f<<(enabled?PrimaryDecodeGroups8Program():std::string(kPrimaryBlockShader));need(bool(f),"Cannot save selected shader fragment");}
 auto initial=MetalContext::PrimaryStats();size_t records=0,words=0,generic=0,specialized=0,prefill=0,head=0;std::ofstream stream(argv[1],std::ios::binary);need(bool(stream),"Cannot open complete groups8 stream");
 auto run=[&](size_t rows,size_t inner,size_t columns,bool raw,bool requestPrepare){
  const auto owners=MetalContext::PrimaryStats();std::vector<uint16_t> bits(columns*inner,0x3f80);auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,!raw);const bool packed=weight.PrimaryReady();
  std::vector<float> input(rows*inner);for(size_t row=0;row<rows;++row)for(size_t k=0;k<inner;++k)input[row*inner+k]=float(row+1)/16.0f;
  auto a=gpu.Upload(input,rows,inner),output=gpu.Allocate(rows,columns);const bool prepared=requestPrepare&&gpu.PrepareBF16BodyShape(a,weight,output);need(prepared==requestPrepare,"Requested specialized body route unavailable");
  const auto before=MetalContext::PrimaryStats();gpu.Begin();gpu.EncodeBF16MatVec(a,weight,output);gpu.SubmitAndWait();const auto actual=gpu.Download(output);need(actual.size()==rows*columns,"Incomplete groups8 output");
  for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col)need(actual[row*columns+col]==float(inner)*float(row+1)/16.0f,"Generic/specialized/tail/head exact reference differs");
  need(gpu.LastBodyShapeDispatches()==size_t(prepared),"Generic request reused a specialized pipeline or specialized route missed");
  if(columns<100000&&rows<=2){if(prepared)++specialized;else{need(gpu.LastBodyShapeMisses()==1,"Generic body did not exercise cache miss");++generic;}}
  if(rows>2)++prefill;if(columns>=100000)++head;
  const auto after=MetalContext::PrimaryStats();const auto d=after.decodeEncoded-before.decodeEncoded,h=after.headEncoded-before.headEncoded,p=after.prefillEncoded-before.prefillEncoded;
  need(d==uint64_t(packed&&rows<=2&&columns<100000)&&h==uint64_t(packed&&rows<=2&&columns>=100000)&&p==uint64_t(packed&&rows>2),"Original primary encode accounting differs");
  need(after.decodeGroups8Encoded-before.decodeGroups8Encoded==(enabled?d:0)&&after.headGroups8Encoded-before.headGroups8Encoded==(enabled?h:0),"Actual selected256thread32output route counters differ");
  need(after.coreLibraryDecodeEncoded==0&&after.coreLibraryPrefillEncoded==0&&after.coreLibraryHeadEncoded==0,"Disabled CORE path used");
  uint64_t header[]={rows,inner,columns,uint64_t(raw),uint64_t(requestPrepare),uint64_t(prepared),actual.size()};stream.write(reinterpret_cast<const char*>(header),sizeof(header));stream.write(reinterpret_cast<const char*>(actual.data()),actual.size()*sizeof(float));need(bool(stream),"Cannot save full native words");++records;words+=actual.size();
  weight=MetalBF16Weight();const auto ended=MetalContext::PrimaryStats();need(ended.activeWeights==owners.activeWeights&&ended.allocatedBytes==owners.allocatedBytes,"Native owner accounting not restored");
 };
 for(size_t inner:{128ul,256ul})for(size_t columns:{260ul,272ul,288ul,1024ul})for(size_t rows:{1ul,2ul,16ul,32ul}){run(rows,inner,columns,false,false);if(rows<=2)run(rows,inner,columns,false,true);}
 for(size_t columns:{100352ul,100368ul})for(size_t rows:{1ul,2ul})run(rows,128,columns,false,false);
 for(size_t columns:{260ul,272ul})for(size_t rows:{1ul,2ul}){run(rows,512,columns,true,false);run(rows,512,columns,true,true);}
 stream.close();const auto final=MetalContext::PrimaryStats();need(records==60&&words==804504&&generic==20&&specialized==20&&prefill==16&&head==4,"Incomplete native route/word matrix");
 need(final.decodeGroups8Encoded-initial.decodeGroups8Encoded==(enabled?32:0)&&final.headGroups8Encoded-initial.headGroups8Encoded==(enabled?4:0),"Complete groups8 actual route counts differ");need(final.activeWeights==initial.activeWeights&&final.allocatedBytes==initial.allocatedBytes,"Final primary owner accounting differs");
 json result={{"groups8_enabled",enabled},{"complete_records",records},{"complete_FP32_words",words},{"generic_body_cases",generic},{"specialized_body_cases",specialized},{"prefill_cases",prefill},{"head_cases",head},{"groups8_decode_delta",final.decodeGroups8Encoded-initial.decodeGroups8Encoded},{"groups8_head_delta",final.headGroups8Encoded-initial.headGroups8Encoded},{"selector_inverse_exact",true},{"actual_FULL_program_exact",true},{"owner_accounting_restored",true},{"no_speed_claim",true}};std::ofstream summary(std::string(argv[1])+".summary.json");summary<<result.dump(2);need(bool(summary),"Cannot save native summary");std::cout<<"Packed8SIMD60complete804504words/generic-specialized/head/prefill/raw/tails exact PASS "<<result.dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
