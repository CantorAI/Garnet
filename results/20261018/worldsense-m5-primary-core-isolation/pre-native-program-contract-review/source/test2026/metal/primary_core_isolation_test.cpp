#include "metal_context.h"
#include "primary_block_shader.h"
#include "primary_isolation_shader.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: primary_core_isolation_test words.bin");MetalContext gpu;auto programs=gpu.ShaderProgramSources();need(programs.size()==2,"Original two-program dump contract changed");
#ifdef GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY
 constexpr bool core=true;need(programs[1]==std::string(kPrimaryIsolatedPreamble)+kPrimaryBlockShader,"Ordinary core program differs from exact original fragment/preamble");need(programs[0].find("kernel void primary_bf16_decode(")==std::string::npos,"Primary still in raw MAIN");
#else
 constexpr bool core=false;
#endif
 auto initial=MetalContext::PrimaryStats();std::ofstream words(argv[1],std::ios::binary);need(bool(words),"Cannot open complete native stream");size_t records=0,total=0,specialized=0;
 auto run=[&](size_t rows,size_t inner,size_t columns,bool forceRaw){
  auto beforeOwner=MetalContext::PrimaryStats();std::vector<uint16_t> original(columns*inner,0x3f80);auto weight=gpu.UploadBF16(original.data(),original.size()*2,columns,inner,false,false,!forceRaw);
  std::vector<float> input(rows*inner);for(size_t row=0;row<rows;++row)for(size_t k=0;k<inner;++k)input[row*inner+k]=float(row+1)/16.0f;
  auto a=gpu.Upload(input,rows,inner);auto output=gpu.Allocate(rows,columns);const bool prepared=gpu.PrepareBF16BodyShape(a,weight,output);auto before=MetalContext::PrimaryStats();
  gpu.Begin();gpu.EncodeBF16MatVec(a,weight,output);gpu.SubmitAndWait();auto values=gpu.Download(output);need(values.size()==rows*columns,"Incomplete native output");
  for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col)need(values[row*columns+col]==float(inner)*float(row+1)/16.0f,"Original exact binary-fraction reference differs");
  auto after=MetalContext::PrimaryStats();uint64_t d=after.decodeEncoded-before.decodeEncoded,p=after.prefillEncoded-before.prefillEncoded,h=after.headEncoded-before.headEncoded;
  bool packed=weight.PrimaryReady();need(d==uint64_t(packed&&rows<=2&&columns<100000)&&p==uint64_t(packed&&rows>2)&&h==uint64_t(packed&&rows<=2&&columns>=100000),"Original actual primary dispatch counters changed");
  need(after.coreLibraryDecodeEncoded-before.coreLibraryDecodeEncoded==(core?d:0)&&after.coreLibraryPrefillEncoded-before.coreLibraryPrefillEncoded==(core?p:0)&&after.coreLibraryHeadEncoded-before.coreLibraryHeadEncoded==(core?h:0),"Actual core dispatch deltas do not match selected packed route");
  if(rows<=2&&columns<100000){need(prepared&&gpu.LastBodyShapeDispatches()==1,"Specialized lookup did not resolve correct library");++specialized;}
  uint64_t header[]={rows,inner,columns,uint64_t(forceRaw),uint64_t(prepared),values.size()};words.write(reinterpret_cast<const char*>(header),sizeof(header));words.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));need(bool(words),"Cannot write complete reference stream");++records;total+=values.size();
  weight=MetalBF16Weight();auto ended=MetalContext::PrimaryStats();need(ended.activeWeights==beforeOwner.activeWeights&&ended.allocatedBytes==beforeOwner.allocatedBytes,"Native weight owner leak");
 };
 for(size_t inner:{128ul,256ul})for(size_t columns:{260ul,1024ul})for(size_t rows:{1ul,2ul,16ul,32ul})run(rows,inner,columns,false);
 for(size_t rows:{1ul,2ul})run(rows,128,100352,false);
 for(size_t rows:{1ul,2ul})run(rows,128,260,true);
 words.close();need(records==20&&total==432804&&specialized==10,"Incomplete core native route matrix");auto final=MetalContext::PrimaryStats();need(final.activeWeights==initial.activeWeights&&final.allocatedBytes==initial.allocatedBytes,"Final native weight accounting changed");
 if(core)need(final.coreLibraryDecodeEncoded-initial.coreLibraryDecodeEncoded==8&&final.coreLibraryPrefillEncoded-initial.coreLibraryPrefillEncoded==8&&final.coreLibraryHeadEncoded-initial.coreLibraryHeadEncoded==2,"Complete core encoded route counts differ");
 else need(final.coreLibraryDecodeEncoded==0&&final.coreLibraryPrefillEncoded==0&&final.coreLibraryHeadEncoded==0,"Disabled/compatibility route used core");
 json report={{"core_enabled",core},{"complete_records",records},{"complete_FP32_words",total},{"specialized_body_cases",specialized},{"core_decode_delta",final.coreLibraryDecodeEncoded-initial.coreLibraryDecodeEncoded},{"core_prefill_delta",final.coreLibraryPrefillEncoded-initial.coreLibraryPrefillEncoded},{"core_head_delta",final.coreLibraryHeadEncoded-initial.coreLibraryHeadEncoded},{"original_two_program_contract",true},{"owner_accounting_restored",true},{"no_speed_claim",true}};std::ofstream summary(std::string(argv[1])+".summary.json");summary<<report.dump(2);need(bool(summary),"Cannot write native summary");std::cout<<"Ordinary primary core partition/actual generic+specialized+head+prefill/raw/tail20complete432804words exact PASS "<<report.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
