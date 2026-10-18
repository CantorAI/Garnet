#include "metal_context.h"
#include "nlohmann/json.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static auto live(){auto v=MetalContext::PrimaryStats();return std::array<uint64_t,5>{v.activeWeights,v.originalBytes,v.payloadBytes,v.metadataBytes,v.allocatedBytes};}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: owned_storage_local_test words.bin");MetalContext gpu;
 std::ofstream stream(argv[1],std::ios::binary);need(bool(stream),"Cannot open complete lifetime words");
 const auto baseline=live();size_t records=0,totalWords=0,rawCases=0,packedCases=0,rejections=0,preparedCases=0;
 auto run=[&](size_t rows,size_t columns,bool forceRaw,size_t mode){
  constexpr size_t inner=128;std::vector<uint16_t> bits(columns*inner,0x3f80);
  std::vector<float> input(rows*inner);for(size_t row=0;row<rows;++row)for(size_t k=0;k<inner;++k)input[row*inner+k]=float(row+1)/16;
  auto a=gpu.Upload(input,rows,inner),out=gpu.Allocate(rows,columns);
  gpu.Update(out,std::vector<float>(rows*columns,std::numeric_limits<float>::quiet_NaN()));
  auto make=[&](){return gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,!forceRaw);};
  MetalBF16Weight original;
  if(mode==2){MetalContext creator;original=creator.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,!forceRaw);}
  else original=make();
  auto alias=original;original=MetalBF16Weight();need(alias.Valid(),"Copy lost storage owner");
  bool packed=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
  packed=!forceRaw;
#endif
  need(alias.PrimaryReady()==packed,"Lifetime case representation differs");
  if(packed)++packedCases;else ++rawCases;
  const bool expectedPrepare=rows<=2&&columns<100000;
  need(gpu.PrepareBF16BodyShape(a,alias,out)==expectedPrepare,"Alias/creator-close prepare changed");preparedCases+=expectedPrepare;
  if(mode==3){
   auto badout=gpu.Allocate(rows,columns+1);need(!gpu.PrepareBF16BodyShape(a,alias,badout),"Invalid output prepared");
   gpu.Begin();bool caught=false;try{gpu.PrepareBF16BodyShape(a,alias,out);}catch(const std::exception&){caught=true;}
   need(caught,"Active batch prepare did not reject");++rejections;caught=false;
   try{gpu.EncodeBF16MatVec(a,alias,badout);}catch(const std::exception&){caught=true;}
   need(caught,"Invalid shape encoding accepted");++rejections;alias=MetalBF16Weight();gpu.Abort();
   need(live()==baseline,"Exception/Abort retained weight owner");alias=make();
  }
  gpu.Begin();gpu.EncodeBF16MatVec(a,alias,out);
  if(mode==1||mode==4)gpu.Flush();
  alias=MetalBF16Weight();need(live()==baseline,"Last host weight owner was not released");
  if(mode==4){
   gpu.Abort();bool caught=false;try{gpu.Download(out);}catch(const std::exception&){caught=true;}
   need(caught,"Flushed aborted output readable");++rejections;
   alias=make();gpu.Begin();gpu.EncodeBF16MatVec(a,alias,out);alias=MetalBF16Weight();
  }
  gpu.SubmitAndWait();auto actual=gpu.Download(out);need(actual.size()==rows*columns,"Lifetime output truncated");
  for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col)need(actual[row*columns+col]==float(inner*(row+1))/16,"Queued released-owner FP32 word changed");
  need(gpu.Download(a)==input,"Input mutated");need(live()==baseline,"Completed case retained logical weight");
  const uint64_t header[]={rows,inner,columns,uint64_t(forceRaw),mode,actual.size()};
  stream.write(reinterpret_cast<const char*>(header),sizeof(header));stream.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);
  need(bool(stream),"Cannot write complete lifetime output");++records;totalWords+=actual.size();
 };
 for(bool raw:{false,true})for(size_t mode=0;mode<5;++mode){
  for(size_t rows:{1,2,16,32,64})for(size_t columns:{260,1024})run(rows,columns,raw,mode);
  for(size_t rows:{1,2})run(rows,100352,raw,mode);
 }
 need(records==120&&rawCases+packedCases==120&&rejections==72&&preparedCases==40,"Lifetime coverage incomplete");
 stream.flush();need(bool(stream)&&live()==baseline,"Final output/accounting invalid");
#ifdef GARNET_METAL_OWNED_STORAGE_LOCAL
 constexpr bool enabled=true;
#else
 constexpr bool enabled=false;
#endif
 json report={{"owned_storage_local_enabled",enabled},{"complete_records",records},{"complete_FP32_words",totalWords},{"raw_cases",rawCases},{"packed_cases",packedCases},{"invalid_cases_rejected",rejections},{"prepared_cases",preparedCases},{"live_logical_accounting_restored",true},{"original_alias_release_before_prepare",true},{"last_owner_release_before_commit",true},{"last_owner_release_after_flush",true},{"creator_context_closed_before_use",true},{"flushed_abort_owner_release_and_recovery",true},{"no_retainCount_or_timing_claim",true}};
 std::ofstream summary(std::string(argv[1])+".summary.json");summary<<report.dump(2);need(bool(summary),"Cannot write lifetime summary");
 std::cout<<"Selected storage alias/queued/Flush/Abort/creator-close lifetime PASS "<<report.dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
