#include "metal_context.h"
#include "primary_isolation_shader.h"
#include "primary_block_shader.h"
#include "wide_checkpoint_policy.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <filesystem>
using json=nlohmann::json;using namespace Garnet;
static void need(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static void save(const char* path,const json& j){auto p=std::string(path)+".tmp";{std::ofstream f(p);f<<j.dump(2);need(bool(f),"Cannot save core corpus");}std::filesystem::rename(p,path);}
static bool sameCounters(const MetalPrimaryStats&a,const MetalPrimaryStats&b){return a.decodeEncoded==b.decodeEncoded&&a.prefillEncoded==b.prefillEncoded&&a.headEncoded==b.headEncoded&&a.vectorDecodeEncoded==b.vectorDecodeEncoded&&a.vectorHeadEncoded==b.vectorHeadEncoded&&a.coreLibraryDecodeEncoded==b.coreLibraryDecodeEncoded&&a.coreLibraryPrefillEncoded==b.coreLibraryPrefillEncoded&&a.coreLibraryHeadEncoded==b.coreLibraryHeadEncoded;}
int main(int argc,char**argv){try{
 need(argc==3,"Usage: core_library_corpus selection.json proof.json");std::ifstream f(argv[1]);json selection;f>>selection;MetalContext gpu;auto programs=gpu.ShaderProgramSources();need(programs.size()==2&&programs[1].empty(),"Original vector-prefill program contract changed");
 need(gpu.PrimaryCoreProgramSource()==std::string(kPrimaryIsolatedPreamble)+kPrimaryBlockShader,"Actual ordinary core does not contain original helpers");need(programs[0].find("kernel void primary_bf16_decode(")==std::string::npos,"Packed kernels remain in MAIN");
 json report={{"scope","Original full learned scalar/vector GPU readers in actual isolated ordinary core program. No new shader/FP/serving operation or speed claim."},{"matrices",json::array()}};
 std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;uint64_t matrices=0,skipped=0,checked=0;save(argv[2],report);
 for(const auto& item:selection.at("matrices")){
  auto model=item.at("model").get<std::string>(),checkpoint=item.at("checkpoint").get<std::string>(),name=item.at("tensor").get<std::string>();
  if(!indexes.count(model)){auto p=std::make_unique<SafeTensorsIndex>();std::string error;if(!p->Open(checkpoint,error))throw std::runtime_error(error);indexes.emplace(model,std::move(p));}
  const auto* entry=indexes.at(model)->Find(name);need(entry&&entry->dataType=="BF16"&&entry->shape.size()==2,"Missing selected BF16 matrix");size_t rows=item.at("rows"),inner=item.at("inner");need(entry->shape[0]==rows&&entry->shape[1]==inner&&entry->dataSize==rows*inner*2,"Matrix selection mismatch");
  bool allow=WideStorage::Eligible(*indexes.at(model)),expected=allow&&rows>256&&rows%4==0&&inner%128==0;if(!expected){++skipped;continue;}
  auto modified=std::filesystem::last_write_time(entry->filePath);auto size=std::filesystem::file_size(entry->filePath);SafeTensorsMappedFile file;std::string error;if(!file.Open(entry->filePath,error))throw std::runtime_error(error);auto data=static_cast<const unsigned char*>(file.DataAt(entry->dataOffset,entry->dataSize));need(data,"Missing mapped original bytes");
  auto beforeOwner=MetalContext::PrimaryStats();auto weight=gpu.UploadBF16(data,entry->dataSize,rows,inner,false,false,allow);need(weight.PrimaryReady(),"Actual core weight not packed");auto info=MetalContext::WeightBufferInfo(weight);need(info.size()==2,"Packed weight has extra/missing buffers");auto before=MetalContext::PrimaryStats();uint64_t words=0;
  for(size_t start=0;start<rows*inner;start+=8*1024*1024){
   size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto scalar=gpu.PrimaryReadWords(weight,start,count),vector=gpu.PrimaryReadVectorWords(weight,start,count);need(scalar.size()==count&&scalar==vector,"Actual core scalar/vector words differ");
   for(size_t i=0;i<count;++i){size_t at=(start+i)*2;uint32_t original=uint32_t(data[at])|(uint32_t(data[at+1])<<8);need(scalar[i]==(original<<16),"Actual core learned FP32 original word differs");}words+=count;
  }
  auto after=MetalContext::PrimaryStats();need(sameCounters(before,after),"Correctness readers changed original/core serving counters");weight=MetalBF16Weight();auto released=MetalContext::PrimaryStats();need(released.activeWeights==beforeOwner.activeWeights&&released.allocatedBytes==beforeOwner.allocatedBytes,"Core learned owner leak");need(modified==std::filesystem::last_write_time(entry->filePath)&&size==std::filesystem::file_size(entry->filePath),"Checkpoint changed during word proof");
  ++matrices;checked+=words;report["matrices"].push_back({{"model",model},{"tensor",name},{"rows",rows},{"inner",inner},{"scalar_original_FP32_words",words},{"vector_original_FP32_words",words},{"owner_accounting_restored",true},{"serving_counters_unchanged",true}});
  report["totals"]={{"matrices",matrices},{"skipped_raw",skipped},{"scalar_original_FP32_words",checked},{"vector_original_FP32_words",checked}};save(argv[2],report);std::cout<<matrices<<" "<<name<<" words="<<words<<std::endl;
 }
 need(matrices==201&&skipped==245&&checked==4200857600ULL,"Incomplete actual isolated-core learned word coverage");report["totals"]["skipped_raw"]=skipped;report["status"]="all201_two_original_readers_exact_in_actual_isolated_core";save(argv[2],report);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
