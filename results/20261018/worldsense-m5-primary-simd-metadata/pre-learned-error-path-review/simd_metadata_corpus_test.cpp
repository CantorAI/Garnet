// Correctness only: complete learned original BF16 words through both serving helpers.
#include "metal_context.h"
#include "primary_32_contract.h"
#include "wide_checkpoint_policy.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char* e){if(!v)throw std::runtime_error(e);}
static void save(const char* p,const json& j){auto t=std::string(p)+".tmp";{std::ofstream f(t);f<<j.dump(2);need(bool(f),"Cannot save new metadata corpus proof");}std::filesystem::rename(t,p);}
int main(int argc,char** argv){try{
 need(argc==3,"Usage: simd_metadata_corpus selection proof");std::ifstream input(argv[1]);json selection;input>>selection;MetalContext gpu;std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;
 json report={{"scope","Original learned checkpoint words through both exact new serving helpers; no inference or speed claim."},{"matrices",json::array()}};uint64_t matrices=0,decodeWords=0,prefillWords=0,skipped=0;save(argv[2],report);
 for(const auto& row:selection.at("matrices")){
  const auto model=row.at("model").get<std::string>(),checkpoint=row.at("checkpoint").get<std::string>(),name=row.at("tensor").get<std::string>();
  if(!indexes.count(model)){auto index=std::make_unique<SafeTensorsIndex>();std::string error;need(index->Open(checkpoint,error),error.c_str());indexes.emplace(model,std::move(index));}
  const auto* entry=indexes.at(model)->Find(name);need(entry&&entry->dataType=="BF16"&&entry->shape.size()==2,"Missing original metadata-corpus matrix");size_t rows=row.at("rows"),inner=row.at("inner");need(entry->shape[0]==rows&&entry->shape[1]==inner&&entry->dataSize==rows*inner*2,"Metadata-corpus shape differs from frozen selection");
  bool eligible=WideStorage::Eligible(*indexes.at(model))&&Primary32::ShapeFits(rows,inner,entry->dataSize);if(!eligible){++skipped;continue;}
  need(model=="Qwen3.5-4B","Narrow checkpoint admitted to new primary corpus");auto time=std::filesystem::last_write_time(entry->filePath);auto size=std::filesystem::file_size(entry->filePath);SafeTensorsMappedFile mapped;std::string error;need(mapped.Open(entry->filePath,error),error.c_str());auto bits=static_cast<const unsigned char*>(mapped.DataAt(entry->dataOffset,entry->dataSize));need(bits,"Cannot map original words");auto before=MetalContext::PrimaryStats();auto weight=gpu.UploadBF16(bits,entry->dataSize,rows,inner,false,false,true);need(weight.PrimaryReady(),"Selected learned primary matrix fell back");auto info=MetalContext::WeightBufferInfo(weight);need(info.size()==2,"New metadata method retained a raw duplicate");auto loaded=MetalContext::PrimaryStats();uint64_t local[2]={0,0};
  for(size_t start=0;start<rows*inner;start+=8*1024*1024){size_t count=std::min(size_t(8*1024*1024),rows*inner-start);
   for(unsigned method=0;method<2;++method){auto actual=gpu.PrimaryReadSIMDMetadataWords(weight,start,count,bool(method));need(actual.size()==count,"Learned new-helper word range changed");for(size_t i=0;i<count;++i){size_t at=(start+i)*2;uint32_t word=uint32_t(bits[at])|(uint32_t(bits[at+1])<<8);need(actual[i]==word<<16,"Learned metadata SIMD helper changed original checkpoint word");}local[method]+=count;}
  }
  auto after=MetalContext::PrimaryStats();need(loaded.activeWeights==after.activeWeights&&loaded.allocatedBytes==after.allocatedBytes&&loaded.decodeEncoded==after.decodeEncoded&&loaded.prefillEncoded==after.prefillEncoded&&loaded.headEncoded==after.headEncoded,"New helper reader changed serving counters or owners");weight=MetalBF16Weight();auto released=MetalContext::PrimaryStats();need(released.activeWeights==before.activeWeights&&released.allocatedBytes==before.allocatedBytes,"New helper corpus leaked primary owner");need(time==std::filesystem::last_write_time(entry->filePath)&&size==std::filesystem::file_size(entry->filePath),"Checkpoint changed during new-helper proof");need(local[0]==rows*inner&&local[1]==rows*inner,"Incomplete learned helper coverage");
  ++matrices;decodeWords+=local[0];prefillWords+=local[1];report["matrices"].push_back({{"model",model},{"tensor",name},{"rows",rows},{"inner",inner},{"decode_helper_original_FP32_words_checked",local[0]},{"prefill_helper_original_FP32_words_checked",local[1]},{"owner_accounting_restored",true},{"serving_counters_unchanged",true}});report["totals"]={{"matrices",matrices},{"skipped_raw",skipped},{"decode_helper_original_FP32_words_checked",decodeWords},{"prefill_helper_original_FP32_words_checked",prefillWords}};save(argv[2],report);std::cout<<matrices<<" "<<name<<" words="<<local[0]<<std::endl;
 }
 need(matrices==201&&skipped==245&&decodeWords==4200857600ULL&&prefillWords==4200857600ULL,"Full201/245 learned helper corpus missing");report["totals"]["skipped_raw"]=skipped;report["status"]="both_exact_serving_helpers_all201_original_words_exact";save(argv[2],report);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
