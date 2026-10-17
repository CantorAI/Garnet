#include "metal_context.h"
#include "primary_storage_policy.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <filesystem>
#include <cstring>
using json=nlohmann::json;using namespace Garnet;
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void save(const char* path,const json& report){auto temp=std::string(path)+".tmp";{std::ofstream f(temp);f<<report.dump(2);need(bool(f),"Cannot save corpus proof");}std::filesystem::rename(temp,path);}
int main(int argc,char**argv){try{
 need(argc==3,"Usage: corpus selection.json proof.json");std::ifstream input(argv[1]);json selection;input>>selection;MetalContext gpu;
 json report={{"scope","GPU bit correctness only, no inference/speed claim. All admitted actual weights read back in bounded chunks; raw fallback includes the single explicitly excluded vocabulary owner; no primary-reader bit claim for raw buffers."},{"matrices",json::array()}};save(argv[2],report);
 std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;uint64_t wordsChecked=0,packedMatrices=0,rawMatrices=0,rawBytes=0,primaryBytes=0,rawVocabulary=0;
 for(const auto& item:selection.at("matrices")){
  const auto model=item.at("model").get<std::string>(),checkpoint=item.at("checkpoint").get<std::string>(),name=item.at("tensor").get<std::string>();
  if(!indexes.count(model)){auto p=std::make_unique<SafeTensorsIndex>();std::string e;if(!p->Open(checkpoint,e))throw std::runtime_error(e);indexes.emplace(model,std::move(p));}
  const auto* entry=indexes.at(model)->Find(name);need(entry&&entry->dataType=="BF16"&&entry->shape.size()==2,"Missing selected BF16 matrix");
  const size_t rows=item.at("rows"),inner=item.at("inner");need(entry->shape[0]==rows&&entry->shape[1]==inner&&entry->dataSize==rows*inner*2,"Matrix selection mismatch");
  auto modified=std::filesystem::last_write_time(entry->filePath);auto size=std::filesystem::file_size(entry->filePath);
  SafeTensorsMappedFile file;std::string e;if(!file.Open(entry->filePath,e))throw std::runtime_error(e);auto data=static_cast<const unsigned char*>(file.DataAt(entry->dataOffset,entry->dataSize));need(data,"Mapped data missing");
  const auto before=MetalContext::PrimaryStats();bool allow=PrimaryPolicy::AllowPrimary(name,rows,inner,true);auto w=gpu.UploadBF16(data,entry->dataSize,rows,inner,false,false,allow);bool expected=allow&&rows>256&&rows%4==0&&inner%128==0;if(!allow){need(model=="Qwen3-1.7B"&&name=="model.embed_tokens.weight"&&rows==151936&&inner==2048,"Unexpected semantic exclusion");++rawVocabulary;}need(w.PrimaryReady()==expected,"Actual checkpoint admission differs from plan");
  uint64_t checked=0;
  if(expected){
   auto info=MetalContext::WeightBufferInfo(w);need(info.size()==2,"Packed checkpoint weight has extra original buffer");uint64_t bytes=0;for(const auto& b:info)bytes+=b.lengthBytes;primaryBytes+=bytes;
   for(size_t start=0;start<rows*inner;start+=8*1024*1024){size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto actual=gpu.PrimaryReadWords(w,start,count);need(actual.size()==count,"GPU read size changed");auto vectorActual=gpu.PrimaryReadVectorWords(w,start,count);need(actual==vectorActual,"Actual checkpoint scalar/vector GPU words differ");for(size_t i=0;i<count;++i){size_t at=(start+i)*2;uint32_t original=uint32_t(data[at])|(uint32_t(data[at+1])<<8);need(actual[i]==(original<<16),"Actual checkpoint GPU FP32 bit reconstruction failed");}checked+=count;}
   ++packedMatrices;wordsChecked+=checked;
  }else{auto info=MetalContext::WeightBufferInfo(w);need(info.size()==1&&info[0].representation=="bf16"&&info[0].lengthBytes==entry->dataSize,"Raw fallback changed");++rawMatrices;primaryBytes+=entry->dataSize;}
  rawBytes+=entry->dataSize;w=MetalBF16Weight();auto after=MetalContext::PrimaryStats();need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Corpus weight owner leak");
  need(modified==std::filesystem::last_write_time(entry->filePath)&&size==std::filesystem::file_size(entry->filePath),"Checkpoint changed during GPU check");
  report["matrices"].push_back({{"semantic_primary_allowed",allow},{"model",model},{"tensor",name},{"rows",rows},{"inner",inner},{"original_bytes",entry->dataSize},{"packed",expected},{"GPU_original_FP32_words_checked",checked},{"GPU_vector_original_FP32_words_checked",checked},{"owner_accounting_restored",true}});
  report["totals"]={{"packed_matrices",packedMatrices},{"raw_fallback_matrices",rawMatrices},{"raw_vocabulary_owners",rawVocabulary},{"original_bytes",rawBytes},{"retained_primary_and_small_raw_bytes",primaryBytes},{"GPU_original_FP32_words_checked",wordsChecked},{"GPU_vector_original_FP32_words_checked",wordsChecked}};save(argv[2],report);
  std::cout<<report["matrices"].size()<<" "<<model<<" "<<name<<" words="<<checked<<std::endl;
 }
 need(rawVocabulary==1&&packedMatrices==397&&rawMatrices==49&&rawBytes==11850481664ULL&&wordsChecked==5610143744ULL&&primaryBytes==9182203392ULL,"Corpus coverage or retained-byte proof mismatch");
 report["status"]="full_actual_both_readers_packed_corpus_GPU_bits_exact_not_inference_validated";save(argv[2],report);std::cout<<"Full397packed/49raw actual checkpoint GPU word and owner proof PASS"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
