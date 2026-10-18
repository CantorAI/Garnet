#include "metal_context.h"
#include "wide_checkpoint_policy.h"
#include "primary_wide_vocab_policy.h"
#ifndef GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB
#error Supplemental candidate corpus requires raw-wide-vocabulary policy
#endif
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
 json report={{"scope","GPU bit correctness only, no inference/speed claim. All admitted actual weights read back in bounded chunks; all246rawbuffers separately HOST-word checked plus full newlyraw widevocabulary GPU words. Original201 compatibility is a separate mandatory subprocess."},{"matrices",json::array()}};save(argv[2],report);
 std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;uint64_t rawWordsChecked=0,wordsChecked=0,packedMatrices=0,rawMatrices=0,rawBytes=0,primaryBytes=0;
 uint64_t rawHeadGPUWords=0,rawHeadMatrices=0;
 std::map<std::string,uint64_t> allTileWords{{"32x32",0},{"64x32",0},{"64x64",0}};
 for(const auto& item:selection.at("matrices")){
  const auto model=item.at("model").get<std::string>(),checkpoint=item.at("checkpoint").get<std::string>(),name=item.at("tensor").get<std::string>();
  if(!indexes.count(model)){auto p=std::make_unique<SafeTensorsIndex>();std::string e;if(!p->Open(checkpoint,e))throw std::runtime_error(e);indexes.emplace(model,std::move(p));}
  const auto* entry=indexes.at(model)->Find(name);need(entry&&entry->dataType=="BF16"&&entry->shape.size()==2,"Missing selected BF16 matrix");
  const size_t rows=item.at("rows"),inner=item.at("inner");need(entry->shape[0]==rows&&entry->shape[1]==inner&&entry->dataSize==rows*inner*2,"Matrix selection mismatch");
  auto modified=std::filesystem::last_write_time(entry->filePath);auto size=std::filesystem::file_size(entry->filePath);
  SafeTensorsMappedFile file;std::string e;if(!file.Open(entry->filePath,e))throw std::runtime_error(e);auto data=static_cast<const unsigned char*>(file.DataAt(entry->dataOffset,entry->dataSize));need(data,"Mapped data missing");
  const auto before=MetalContext::PrimaryStats();const bool baseAllow=WideStorage::Eligible(*indexes.at(model));const bool roleAllow=WideVocabularyPolicy::AllowPrimary(name,rows,inner,baseAllow,true);const bool rawHeadExcluded=baseAllow&&!roleAllow;const bool allow=baseAllow&&roleAllow;auto w=gpu.UploadBF16(data,entry->dataSize,rows,inner,false,false,allow);bool expected=allow&&rows>256&&rows%4==0&&inner%128==0;need(w.PrimaryReady()==expected,"Actual checkpoint admission differs from plan");
  uint64_t rawChecked=0,checked=0,rawHeadChecked=0;
  std::map<std::string,uint64_t> tileWords{{"32x32",0},{"64x32",0},{"64x64",0}};
  if(expected){
   auto info=MetalContext::WeightBufferInfo(w);need(info.size()==2,"Packed checkpoint weight has extra original buffer");uint64_t bytes=0;for(const auto& b:info)bytes+=b.lengthBytes;primaryBytes+=bytes;
   for(size_t start=0;start<rows*inner;start+=8*1024*1024){size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto actual=gpu.PrimaryReadWords(w,start,count);need(actual.size()==count,"GPU read size changed");auto vectorActual=gpu.PrimaryReadVectorWords(w,start,count);need(actual==vectorActual,"Actual checkpoint scalar/vector GPU words differ");for(size_t i=0;i<count;++i){size_t at=(start+i)*2;uint32_t original=uint32_t(data[at])|(uint32_t(data[at+1])<<8);need(actual[i]==(original<<16),"Actual checkpoint GPU FP32 bit reconstruction failed");}checked+=count;}
   for(auto geometry:std::vector<std::pair<size_t,size_t>>{{32,32},{64,32},{64,64}}) {
    const auto key=std::to_string(geometry.first)+"x"+std::to_string(geometry.second);
    const size_t rowChunk=(8*1024*1024/inner/geometry.first)*geometry.first;need(rowChunk>0,"Learned tile row chunk is empty");
    for(size_t first=0;first<rows;first+=rowChunk) {
     const size_t count=std::min(rowChunk,rows-first);auto actual=gpu.PrimaryReadPrefillTileWords(w,first,count,geometry.first,geometry.second);
     need(actual.size()==count*inner,"Learned prefill tile readback size mismatch");
     for(size_t i=0;i<actual.size();++i){size_t at=(first*inner+i)*2;uint32_t original=uint32_t(data[at])|(uint32_t(data[at+1])<<8);need(actual[i]==(original<<16),"Actual checkpoint prefill tile FP32 word mismatch");}
     tileWords[key]+=actual.size();allTileWords[key]+=actual.size();
    }
    need(tileWords[key]==checked,"Learned prefill tile word coverage mismatch");
   }
   ++packedMatrices;wordsChecked+=checked;
  }else{auto info=MetalContext::WeightBufferInfo(w);need(info.size()==1&&info[0].representation=="bf16"&&info[0].lengthBytes==entry->dataSize,"Raw small fallback changed");++rawMatrices;primaryBytes+=entry->dataSize;for(size_t start=0;start<rows*inner;start+=8*1024*1024){size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto actual=gpu.RawBF16ReadWords(w,start,count);need(actual.size()==count,"Raw word size changed");for(size_t i=0;i<count;++i){uint16_t word;std::memcpy(&word,data+2*(start+i),2);need(actual[i]==uint32_t(word)<<16,"Raw learned BF16 original word changed");}rawChecked+=count;rawWordsChecked+=count;}}
  if(rawHeadExcluded){
   need(!expected&&!w.PrimaryReady()&&rows==248320&&inner==2560,"New raw vocabulary identity/admission differs");
   for(size_t start=0;start<rows*inner;start+=8*1024*1024){const size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto actual=gpu.RawBF16ReadGPUWords(w,start,count);need(actual.size()==count,"Raw vocabulary GPU word count differs");for(size_t i=0;i<count;++i){uint16_t word;std::memcpy(&word,data+2*(start+i),2);need(actual[i]==uint32_t(word)<<16,"Raw vocabulary original GPU BF16 word differs");}rawHeadChecked+=count;rawHeadGPUWords+=count;}
   need(rawHeadChecked==rows*inner,"Incomplete actual raw vocabulary GPU words");++rawHeadMatrices;
  }
  rawBytes+=entry->dataSize;w=MetalBF16Weight();auto after=MetalContext::PrimaryStats();need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Corpus weight owner leak");
  need(modified==std::filesystem::last_write_time(entry->filePath)&&size==std::filesystem::file_size(entry->filePath),"Checkpoint changed during GPU check");
  report["matrices"].push_back({{"HOST_raw_FP32_words_checked",rawChecked},{"GPU_raw_wide_vocabulary_FP32_words_checked",rawHeadChecked},{"model",model},{"tensor",name},{"rows",rows},{"inner",inner},{"original_bytes",entry->dataSize},{"packed",expected},{"GPU_original_FP32_words_checked",checked},{"GPU_vector_original_FP32_words_checked",checked},{"GPU_prefill_tile_FP32_words_checked",tileWords},{"owner_accounting_restored",true}});
  report["totals"]={{"GPU_raw_wide_vocabulary_FP32_words_checked",rawHeadGPUWords},{"raw_wide_vocabulary_matrices",rawHeadMatrices},{"HOST_raw_FP32_words_checked",rawWordsChecked},{"packed_matrices",packedMatrices},{"raw_small_matrices",rawMatrices},{"original_bytes",rawBytes},{"retained_primary_and_small_raw_bytes",primaryBytes},{"GPU_original_FP32_words_checked",wordsChecked},{"GPU_vector_original_FP32_words_checked",wordsChecked},{"GPU_prefill_tile_FP32_words_checked",allTileWords}};save(argv[2],report);
  std::cout<<report["matrices"].size()<<" "<<model<<" "<<name<<" words="<<checked<<std::endl;
 }
 need(packedMatrices==200&&rawMatrices==246&&rawBytes==11850481664ULL&&wordsChecked==3565158400ULL&&rawWordsChecked==2360082432ULL&&primaryBytes==10153229824ULL,"Corpus coverage or retained-byte proof mismatch");
 for(const auto& entry:allTileWords)need(entry.second==3565158400ULL,"Complete actual prefill tile words missing");
 need(rawHeadMatrices==1&&rawHeadGPUWords==635699200ULL&&wordsChecked+rawHeadGPUWords==4200857600ULL,"Original word-family coverage missing after representation change");
 report["status"]="full_actual_two_original_and_three_prefill_tile_readers_GPU_bits_exact_not_inference_validated";save(argv[2],report);std::cout<<"Actualcandidate200packed/246raw plus fullrawwidevocabulary GPU word and owner proof PASS"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
