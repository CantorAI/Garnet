#include "metal_context.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <cstring>
#include <filesystem>
using namespace Garnet;using json=nlohmann::json;
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static void save(const char* p,const json& j){std::ofstream f(p);f<<j.dump(2);need(bool(f),"Cannot save direct corpus proof");}
static std::string digest(const void* p,size_t bytes){need(bytes<=UINT32_MAX,"Digest input too large");unsigned char out[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,CC_LONG(bytes),out);const char* hex="0123456789abcdef";std::string s;for(auto c:out){s+=hex[c>>4];s+=hex[c&15];}return s;}
int main(int argc,char**argv){try{
 need(argc==3,"Usage direct corpus selection.json result.json");std::ifstream f(argv[1]);json selection;f>>selection;MetalContext gpu;
 std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;uint64_t words=0;json report={{"scope","Exact native direct-upload/download words for every selected learned BF16 vision/norm/depthwise/patch raw-weight candidate. No speed evidence."},{"weights",json::array()}};
 for(const auto& item:selection.at("weights")){
  std::string model=item.at("model"),checkpoint=item.at("checkpoint"),name=item.at("tensor");
  if(!indexes.count(model)){auto index=std::make_unique<SafeTensorsIndex>();std::string e;need(index->Open(checkpoint,e),e.c_str());indexes.emplace(model,std::move(index));}
  const auto* entry=indexes.at(model)->Find(name);need(entry&&entry->dataType=="BF16","Missing BF16 tensor");need(entry->shape==item.at("shape").get<std::vector<long long>>(),"Shape changed");
  const size_t rows=item.at("rows"),columns=item.at("columns");need(entry->dataSize==rows*columns*2,"BF16 size mismatch");const auto modified=std::filesystem::last_write_time(entry->filePath);const auto bytes=std::filesystem::file_size(entry->filePath);
  SafeTensorsMappedFile mapped;std::string e;need(mapped.Open(entry->filePath,e),e.c_str());const auto* raw=static_cast<const char*>(mapped.DataAt(entry->dataOffset,entry->dataSize));need(raw,"Invalid mapped tensor");
  auto matrix=gpu.UploadBF16AsFloat(raw,entry->dataSize,rows,columns);auto actual=gpu.Download(matrix);need(actual.size()==rows*columns,"Direct readback size changed");
  for(size_t i=0;i<actual.size();++i){uint16_t original;uint32_t value;std::memcpy(&original,raw+2*i,2);std::memcpy(&value,&actual[i],4);need(value==uint32_t(original)<<16,"Direct learned FP32 word changed");}
  auto info=MetalContext::MatrixBufferInfo(matrix);need(info.representation=="matrix_fp32","Direct representation changed");
  need(modified==std::filesystem::last_write_time(entry->filePath)&&bytes==std::filesystem::file_size(entry->filePath),"Checkpoint changed");words+=actual.size();
  report["weights"].push_back({{"model",model},{"tensor",name},{"rows",rows},{"columns",columns},{"shape",entry->shape},{"BF16_sha256",digest(raw,entry->dataSize)},{"FP32_words_checked",actual.size()},{"storage",info.representation}});report["totals"]={{"weights",report["weights"].size()},{"FP32_words_checked",words}};save(argv[2],report);std::cout<<model<<" "<<name<<" words="<<actual.size()<<std::endl;
 }
 std::cout<<"DIRECT_LEARNED_FLOAT_UPLOAD_WORDS_EXACT"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
