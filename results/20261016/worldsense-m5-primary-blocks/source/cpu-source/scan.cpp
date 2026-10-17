#include "primary_block_codec.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <map>
#include <memory>
#include <algorithm>
using json=nlohmann::json;
using namespace PrimaryBlocks;
static void Need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
struct Hash{
 CC_SHA256_CTX context;Hash(){CC_SHA256_Init(&context);}
 void add(const void* p,uint64_t n){const auto* b=static_cast<const uint8_t*>(p);while(n){const auto part=std::min<uint64_t>(n,8*1024*1024);CC_SHA256_Update(&context,b,CC_LONG(part));n-=part;b+=part;}}
 std::string finish(){unsigned char data[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(data,&context);std::ostringstream o;for(auto v:data)o<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(v);return o.str();}
};
static void Save(const char* path,const json& report){auto tmp=std::string(path)+".tmp";{std::ofstream f(tmp);f<<report.dump(2);Need(bool(f),"Report write failed");}std::filesystem::rename(tmp,path);}
static uint64_t Round(uint64_t n,uint64_t page){Need(page&&n<=UINT64_MAX-page+1,"Page-rounded size overflow");return (n+page-1)/page*page;}
static json SelfTest(){
 std::array<uint16_t,Words> words{};uint64_t finite=0,mixed=0,compact=0,raw=0;
 for(unsigned value=0;value<=65535;++value){if(((value>>7)&255)==255)continue;words.fill(uint16_t(value));auto b=Encode(words,0);Need(b.compact,"Uniform finite block was not compact");for(size_t i=0;i<Words;++i)Need(DecodeLocal(b.metadata,b.data.data(),b.bytes,i)==value,"Exhaustive finite word mismatch");++finite;}
 std::vector<uint8_t> payload;std::vector<uint64_t> metadata;std::vector<std::array<uint16_t,Words>> original;
 for(unsigned span:{0u,1u,14u,15u,16u,254u})for(unsigned pattern=0;pattern<9;++pattern){
  for(size_t i=0;i<Words;++i){unsigned e=span?unsigned((i*29+pattern*7)%(span+1)):0;unsigned sm=unsigned((i*67+pattern*19)&127);unsigned sign=unsigned(((i+pattern)&1)<<15);words[i]=uint16_t(sign|(e<<7)|sm);}
  words[0]=0;words[1]=uint16_t(span<<7);auto b=Encode(words,payload.size());Need(b.compact==(span<=15),"Exponent range admission boundary changed");Need(Offset(b.metadata)==payload.size(),"Offset packing changed");for(size_t i=0;i<Words;++i)Need(DecodeLocal(b.metadata,b.data.data(),b.bytes,i)==words[i],"Mixed-word reconstruction mismatch");
  b.compact?++compact:++raw;metadata.push_back(b.metadata);original.push_back(words);payload.insert(payload.end(),b.data.begin(),b.data.begin()+b.bytes);++mixed;
 }
 for(size_t b=0;b<metadata.size();++b)for(size_t i=0;i<Words;++i)Need(Decode(metadata[b],payload.data(),payload.size(),i)==original[b][i],"Variable mixed-payload address mismatch");
 auto rejects=[](auto fn){try{fn();return false;}catch(const std::exception&){return true;}};
 Need(rejects([&]{Encode(words,1);}),"Unaligned metadata accepted");
 Need(rejects([&]{Encode(words,UINT64_MAX-255);}),"Overflow metadata accepted");
 words.fill(0);words[511]=0x7f80;Need(rejects([&]{Encode(words,0);}),"Infinity accepted");
 words[511]=0xffff;Need(rejects([&]{Encode(words,0);}),"NaN accepted");
 Need(rejects([&]{Decode(0,payload.data(),1023,0);}),"Short raw block accepted");
 Need(rejects([&]{Decode(256,payload.data(),767,0);}),"Short compact block accepted");
 Need(rejects([&]{Decode(1,payload.data(),payload.size(),0);}),"Raw exponent metadata accepted");
 Need(rejects([&]{Decode(511,payload.data(),payload.size(),0);}),"Nonfinite compact metadata accepted");
 Need(rejects([&]{Decode(256,payload.data(),payload.size(),512);}),"Out-of-range word accepted");
 Need(!Eligible(3,128)&&!Eligible(4,127)&&!Eligible(0,128)&&Eligible(4,128),"Shape fallback changed");
 return {{"finite_BF16_words_exhaustively_checked",finite},{"each_word_positions",Words},{"mixed_variable_blocks",mixed},{"compact_blocks",compact},{"raw_blocks",raw},{"signed_zero_subnormals_and_15_vs16",true},{"bad_offset_nonfinite_shape_and_decode_bounds_rejected",true}};
}
int main(int argc,char**argv){try{
 Need(argc==3,"Usage: scan selection.json output.json");std::ifstream input(argv[1]);json selection;input>>selection;
 json report={{"scope","Native CPU codec/count feasibility only. No GPU allocation, inference, speed or physical-memory claim."},{"selftest",SelfTest()},{"matrices",json::array()}};
 Save(argv[2],report);std::map<std::string,std::unique_ptr<Garnet::SafeTensorsIndex>> indices;uint64_t totalRaw=0,totalStored=0,totalRounded=0,totalBlocks=0,totalCompact=0;
 const auto page=uint64_t(sysconf(_SC_PAGESIZE));Need(page>0&&page<=1048576,"Invalid OS page size");report["page_bytes"]=page;size_t index=0;
 for(const auto& item:selection.at("matrices")){
  const auto model=item.at("model").get<std::string>(),checkpoint=item.at("checkpoint").get<std::string>(),name=item.at("tensor").get<std::string>();
  if(!indices.count(model)){auto ptr=std::make_unique<Garnet::SafeTensorsIndex>();std::string e;if(!ptr->Open(checkpoint,e))throw std::runtime_error(e);indices.emplace(model,std::move(ptr));}
  const auto* entry=indices.at(model)->Find(name);Need(entry&&entry->dataType=="BF16"&&entry->shape.size()==2,"Selected matrix missing or not BF16");
  const uint64_t rows=item.at("rows"),inner=item.at("inner");Need(rows>0&&inner>0&&rows<=UINT64_MAX/inner/2,"Matrix dimensions overflow");
  const auto originalBytes=rows*inner*2;Need(entry->shape[0]==rows&&entry->shape[1]==inner&&entry->dataSize==originalBytes&&originalBytes==item.at("original_buffer_length_bytes").get<uint64_t>(),"Selection/metadata dimensions mismatch");
  const auto modified=std::filesystem::last_write_time(entry->filePath);const auto fileSize=std::filesystem::file_size(entry->filePath);
  Garnet::SafeTensorsMappedFile file;std::string error;if(!file.Open(entry->filePath,error))throw std::runtime_error(error);auto data=static_cast<const uint8_t*>(file.DataAt(entry->dataOffset,entry->dataSize));Need(data,"Mapped tensor range invalid");
  Hash rawHash;rawHash.add(data,originalBytes);Hash payloadHash,metaHash;uint64_t bytes=0,blocks=0,compacted=0;const bool eligible=Eligible(rows,inner);
  if(eligible){
   std::array<uint16_t,Words> words;
   for(uint64_t group=0;group<rows/4;++group)for(uint64_t block=0;block<inner/128;++block){
    for(uint64_t k=0;k<128;++k)for(uint64_t j=0;j<4;++j){const auto pos=((group*4+j)*inner+block*128+k)*2;words[k*4+j]=uint16_t(data[pos])|uint16_t(data[pos+1])<<8;}
    const auto encoded=Encode(words,bytes);Need(Offset(encoded.metadata)==bytes,"Encoded block address mismatch");
    for(size_t i=0;i<Words;++i)Need(DecodeLocal(encoded.metadata,encoded.data.data(),encoded.bytes,i)==words[i],"Checkpoint original BF16 bit mismatch");
    uint8_t meta[8];for(size_t i=0;i<8;++i)meta[i]=uint8_t(encoded.metadata>>(i*8));metaHash.add(meta,8);payloadHash.add(encoded.data.data(),encoded.bytes);
    bytes+=encoded.bytes;++blocks;if(encoded.compact)++compacted;
   }
  }else{
   for(uint64_t i=0;i<originalBytes;i+=2)Need(((uint16_t(data[i])|uint16_t(data[i+1])<<8)>>7&255)!=255,"Nonfinite unsupported matrix");
  }
  Need(modified==std::filesystem::last_write_time(entry->filePath)&&fileSize==std::filesystem::file_size(entry->filePath),"Checkpoint changed during scan");
  const uint64_t metaBytes=blocks*8;const bool admitted=eligible&&bytes+metaBytes<originalBytes;
  const auto retained=admitted?bytes+metaBytes:originalBytes;const auto rounded=admitted?Round(bytes,page)+Round(metaBytes,page):Round(originalBytes,page);
  json row={{"model",model},{"tensor",name},{"rows",rows},{"inner",inner},{"original_bytes",originalBytes},{"original_Metal_allocated_bytes",item.at("original_allocated_bytes")},{"eligible_shape",eligible},{"admitted_by_bytes_only",admitted},{"blocks",blocks},{"compact_blocks",compacted},{"raw_blocks",blocks-compacted},{"encoded_payload_bytes",bytes},{"metadata_bytes",metaBytes},{"retained_primary_bytes",retained},{"page_rounded_retained_estimate",rounded},{"all_original_BF16_bits_exact",true},{"raw_tensor_sha256",rawHash.finish()},{"encoded_payload_stream_sha256",payloadHash.finish()},{"metadata_stream_sha256",metaHash.finish()}};
  report["matrices"].push_back(row);totalRaw+=originalBytes;totalStored+=retained;totalRounded+=rounded;totalBlocks+=blocks;totalCompact+=compacted;
  report["totals"]={{"matrices",++index},{"original_bytes",totalRaw},{"retained_primary_bytes",totalStored},{"page_rounded_retained_estimate",totalRounded},{"blocks",totalBlocks},{"compact_blocks",totalCompact}};
  Save(argv[2],report);std::cout<<index<<" "<<model<<" "<<name<<" "<<originalBytes<<" -> "<<retained<<std::endl;
 }
 Need(index==446&&totalRaw==11850481664ULL,"Full actual-buffer coverage incomplete");
 report["status"]="cpu_feasibility_complete_exact_bits_not_GPU_validated";Save(argv[2],report);std::cout<<"COMPLETE "<<report["totals"].dump()<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
