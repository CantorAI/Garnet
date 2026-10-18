#include "metal_context.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <set>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char*m){if(!v)throw std::runtime_error(m);}
static void save(const char*path,const json&report){auto tmp=std::string(path)+".tmp";{std::ofstream f(tmp);f<<report.dump(2);need(bool(f),"Cannot save paired corpus");}std::filesystem::rename(tmp,path);}
int main(int argc,char**argv){try{
 need(argc==3,"Usage: paired selection.json proof.json");json selection;{std::ifstream f(argv[1]);f>>selection;}MetalContext gpu;json report={{"scope","Every actual paired gate/up BF16 word checked through the SAME vector helper as new paired decode. Correctness only, no inference/speed/unique-total claim."},{"pairs",json::array()}};save(argv[2],report);uint64_t pairs=0,matrices=0,words=0;std::set<std::string>names;
 for(const auto& item:selection.at("matrices")){
  const auto model=item.at("model").get<std::string>(),gate=item.at("tensor").get<std::string>();const std::string suffix=".mlp.gate_proj.weight";
  if(model!="Qwen3-1.7B"||gate.size()<suffix.size()||gate.compare(gate.size()-suffix.size(),suffix.size(),suffix)!=0)continue;
  auto up=gate;up.replace(up.find("gate_proj"),9,"up_proj");need(names.insert(gate).second&&names.insert(up).second,"Duplicate paired tensor");
  SafeTensorsIndex index;std::string error;const auto checkpoint=item.at("checkpoint").get<std::string>();need(index.Open(checkpoint,error),"Cannot open original paired checkpoint");const auto* g=index.Find(gate);const auto* u=index.Find(up);need(g&&u,"Missing learned gate/up pair");
  for(const auto* p:{g,u})need(p->dataType=="BF16"&&p->shape.size()==2&&p->shape[0]==6144&&p->shape[1]==2048&&p->dataSize==6144ull*2048*2,"Learned paired shape/dtype mismatch");
  auto gm=std::filesystem::last_write_time(g->filePath),um=std::filesystem::last_write_time(u->filePath);auto gb=std::filesystem::file_size(g->filePath),ub=std::filesystem::file_size(u->filePath);SafeTensorsMappedFile gf,uf;need(gf.Open(g->filePath,error)&&uf.Open(u->filePath,error),"Cannot map original paired checkpoint");auto gd=static_cast<const unsigned char*>(gf.DataAt(g->dataOffset,g->dataSize));auto ud=static_cast<const unsigned char*>(uf.DataAt(u->dataOffset,u->dataSize));need(gd&&ud,"Mapped pair out of bounds");auto before=MetalContext::SoleStats();auto serving=MetalContext::DualProjectionStats();uint64_t checked=0;
  {
   auto wg=gpu.UploadBF16(gd,g->dataSize,6144,2048,false,false,false),wu=gpu.UploadBF16(ud,u->dataSize,6144,2048,false,false,false);
   for(const auto* p:{&wg,&wu}){auto info=MetalContext::WeightBufferInfo(*p);need(info.size()==1&&info[0].representation=="sole_bf16_interleaved"&&info[0].lengthBytes==6144ull*2048*2,"Paired learned weight has unexpected representation/master");}
   const size_t K=2048,total=6144*2048/16;
   for(size_t first=0;first<total;first+=262144){size_t count=std::min(size_t(262144),total-first);auto actual=gpu.SoleDualProjectionReadWords(wg,wu,first,count);need(actual.size()==count*32,"Paired learned packet size");
    for(size_t i=0;i<count;++i){size_t block=first+i,col=(block/(K/4))*4,k=(block%(K/4))*4;for(size_t j=0;j<4;++j)for(size_t c=0;c<4;++c){size_t at=((col+c)*K+k+j)*2;uint32_t gw=uint32_t(gd[at])|(uint32_t(gd[at+1])<<8),uw=uint32_t(ud[at])|(uint32_t(ud[at+1])<<8);need(actual[i*32+j*4+c]==gw<<16&&actual[i*32+16+j*4+c]==uw<<16,"Actual paired vector helper word mismatch");}}
    checked+=count*32;
   }
  }
  auto after=MetalContext::SoleStats();auto endServing=MetalContext::DualProjectionStats();need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes&&after.scratchBytes==before.scratchBytes,"Learned pair owner/bytes leak");need(endServing.encodedPairs==serving.encodedPairs&&endServing.representedProjections==serving.representedProjections,"Learned probe mutated serving counter");need(checked==2*6144ull*2048&&gm==std::filesystem::last_write_time(g->filePath)&&um==std::filesystem::last_write_time(u->filePath)&&gb==std::filesystem::file_size(g->filePath)&&ub==std::filesystem::file_size(u->filePath),"Paired corpus coverage/checkpoint changed");
  ++pairs;matrices+=2;words+=checked;report["pairs"].push_back({{"model",model},{"gate",gate},{"up",up},{"original_FP32_words_checked",checked},{"same_decode_vector_helper",true},{"owner_accounting_restored",true},{"serving_counters_unchanged",true}});report["totals"]={{"pairs",pairs},{"matrices",matrices},{"original_FP32_words_checked",words}};save(argv[2],report);std::cout<<pairs<<" "<<gate<<"/up original_words="<<checked<<std::endl;
 }
 need(pairs==28&&matrices==56&&names.size()==56&&words==704643072ULL,"Missing original learned gate/up pairs");report["status"]="all56original_gate_up_weights_same_decode_vector_helper_GPU_words_exact";save(argv[2],report);std::cout<<"All28learnedpairs/56matrices/704643072originalFP32words SAME paired-decode vector helper PASS"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
