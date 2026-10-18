#include "metal_context.h"
#include "wide_checkpoint_policy.h"
#include "safetensors_index.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
using json=nlohmann::json;using namespace Garnet;
static void need(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static void save(const char* path,const json& j){auto p=std::string(path)+".tmp";{std::ofstream f(p);f<<j.dump(2);need(bool(f),"Cannot save lease corpus");}std::filesystem::rename(p,path);}
static bool sameCounters(const MetalPrimaryStats&a,const MetalPrimaryStats&b){return a.decodeEncoded==b.decodeEncoded&&a.prefillEncoded==b.prefillEncoded&&a.headEncoded==b.headEncoded&&a.vectorDecodeEncoded==b.vectorDecodeEncoded&&a.vectorHeadEncoded==b.vectorHeadEncoded;}
static void clean(MetalContext& gpu){auto q=json::parse(gpu.PackedRequestWeightResidencyStatsJSON());need(!q.at("scope_open").get<bool>()&&!q.at("attached").get<bool>()&&!q.at("requested").get<bool>(),"Lease flags retained");for(auto k:{"active_bytes","global_used_bytes","set_allocation_count","owner_count"})need(q.at(k)==0,"Lease allocation/budget/owner retained");}
int main(int argc,char**argv){try{
 need(argc==3,"Usage: packed_lease_corpus selection.json proof.json");std::ifstream f(argv[1]);json selection;f>>selection;MetalContext gpu;
 json report={{"scope","Original learned FP32 words while actual packed payload and metadata lease is active. Allocation hints only; no physical residency or speed claim."},{"matrices",json::array()}};
 std::map<std::string,std::unique_ptr<SafeTensorsIndex>> indexes;uint64_t matrices=0,skipped=0,checked=0,allocationBytes=0;save(argv[2],report);
 for(const auto& item:selection.at("matrices")){
  auto model=item.at("model").get<std::string>(),checkpoint=item.at("checkpoint").get<std::string>(),name=item.at("tensor").get<std::string>();
  if(!indexes.count(model)){auto p=std::make_unique<SafeTensorsIndex>();std::string e;need(p->Open(checkpoint,e),e.c_str());indexes.emplace(model,std::move(p));}
  const auto* entry=indexes.at(model)->Find(name);need(entry&&entry->dataType=="BF16"&&entry->shape.size()==2,"Missing selected matrix");
  size_t rows=item.at("rows"),inner=item.at("inner");need(entry->shape[0]==rows&&entry->shape[1]==inner&&entry->dataSize==rows*inner*2,"Matrix selection mismatch");
  bool allow=WideStorage::Eligible(*indexes.at(model)),expected=allow&&rows>256&&rows%4==0&&inner%128==0;
  if(!expected){++skipped;continue;}
  auto modified=std::filesystem::last_write_time(entry->filePath);auto fileSize=std::filesystem::file_size(entry->filePath);
  SafeTensorsMappedFile file;std::string e;need(file.Open(entry->filePath,e),e.c_str());auto data=static_cast<const unsigned char*>(file.DataAt(entry->dataOffset,entry->dataSize));need(data,"Missing mapped words");
  auto before=MetalContext::PrimaryStats();auto weight=gpu.UploadBF16(data,entry->dataSize,rows,inner,false,false,allow);need(weight.PrimaryReady(),"Selected weight did not pack");
  auto info=MetalContext::WeightBufferInfo(weight);need(info.size()==2,"Packed weight has extra/missing buffers");
  uint64_t logical=0,allocated=0;for(const auto& buffer:info){logical+=buffer.lengthBytes;allocated+=buffer.allocatedBytes;}
  need(info[0].identity!=info[1].identity,"Packed payload and metadata unexpectedly alias");
  need(gpu.BeginPackedRequestWeightResidency({weight,weight}),"Learned two-buffer lease not admitted");
  auto active=json::parse(gpu.PackedRequestWeightResidencyStatsJSON());
  need(active.at("scope_open").get<bool>()&&active.at("attached").get<bool>()&&active.at("requested").get<bool>(),"Lease not attached/requested");
  need(active.at("last_count")==2&&active.at("attempted_count")==2&&active.at("set_allocation_count")==2&&active.at("owner_count")==1&&active.at("reason")==1,"Actual learned allocation/owner tuple changed");
  need(active.at("active_bytes")==active.at("last_bytes")&&active.at("global_used_bytes")==active.at("last_bytes")&&active.at("last_bytes")==allocated&&allocated>=logical,"Actual learned allocation bytes invalid");
  need(gpu.PackedRequestWeightsCovered({weight}),"Leased weight not covered");
  auto preRead=MetalContext::PrimaryStats();uint64_t words=0;
  for(size_t start=0;start<rows*inner;start+=8*1024*1024){
   size_t count=std::min(size_t(8*1024*1024),rows*inner-start);auto scalar=gpu.PrimaryReadWords(weight,start,count),vector=gpu.PrimaryReadVectorWords(weight,start,count);
   need(scalar.size()==count&&vector.size()==count&&scalar==vector,"Scalar/vector learned words differ");
   for(size_t i=0;i<count;++i){size_t at=(start+i)*2;uint32_t original=uint32_t(data[at])|(uint32_t(data[at+1])<<8);need(scalar[i]==(original<<16),"Leased original FP32 reconstruction differs");}
   need(gpu.PackedRequestWeightResidencyActive()&&gpu.PackedRequestWeightsCovered({weight}),"Reader dropped active lease");words+=count;
  }
  auto postRead=MetalContext::PrimaryStats();need(sameCounters(preRead,postRead),"Correctness readers changed serving counters");
  auto afterRead=json::parse(gpu.PackedRequestWeightResidencyStatsJSON());need(afterRead==active,"Readers changed actual lease counters/ownership");
  gpu.EndPackedRequestWeightResidency();clean(gpu);gpu.EndPackedRequestWeightResidency();clean(gpu);
  auto ended=json::parse(gpu.PackedRequestWeightResidencyStatsJSON());need(ended.at("calls")==ended.at("admitted")&&ended.at("calls")==ended.at("ends")&&ended.at("fallback")==0,"Lease calls/admissions/end accounting changed");
  weight=MetalBF16Weight();auto after=MetalContext::PrimaryStats();need(after.activeWeights==before.activeWeights&&after.allocatedBytes==before.allocatedBytes,"Learned owner leak");
  need(modified==std::filesystem::last_write_time(entry->filePath)&&fileSize==std::filesystem::file_size(entry->filePath),"Checkpoint changed");
  ++matrices;checked+=words;allocationBytes+=active.at("last_bytes").get<uint64_t>();
  report["matrices"].push_back({{"model",model},{"tensor",name},{"rows",rows},{"inner",inner},{"scalar_original_FP32_words",words},{"vector_original_FP32_words",words},{"active_lease",active},{"after_cleanup",ended},{"owner_accounting_restored",true},{"serving_counters_unchanged",true}});
  report["totals"]={{"matrices",matrices},{"skipped_raw",skipped},{"scalar_original_FP32_words",checked},{"vector_original_FP32_words",checked},{"sum_individual_actual_allocation_bytes",allocationBytes}};save(argv[2],report);std::cout<<matrices<<" "<<name<<" words="<<words<<std::endl;
 }
 need(matrices==201&&skipped==245&&checked==4200857600ULL,"Incomplete learned leased readers");report["totals"]["skipped_raw"]=skipped;report["status"]="all201_two_original_readers_exact_under_actual_packed_lease";save(argv[2],report);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
