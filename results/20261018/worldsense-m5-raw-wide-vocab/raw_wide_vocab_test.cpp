#include "metal_context.h"
#include "primary_wide_vocab_policy.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <vector>
#include <array>
#include <limits>
#include <iostream>
#include <stdexcept>
using namespace Garnet;using json=nlohmann::json;
static void need(bool v,const char* text){if(!v)throw std::runtime_error(text);}
int main(int argc,char**argv){try{
 need(argc==2,"Usage: raw_wide_vocab_test words.bin");size_t policies=0;
 const std::array<std::pair<std::string,bool>,8> names={{{"lm_head.weight",true},{"model.lm_head.weight",true},{"model.embed_tokens.weight",true},{"model.language_model.embed_tokens.weight",true},{"ordinary.weight",false},{"model.layers.0.self_attn.q_proj.weight",false},{"almostlm_head.weight",false},{"lm_head.weight.extra",false}}};
 const std::array<std::pair<size_t,size_t>,6> shapes={{{0,2560},{65536,2560},{65537,2560},{248320,2560},{248320,2048},{248320,0}}};
 for(const auto& name:names)for(auto shape:shapes)for(bool wide:{false,true})for(bool enabled:{false,true}){
  const bool excluded=name.second&&shape.first>65536&&shape.second>2048&&wide&&enabled;
  need(WideVocabularyPolicy::AllowPrimary(name.first,shape.first,shape.second,wide,enabled)==!excluded,"Semantic wide-vocabulary policy boundary differs");++policies;
 }
 need(policies==192,"Incomplete policy cases");MetalContext gpu,other;const auto initial=MetalContext::PrimaryStats();
 std::vector<uint16_t> bits(512*128);for(size_t i=0;i<bits.size();++i){const auto n=i%65280;bits[i]=uint16_t(n<32640?n:n+128);}
 auto raw=gpu.UploadBF16(bits.data(),bits.size()*2,512,128,false,false,false);need(!raw.PrimaryReady(),"Explicit raw owner packed");
 std::ofstream output(argv[1],std::ios::binary);need(bool(output),"Cannot open raw GPU word stream");size_t records=0,words=0,rejections=0;
 auto check=[&](MetalContext& context,size_t start,size_t count){auto values=context.RawBF16ReadGPUWords(raw,start,count);need(values.size()==count,"Raw GPU result truncated");for(size_t i=0;i<count;++i)need(values[i]==uint32_t(bits[start+i])<<16,"Raw GPU word differs from original BF16 bits");uint64_t head[]={start,count};output.write(reinterpret_cast<const char*>(head),sizeof(head));output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(uint32_t));need(bool(output),"Cannot write raw GPU complete words");++records;words+=count;};
 check(gpu,0,bits.size());check(gpu,1,17);check(gpu,127,129);check(gpu,bits.size()-33,33);check(other,64,128);
 auto rejected=[&](auto call){bool caught=false;try{call();}catch(const std::exception&){caught=true;}need(caught,"Invalid raw GPU readback accepted");++rejections;};
 rejected([&]{gpu.RawBF16ReadGPUWords(raw,0,0);});rejected([&]{gpu.RawBF16ReadGPUWords(raw,0,8*1024*1024+1);});rejected([&]{gpu.RawBF16ReadGPUWords(raw,bits.size()+1,1);});rejected([&]{gpu.RawBF16ReadGPUWords(raw,bits.size()-1,2);});rejected([&]{gpu.RawBF16ReadGPUWords(raw,std::numeric_limits<size_t>::max(),1);});
 gpu.Begin();rejected([&]{gpu.RawBF16ReadGPUWords(raw,0,4);});gpu.Abort();check(gpu,0,4);
 {std::vector<uint16_t> constant(260*128,0x3f80);auto packed=gpu.UploadBF16(constant.data(),constant.size()*2,260,128);
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 need(packed.PrimaryReady(),"Packed-owner negative fixture did not pack");rejected([&]{gpu.RawBF16ReadGPUWords(packed,0,4);});
#else
 need(!packed.PrimaryReady(),"Disabled primary unexpectedly packed fixture");
#endif
 }
 need(records==6&&words==65847&&rejections>=6,"Incomplete raw GPU cases");auto host=gpu.RawBF16ReadWords(raw,0,bits.size());need(host.size()==bits.size(),"Original HOST reader truncated");for(size_t i=0;i<host.size();++i)need(host[i]==uint32_t(bits[i])<<16,"Original HOST word differs");
 raw=MetalBF16Weight();const auto final=MetalContext::PrimaryStats();need(final.activeWeights==initial.activeWeights&&final.originalBytes==initial.originalBytes&&final.payloadBytes==initial.payloadBytes&&final.metadataBytes==initial.metadataBytes&&final.allocatedBytes==initial.allocatedBytes,"GPU diagnostic retained weight owner");
 const auto program=MetalContext::RawBF16ReadGPUProgramSource();std::ofstream shader("raw_bf16_gpu_program.metal",std::ios::binary);shader<<program;need(bool(shader),"Cannot save actual raw diagnostic source");
#ifdef GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB
 constexpr bool enabled=true;
#else
 constexpr bool enabled=false;
#endif
 json result={{"new_policy_enabled",enabled},{"pure_policy_boundary_cases",policies},{"complete_records",records},{"complete_FP32_words",words},{"original_HOST_words",host.size()},{"same_device_other_context_pass",true},{"invalid_cases_rejected",rejections},{"all_finite_BF16_patterns",65280},{"owner_accounting_restored",true},{"no_model_or_speed_claim",true}};
 std::ofstream report(std::string(argv[1])+".summary.json");report<<result.dump(2);need(bool(report),"Cannot save raw GPU summary");std::cout<<"Raw wide vocabulary policy and bounded diagnostic GPU complete words PASS "<<result.dump()<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
