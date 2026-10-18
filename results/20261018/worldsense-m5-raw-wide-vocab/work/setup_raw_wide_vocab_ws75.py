"""Implement only the declared semantic raw-wide vocabulary storage policy."""
from pathlib import Path
import json,hashlib,shutil,ast,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-raw-wide-vocab';old=b/'work/worldsense-minimal-selective-defaults';s=r/'out/research/worldsense-raw-wide-vocab/source';seed=r/'out/research/worldsense-minimal-selective-defaults/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'protocol.json');m=load(w/'seed-source-manifest.json');assert p['state']=='DESIGN_BEFORE_SOURCE'and p['design_sha256']==sha(w/'native-design-before-implementation.txt')and not s.exists();assert len(m)==1721 and all(sha(seed/k)==v for k,v in m.items())
shutil.copytree(seed,s,ignore=shutil.ignore_patterns('__pycache__','.git'));ops=[]
def change(rel,before,after,append=False):
 path=s/rel;t=path.read_text()
 if append:assert not before;new=t+after
 else:assert t.count(before)==1,(rel,before,t.count(before));new=t.replace(before,after)
 path.write_text(new);ops.append(dict(path=rel,before=before,after=after,append=append,seed_sha256=m[rel],result_sha256=sha(path)))
policy='''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <cstddef>
namespace Garnet::WideVocabularyPolicy {
// An exclusion only: all original shape/dtype/device/packing checks still apply.
inline bool EndsWith(const std::string& name,const std::string& suffix) {
 return name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
}
inline bool AllowPrimary(const std::string& name,size_t rows,size_t columns,bool wideCheckpoint,bool rawWideVocabulary) {
 const bool vocabulary=name=="lm_head.weight"||EndsWith(name,".lm_head.weight")||EndsWith(name,".embed_tokens.weight");
 return !(rawWideVocabulary&&wideCheckpoint&&vocabulary&&rows>65536&&columns>2048);
}
}
''';(s/'src/backends/metal/primary_wide_vocab_policy.h').write_text(policy)
change('src/backends/metal/compiled_graph.cpp','#include "wide_checkpoint_policy.h"\n','#include "wide_checkpoint_policy.h"\n#ifdef GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB\n#include "primary_wide_vocab_policy.h"\n#endif\n')
anchor='    allowPrimary=wideCheckpoint_;\n#endif\n';change('src/backends/metal/compiled_graph.cpp',anchor,anchor+'''#ifdef GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB
    allowPrimary=allowPrimary&&WideVocabularyPolicy::AllowPrimary(name,size_t(entry->shape[0]),size_t(entry->shape[1]),wideCheckpoint_,true);
#endif
''')
change('src/backends/metal/metal_context.h','    std::vector<uint32_t> RawBF16ReadWords(const MetalBF16Weight&,size_t start,size_t count);\n','    std::vector<uint32_t> RawBF16ReadWords(const MetalBF16Weight&,size_t start,size_t count);\n    std::vector<uint32_t> RawBF16ReadGPUWords(const MetalBF16Weight&,size_t start,size_t count);\n    static std::string RawBF16ReadGPUProgramSource();\n')
diagnostic='''
namespace Garnet {
std::string MetalContext::RawBF16ReadGPUProgramSource() {
 return R"RAWGPU(#include <metal_stdlib>
using namespace metal;
kernel void raw_bf16_expand_words(device const ushort* weight [[buffer(0)]],device uint* output [[buffer(1)]],
 constant ulong& start [[buffer(2)]],constant ulong& count [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(ulong(id)<count)output[id]=uint(weight[start+ulong(id)])<<16;
}
)RAWGPU";
}
std::vector<uint32_t> MetalContext::RawBF16ReadGPUWords(const MetalBF16Weight& weight,size_t start,size_t count) {
 @autoreleasepool {
  if(impl_->command||impl_->submitted.count)throw std::logic_error("Raw GPU readback requires closed drained batch");
  auto owner=weight.impl_;
  if(!owner||weight.PrimaryReady()||!owner->buffer||owner->buffer.device!=impl_->device||owner->buffer.storageMode!=MTLStorageModeShared)
   throw std::invalid_argument("Raw GPU readback requires original same-device shared BF16 owner");
  const size_t total=Elements(owner->rows,owner->columns);
  if(!count||count>8*1024*1024||start>total||count>total-start||owner->buffer.length!=total*sizeof(uint16_t))
   throw std::invalid_argument("Raw GPU readback range/length invalid");
  // Diagnostic only: no serving kernel/program change or retained decoded cache.
  const auto program=RawBF16ReadGPUProgramSource();NSError* error=nil;
  auto library=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:program.c_str()] options:nil error:&error];
  if(!library){NSString* message=error.localizedDescription;throw std::runtime_error(message?std::string(message.UTF8String):"Raw GPU readback shader failed");}
  auto function=[library newFunctionWithName:@"raw_bf16_expand_words"];
  if(!function)throw std::runtime_error("Raw GPU readback function missing");
  auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
  if(!pipeline||pipeline.maxTotalThreadsPerThreadgroup<256)throw std::runtime_error("Raw GPU readback pipeline unavailable");
  id<MTLBuffer> output=[impl_->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
  if(!output)throw std::runtime_error("Raw GPU readback allocation failed");
  auto command=[impl_->queue commandBuffer];if(!command)throw std::runtime_error("Raw GPU readback command failed");
  auto encoder=[command computeCommandEncoder];if(!encoder)throw std::runtime_error("Raw GPU readback encoder failed");
  const uint64_t offset=start,length=count;
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:owner->buffer offset:0 atIndex:0];[encoder setBuffer:output offset:0 atIndex:1];
  [encoder setBytes:&offset length:sizeof(offset) atIndex:2];[encoder setBytes:&length length:sizeof(length) atIndex:3];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
  [command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Raw GPU readback execution failed");
  std::vector<uint32_t> result(count);std::memcpy(result.data(),output.contents,count*4);return result;
 }
}
}
''';change('src/backends/metal/metal_context.mm','',diagnostic,append=True)
change('src/entry/metal_main.cpp','        return info;\n','''#ifdef GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB
        info+="; primary_raw_wide_vocab";
#endif
        return info;
''')
cmake='''
option(GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB "Keep semantic wide-checkpoint vocabulary in original raw BF16" OFF)
if(GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_WIDE_PRIMARY_STORAGE)
  message(FATAL_ERROR "Raw wide vocabulary requires primary weights and wide checkpoint storage")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=1)
endif()
add_executable(garnet_raw_wide_vocab_test test2026/metal/raw_wide_vocab_test.cpp)
target_link_libraries(garnet_raw_wide_vocab_test PRIVATE garnet_metal)
target_include_directories(garnet_raw_wide_vocab_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
add_test(NAME metal_raw_wide_vocab_reference COMMAND garnet_raw_wide_vocab_test raw_wide_vocab_words.bin)
''';change('cmake/GarnetMetal.cmake','',cmake,append=True)
native='''#include "metal_context.h"
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
 {std::vector<uint16_t> constant(4*128,0x3f80);auto packed=gpu.UploadBF16(constant.data(),constant.size()*2,4,128);if(packed.PrimaryReady())rejected([&]{gpu.RawBF16ReadGPUWords(packed,0,4);});}
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
''';(s/'test2026/metal/raw_wide_vocab_test.cpp').write_text(native)
actual={str(p.relative_to(s)):sha(p)for p in s.rglob('*')if p.is_file()};changed=sorted(k for k in actual if actual[k]!=m.get(k));expected=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/primary_wide_vocab_policy.h','src/entry/metal_main.cpp','test2026/metal/raw_wide_vocab_test.cpp'];assert changed==expected and len(actual)==1723
for rel in set(x['path']for x in ops):
 t=(s/rel).read_text()
 for op in reversed([x for x in ops if x['path']==rel]):
  if op['append']:assert t.endswith(op['after']);t=t[:-len(op['after'])]
  else:assert t.count(op['after'])==1;t=t.replace(op['after'],op['before'])
 assert hashlib.sha256(t.encode()).hexdigest()==m[rel]
(w/'correctness-source-manifest.json').write_text(json.dumps(actual,indent=2)+'\n');(w/'source-transformations.json').write_text(json.dumps(ops,indent=2)+'\n');(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1721,source_files=1723,changed_or_added=changed,new_files=[k for k in actual if k not in m],all_existing_changes_inverse_exact_to74=True,all_other1716files_byte_exact=True,all_production_shaders_math_codec_payload_CUDA_engine_original_tests_unchanged=True),indent=2)+'\n')
shutil.copytree(old/'build-profile',w/'build-profile')
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','primary_core_isolation_test.cpp']:shutil.copy2(old/name,w/name)
shutil.copy2(s/'test2026/metal/raw_wide_vocab_test.cpp',w/'raw_wide_vocab_test.cpp')
t=(old/'build.py').read_text().replace('worldsense-minimal-selective-defaults','worldsense-raw-wide-vocab').replace('len(manifest)==1721','len(manifest)==1723');anchor=" flags['GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS']='ON'if role in ['primary','on','fault']else'OFF'";assert t.count(anchor)==1;t=t.replace(anchor,anchor+"\n flags['GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB']='ON'if role in ['on','fault']else'OFF'");ast.parse(t);(w/'build.py').write_text(t)
d=load(b/'work/worldsense-next.json');d.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='SOURCE_READY_BUILD_NEXT',source_implemented=True,source_files=1723,inference_source_delta=7);(b/'work/worldsense-next.json').write_text(json.dumps(d,indent=2)+'\n');print('WS75 source1723/sevendelta, exact74inverse/production shader invariants checked. New purepolicy and bounded GPU diagnostic/native proof ready; no build/model yet.')
