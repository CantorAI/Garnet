"""Implement prospectively recorded sole-layout experiment in an isolated seed."""
from pathlib import Path
import json,hashlib,shutil,re,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-sole-interleaved';r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-isolated-primary/source';s=r/'out/research/worldsense-sole-interleaved/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();protocol=json.loads((w/'protocol.json').read_text());assert protocol['design_sha256']==sha(w/'native-design-before-implementation.txt')
seed=json.loads((w/'read-only-consumer-audit.json').read_text())['seed_source'];assert len(seed)==1718 and all(sha(old/k)==v for k,v in seed.items())
s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
changes={}
def replace(name,a,z,count=1):
 p=s/name;t=p.read_text();assert t.count(a)==count,(name,a,t.count(a));p.write_text(t.replace(a,z));changes.setdefault(name,[]).append(dict(before=a,after=z,count=count))
def append(name,z):
 p=s/name;p.write_text(p.read_text()+z);changes.setdefault(name,[]).append(dict(before='',after=z,count=1,append=True))
h='src/backends/metal/metal_context.h';m='src/backends/metal/metal_context.mm';e='src/entry/metal_main.cpp';serv='runtime/metal_serving.py'
replace(h,'class MetalContext {','''struct MetalSoleStats {
 uint64_t activeWeights=0,allocatedBytes=0,uploads=0,decodeEncoded=0,restoreEncoded=0,scratchBytes=0;
};
class MetalContext {''')
replace(h,'    static MetalPrimaryStats PrimaryStats();','''    static MetalPrimaryStats PrimaryStats();
    static MetalSoleStats SoleStats();
    std::vector<uint32_t> SoleInterleavedReadWords(const MetalBF16Weight&,size_t start,size_t count);
    std::string SoleShaderProgramSource() const;
#if defined(GARNET_METAL_SOLE_INTERLEAVED_MLP) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
    static void SoleTestFailAfter(int);
    static void SoleTestUnavailable(bool);
#endif''')
replace(m,'#include "exact_half_codec.h"','''#include "exact_half_codec.h"
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
#include "sole_interleaved_mlp_shader.h"
#include <atomic>
#endif''')
replace(m,'namespace Garnet {\n#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS','''namespace Garnet {
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
namespace {
constexpr size_t kSoleScratchBytes=2048ull*6144*2;
struct SoleCounters {std::atomic<uint64_t> weights{0},allocated{0},uploads{0},decode{0},restore{0},scratch{0};};
SoleCounters soleCounters;
struct SoleWeightCharge {uint64_t bytes=0;~SoleWeightCharge(){if(bytes){soleCounters.weights.fetch_sub(1);soleCounters.allocated.fetch_sub(bytes);}}};
struct SoleScratchCharge {uint64_t bytes=0;~SoleScratchCharge(){if(bytes)soleCounters.scratch.fetch_sub(bytes);}};
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
std::atomic<int> soleFailure{0};std::atomic<bool> soleUnavailable{false};
void SoleMaybeFail(int stage){int expected=stage;if(soleFailure.compare_exchange_strong(expected,0))throw std::runtime_error("Injected sole-layout allocation failure");}
#endif
}
#endif
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS''')
replace(m,'    id<MTLBuffer> buffer;','''    id<MTLBuffer> buffer;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
    bool soleInterleaved=false;SoleWeightCharge soleCharge;
#endif''')
replace(m,'    id<MTLLibrary> library;','''    id<MTLLibrary> library;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
    id<MTLLibrary> soleLibrary=nil;
    id<MTLComputePipelineState> soleDecode=nil,soleRestore=nil,soleRead=nil;
    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> soleBodyShape;
    std::string soleProgram;
#endif''')
replace(m,'    double gpuMs=0;','''    double gpuMs=0;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
    id<MTLBuffer> soleScratch=nil;SoleScratchCharge soleScratchCharge;
#endif''')
needle='        impl_->kernels->mainProgram=source.UTF8String;'
replace(m,needle,needle+'''
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        // Independent library leaves existing raw/primary source strings intact.
        impl_->kernels->soleProgram=kSoleInterleavedMlpShader;
        auto soleLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kSoleInterleavedMlpShader] options:compileOptions error:&error];
        if(!soleLibrary)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        auto decode=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_bf16_decode"] error:&error];
        auto restore=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_bf16_restore"] error:&error];
        auto read=[impl_->device newComputePipelineStateWithFunction:[soleLibrary newFunctionWithName:@"sole_bf16_read"] error:&error];
        if(!decode||!restore||!read)throw std::runtime_error("Cannot create sole-layout pipelines");
        if(decode.threadExecutionWidth==32&&decode.maxTotalThreadsPerThreadgroup>=128&&
           decode.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength&&
           restore.maxTotalThreadsPerThreadgroup>=128&&read.maxTotalThreadsPerThreadgroup>=128) {
            impl_->kernels->soleLibrary=soleLibrary;impl_->kernels->soleDecode=decode;
            impl_->kernels->soleRestore=restore;impl_->kernels->soleRead=read;
        }
#endif''')
needle='        value->buffer=[impl_->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];'
replace(m,needle,'''#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        bool soleAvailable=bool(impl_->kernels->soleDecode)&&bool(impl_->kernels->soleRestore)&&bool(impl_->kernels->soleRead);
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
        soleAvailable=soleAvailable&&!soleUnavailable.load();
#endif
        if(soleAvailable&&!floatPrefill&&!visionPrefill&&
           ((rows==6144&&columns==2048)||(rows==2048&&columns==6144))) {
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
            SoleMaybeFail(1);
#endif
            auto buffer=[impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if(!buffer||!buffer.contents)throw std::runtime_error("Sole BF16 weight allocation failed");
            auto dst=static_cast<char*>(buffer.contents);auto src=static_cast<const char*>(data);
            for(size_t row=0;row<rows;++row)for(size_t k=0;k<columns;++k)
                std::memcpy(dst+2*((row/4)*columns*4+k*4+row%4),src+2*(row*columns+k),2);
            for(size_t row=0;row<rows;++row)for(size_t k=0;k<columns;++k)
                if(std::memcmp(dst+2*((row/4)*columns*4+k*4+row%4),src+2*(row*columns+k),2))
                    throw std::runtime_error("Sole BF16 original-bit roundtrip failed");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
            SoleMaybeFail(2);
#endif
            value->buffer=buffer;value->soleInterleaved=true;value->soleCharge.bytes=buffer.allocatedSize;
            soleCounters.weights.fetch_add(1);soleCounters.allocated.fetch_add(buffer.allocatedSize);soleCounters.uploads.fetch_add(1);
            return MetalBF16Weight(value);
        }
#endif
'''+needle)
replace(m,'        add(w->buffer,"bf16");if(w->exactHalf)', '''#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        add(w->buffer,w->soleInterleaved?"sole_bf16_interleaved":"bf16");
#else
        add(w->buffer,"bf16");
#endif
        if(w->exactHalf)''')
replace(m,'auto& selectedCache=primary?impl_->kernels->primaryBodyShape:cache;','''auto& selectedCache=primary?impl_->kernels->primaryBodyShape:
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
            w->soleInterleaved?impl_->kernels->soleBodyShape:
#endif
            cache;''',2)
replace(m,'auto& selectedCache=cache;','''auto& selectedCache=
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
            w->soleInterleaved?impl_->kernels->soleBodyShape:
#endif
            cache;''',2)
needle='        if(primary)functionName=@"primary_specialized_bf16_decode";\n#endif'
replace(m,needle,needle+'''
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        if(w->soleInterleaved)functionName=@"sole_specialized_bf16_decode";
#endif''')
replace(m,'auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:impl_->kernels->library;','''auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
            w->soleInterleaved?impl_->kernels->soleLibrary:
#endif
            impl_->kernels->library;''')
replace(m,'auto function=[impl_->kernels->library newFunctionWithName:functionName constantValues:constants error:&error];','''auto sourceLibrary=impl_->kernels->library;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        if(w->soleInterleaved)sourceLibrary=impl_->kernels->soleLibrary;
#endif
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];''')
replace(m,'+impl_->kernels->primaryBodyShape.size()','+impl_->kernels->primaryBodyShape.size()')
replace(m,'        ;\n#else\n    return 0;\n#endif\n}\nsize_t MetalContext::LastBodyShapeDispatches()', '''#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        +impl_->kernels->soleBodyShape.size()
#endif
        ;
#else
    return 0;
#endif
}
size_t MetalContext::LastBodyShapeDispatches()''')
# Set generic sole pipeline before optional shape lookup; specialized sole cache may override it.
needle='#ifdef GARNET_METAL_BODY_SHAPE\n        if(a.rows<=2&&o.columns<100000) {'
replace(m,needle,'''#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        if(w->soleInterleaved&&a.rows<=2){pipeline=impl_->kernels->soleDecode;threads=128;decodeColumns=16;}
#endif
'''+needle)
needle='        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads)'
replace(m,needle,'''#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        if(w->soleInterleaved&&a.rows>2) {
            if(primary||w->floatPrefill||w->visionPrefill||w->buffer.length!=kSoleScratchBytes)
                throw std::logic_error("Invalid sole-layout prefill representation");
            if(!impl_->soleScratch) {
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
                SoleMaybeFail(3);
#endif
                auto scratch=[impl_->device newBufferWithLength:kSoleScratchBytes options:MTLResourceStorageModePrivate];
                if(!scratch)throw std::runtime_error("Sole-layout prefill scratch allocation failed");
                impl_->soleScratch=scratch;impl_->soleScratchCharge.bytes=scratch.allocatedSize;
                soleCounters.scratch.fetch_add(scratch.allocatedSize);
            }
            auto restore=[impl_->command computeCommandEncoder];
            if(!restore)throw std::runtime_error("Cannot create sole-layout restore encoder");
            uint64_t dimensions[2]={w->rows,w->columns};
            [restore setComputePipelineState:impl_->kernels->soleRestore];
            [restore setBuffer:w->buffer offset:0 atIndex:0];[restore setBuffer:impl_->soleScratch offset:0 atIndex:1];
            [restore setBytes:dimensions length:sizeof(dimensions) atIndex:2];
            [restore dispatchThreads:MTLSizeMake(w->rows*w->columns,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[restore endEncoding];
            storage=impl_->soleScratch;soleCounters.restore.fetch_add(1);
        }
        if(w->soleInterleaved&&a.rows<=2)soleCounters.decode.fetch_add(1);
#endif
'''+needle)
needle='  for(size_t i=0;i<count;++i){uint16_t word;std::memcpy(&word,data+2*(start+i),2);output[i]=uint32_t(word)<<16;}'
replace(m,needle,'''  for(size_t i=0;i<count;++i){size_t at=start+i;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
   if(owner->soleInterleaved){size_t row=at/owner->columns,k=at%owner->columns;at=(row/4)*owner->columns*4+k*4+row%4;}
#endif
   uint16_t word;std::memcpy(&word,data+2*at,2);output[i]=uint32_t(word)<<16;}''')
append(m,'''
namespace Garnet {
MetalSoleStats MetalContext::SoleStats() {
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
 return {soleCounters.weights.load(),soleCounters.allocated.load(),soleCounters.uploads.load(),soleCounters.decode.load(),soleCounters.restore.load(),soleCounters.scratch.load()};
#else
 return {};
#endif
}
std::string MetalContext::SoleShaderProgramSource() const {
#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
 return impl_->kernels->soleProgram;
#else
 return {};
#endif
}
#if defined(GARNET_METAL_SOLE_INTERLEAVED_MLP) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
void MetalContext::SoleTestFailAfter(int stage){if(stage<0||stage>3)throw std::invalid_argument("Invalid sole-layout failure stage");soleFailure.store(stage);}
void MetalContext::SoleTestUnavailable(bool flag){soleUnavailable.store(flag);}
#endif
std::vector<uint32_t> MetalContext::SoleInterleavedReadWords(const MetalBF16Weight& value,size_t start,size_t count) {
#ifndef GARNET_METAL_SOLE_INTERLEAVED_MLP
 (void)value;(void)start;(void)count;throw std::logic_error("Sole interleaved storage disabled");
#else
 @autoreleasepool {
 if(impl_->command||impl_->submitted.count)throw std::logic_error("Sole readback requires closed drained batch");
 auto w=value.impl_;
 if(!w||!w->soleInterleaved||!w->buffer||w->buffer.device!=impl_->device||w->buffer.length!=kSoleScratchBytes)
  throw std::invalid_argument("Sole readback requires same-device sole interleaved weight");
 size_t total=Elements(w->rows,w->columns);
 if(!count||count>8*1024*1024||start>total||count>total-start)throw std::invalid_argument("Sole readback range invalid");
 auto output=[impl_->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];
 if(!output)throw std::runtime_error("Cannot allocate sole readback output");
 auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];
 if(!command||!encoder)throw std::runtime_error("Cannot create sole readback command");
 uint64_t p[4]={w->rows,w->columns,start,count};
 [encoder setComputePipelineState:impl_->kernels->soleRead];[encoder setBuffer:w->buffer offset:0 atIndex:0];
 [encoder setBuffer:output offset:0 atIndex:1];[encoder setBytes:p length:sizeof(p) atIndex:2];
 [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
 [command commit];[command waitUntilCompleted];
 if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Sole readback command failed");
 std::vector<uint32_t> result(count);std::memcpy(result.data(),output.contents,count*4);return result;
 }
#endif
}
}
''')
method='''    std::string SoleStatsJson() {
        const auto p=MetalContext::SoleStats();
        return nlohmann::json{{"active_weights",p.activeWeights},{"allocated_bytes",p.allocatedBytes},{"uploads",p.uploads},{"decode_encoded",p.decodeEncoded},{"restore_encoded",p.restoreEncoded},{"scratch_bytes",p.scratchBytes}}.dump();
    }
'''
replace(e,'    std::string PrimaryStatsJson() {',method+'    std::string PrimaryStatsJson() {')
needle='        APISET().AddFunc<0>("primary_storage_stats_json",&GarnetMetalAPI::PrimaryStatsJson);'
replace(e,needle,needle+'\n        APISET().AddFunc<0>("sole_interleaved_stats_json",&GarnetMetalAPI::SoleStatsJson);')
replace(e,'        return info;','''#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP
        info+="; sole_interleaved_mlp";
#endif
        return info;''')
replace(serv,'            before=json.loads(garnet.primary_storage_stats_json())','            before=json.loads(garnet.primary_storage_stats_json())\n            sole_before=json.loads(garnet.sole_interleaved_stats_json())')
replace(serv,'            after=json.loads(garnet.primary_storage_stats_json())','            after=json.loads(garnet.primary_storage_stats_json())\n            sole_after=json.loads(garnet.sole_interleaved_stats_json())')
replace(serv,"                result['timings']['primary_storage_before']=before","                result['timings']['sole_storage_before']=sole_before\n                result['timings']['sole_storage_after']=sole_after\n                result['timings']['sole_encoded_delta']={key:sole_after[key]-sole_before[key] for key in ['decode_encoded','restore_encoded']}\n                result['timings']['primary_storage_before']=before")
append('cmake/GarnetMetal.cmake','''
option(GARNET_METAL_SOLE_INTERLEAVED_MLP "Sole interleaved raw text MLP weights with bounded exact prefill restore" OFF)
if(GARNET_METAL_SOLE_INTERLEAVED_MLP)
 if(NOT GARNET_METAL_BODY_SHAPE)
  message(FATAL_ERROR "Sole interleaved MLP requires body-shape support")
 endif()
 foreach(policy GARNET_METAL_EXACT_HALF_PREFILL GARNET_METAL_GLOBAL_HALF_ALPHA GARNET_METAL_NATIVE_BFLOAT_BODY GARNET_METAL_CONSTANT_BF16_HEAD GARNET_METAL_SINGLE_CODER_HEAD GARNET_METAL_INTERLEAVED_BF16_BODY GARNET_METAL_COALESCED_BF16_BODY GARNET_METAL_FUSED_HEAD_ARGMAX GARNET_METAL_SCREENED_HEAD GARNET_METAL_FLOAT_DENSE_KEY_PREFILL GARNET_METAL_JOINED_PREFILL GARNET_METAL_JOINED_GLOBAL_MLP GARNET_METAL_MLP_PREFILL_GRAPH)
  if(${policy})
   message(FATAL_ERROR "Sole interleaved MLP excludes row-major-only auxiliary policy: ${policy}")
  endif()
 endforeach()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_SOLE_INTERLEAVED_MLP=1)
endif()
add_executable(garnet_sole_interleaved_test test2026/metal/sole_interleaved_test.cpp)
target_link_libraries(garnet_sole_interleaved_test PRIVATE garnet_metal)
add_test(NAME metal_sole_interleaved_reference COMMAND garnet_sole_interleaved_test "${CMAKE_CURRENT_BINARY_DIR}/sole_projection_words.bin")
''')
body=(old/'src/backends/metal/interleaved_bf16_body_shader.h').read_text().split('R"METAL(',1)[1].split(')METAL"',1)[0]
generic=body.replace('bf16_body_interleaved','sole_bf16_decode')
special=body.replace('bf16_body_interleaved','sole_specialized_bf16_decode')
for key in ['rows','inner','columns','aStride','oStride']:special=special.replace('p.'+key,'ulong(fixed_'+key+')')
constants=(old/'src/backends/metal/body_shape_shader.h').read_text().split('R"BODYSHAPE(',1)[1].split('kernel void',1)[0]
integer='''
kernel void sole_bf16_restore(device const ushort* input [[buffer(0)]],device ushort* out [[buffer(1)]],
 constant ulong2& shape [[buffer(2)]],uint id [[thread_position_in_grid]]) {
 ulong at=id;if(at>=shape.x*shape.y)return;
 ulong row=at/shape.y,k=at%shape.y;out[at]=input[(row/4)*shape.y*4+k*4+row%4];
}
kernel void sole_bf16_read(device const ushort* input [[buffer(0)]],device uint* out [[buffer(1)]],
 constant ulong4& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
 ulong i=id;if(i>=p.w)return;ulong at=p.z+i,row=at/p.y,k=at%p.y;
 out[i]=uint(input[(row/4)*p.y*4+k*4+row%4])<<16;
}
'''
shader='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+constants+generic+special+integer
(s/'src/backends/metal/sole_interleaved_mlp_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kSoleInterleavedMlpShader=R"SOLEMLP(\n'+shader+')SOLEMLP";\n}\n')
shutil.copy2(w/'sole_interleaved_test.cpp',s/'test2026/metal/sole_interleaved_test.cpp')
for name,ops in changes.items():
 text=(s/name).read_text()
 for op in reversed(ops):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'],(name,'inverse');text=text.replace(op['after'],op['before'])
 assert text==(old/name).read_text(),name
 if name.endswith('.py'):ast.parse((s/name).read_text())
pattern=r'R"([A-Za-z_]+)\((.*?)\)\1"';assert re.findall(pattern,(s/m).read_text(),re.S)==re.findall(pattern,(old/m).read_text(),re.S)
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()}
assert len(manifest)==1720 and all(manifest[k]==v for k,v in seed.items()if k not in changes)
(w/'seed-source-manifest.json').write_text(json.dumps(seed,indent=2)+'\n');(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'source-transformations.json').write_text(json.dumps(changes,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1718,current_files=1720,changed_or_added=sorted(k for k,v in manifest.items()if seed.get(k)!=v),original_shader_literals_graph_session_xModel_image_CUDA_tests_unchanged=True,new_shader_decode_generated_from_exact_seed_interleaved=True),indent=2)+'\n')
shutil.copytree(b/'work/worldsense-isolated-primary/build-profile',w/'build-profile')
print('WS63 isolated source1720/seven-file delta, inverse exact; no build/model/performance yet')
