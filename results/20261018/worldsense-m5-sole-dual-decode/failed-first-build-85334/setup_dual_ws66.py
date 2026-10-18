from pathlib import Path
import json,hashlib,shutil,re,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-sole-dual-decode';old=r/'out/research/worldsense-direct-interleaved-prefill/source';s=r/'out/research/worldsense-sole-dual-decode/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((w/'protocol.json').read_text())['design_sha256']==sha(w/'native-design-before-implementation.txt');seed=json.loads((w/'read-only-dual-consumer-audit.json').read_text())['seed_manifest'];assert all(sha(old/k)==v for k,v in seed.items());s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
ops={}
def change(rel,a,z,count=1):
 p=s/rel;t=p.read_text();assert t.count(a)==count,(rel,a,t.count(a));p.write_text(t.replace(a,z));ops.setdefault(rel,[]).append(dict(before=a,after=z,count=count))
def append(rel,z):
 p=s/rel;p.write_text(p.read_text()+z);ops.setdefault(rel,[]).append(dict(before='',after=z,count=1,append=True))
h='src/backends/metal/metal_context.h';m='src/backends/metal/metal_context.mm';g='src/backends/metal/compiled_graph.cpp';e='src/entry/metal_main.cpp';runtime='runtime/metal_serving.py'
change(h,'class MetalContext {','struct MetalDualProjectionStats {uint64_t encodedPairs=0,representedProjections=0;};\nclass MetalContext {')
change(h,'    static MetalPrimaryStats PrimaryStats();','''    static MetalPrimaryStats PrimaryStats();
    static MetalDualProjectionStats DualProjectionStats();
    bool EncodeSoleDualProjectionDecode(const MetalMatrix&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalMatrix&,const MetalMatrix&);
    std::vector<uint32_t> SoleDualProjectionReadWords(const MetalBF16Weight&,const MetalBF16Weight&,size_t firstBlock,size_t countBlocks);
    std::string DualProjectionShaderProgramSource() const;
#if defined(GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
    static void DualProjectionTestUnavailable(bool);
    static void DualProjectionTestFailNext(bool);
#endif''')
change(m,'#include "sole_interleaved_mlp_shader.h"','#include "sole_interleaved_mlp_shader.h"\n#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE\n#include "sole_dual_projection_shader.h"\n#endif')
change(m,'SoleCounters soleCounters;','''SoleCounters soleCounters;
#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
std::atomic<uint64_t> soleDualPairs{0},soleDualProjections{0};
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
std::atomic<bool> soleDualUnavailable{false},soleDualFailNext{false};
#endif
#endif''')
change(m,'    std::string soleProgram;','''    std::string soleProgram;
#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
    id<MTLLibrary> dualLibrary=nil;id<MTLComputePipelineState> dualB1=nil,dualB2=nil,dualRead=nil;
    std::string dualProgram;
#endif''')
needle='#ifdef GARNET_METAL_PRIMARY_VECTOR_PREFILL\n        id<MTLLibrary> prefillLibrary=library;'
extra='''#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
        impl_->kernels->dualProgram=impl_->kernels->soleProgram+kSoleDualProjectionShader;
        auto dualLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:impl_->kernels->dualProgram.c_str()] options:compileOptions error:&error];
        if(!dualLibrary)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        auto fitsDual=[&](id<MTLComputePipelineState> p){return p&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=128&&p.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength;};
        id<MTLComputePipelineState> dual[2]={nil,nil};
        for(uint32_t rows=1;rows<=2;++rows){
            uint32_t values[5]={rows,2048,6144,2048,6144};auto constants=[MTLFunctionConstantValues new];
            for(NSUInteger i=0;i<5;++i)[constants setConstantValue:&values[i] type:MTLDataTypeUInt atIndex:i];
            bool noTail=true;[constants setConstantValue:&noTail type:MTLDataTypeBool atIndex:5];
            auto fn=[dualLibrary newFunctionWithName:@"sole_dual_specialized_decode" constantValues:constants error:&error];
            if(!fn)throw std::runtime_error("Cannot specialize sole dual decode");
            dual[rows-1]=[impl_->device newComputePipelineStateWithFunction:fn error:&error];
            if(!dual[rows-1])throw std::runtime_error("Cannot create sole dual decode pipeline");
        }
        auto read=[impl_->device newComputePipelineStateWithFunction:[dualLibrary newFunctionWithName:@"sole_dual_read_pair"] error:&error];
        if(!read)throw std::runtime_error("Cannot create sole dual word reader");
        if(fitsDual(dual[0])&&fitsDual(dual[1])&&fitsDual(read)){
            impl_->kernels->dualLibrary=dualLibrary;impl_->kernels->dualB1=dual[0];impl_->kernels->dualB2=dual[1];impl_->kernels->dualRead=read;
        }
#endif
''';change(m,needle,extra+needle)
# New atomic two-output method, diagnostics and no allocation in serving path.
append(m,'''
namespace Garnet {
MetalDualProjectionStats MetalContext::DualProjectionStats(){
#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
 return {soleDualPairs.load(),soleDualProjections.load()};
#else
 return {};
#endif
}
std::string MetalContext::DualProjectionShaderProgramSource()const{
#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
 return impl_->kernels->dualProgram;
#else
 return {};
#endif
}
#if defined(GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
void MetalContext::DualProjectionTestUnavailable(bool v){soleDualUnavailable.store(v);}
void MetalContext::DualProjectionTestFailNext(bool v){soleDualFailNext.store(v);}
#endif
bool MetalContext::EncodeSoleDualProjectionDecode(const MetalMatrix& input,const MetalBF16Weight& first,const MetalBF16Weight& second,const MetalMatrix& outputA,const MetalMatrix& outputB){
#ifndef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
 (void)input;(void)first;(void)second;(void)outputA;(void)outputB;return false;
#else
 @autoreleasepool {
 auto g=first.impl_,u=second.impl_;
 auto eligible=[](const auto& w){return w&&w->soleInterleaved&&w->buffer&&!w->floatPrefill&&!w->visionPrefill&&w->rows==6144&&w->columns==2048&&w->buffer.length==6144ull*2048*2;};
 if(!eligible(g)||!eligible(u)||!input.impl_||!outputA.impl_||!outputB.impl_)return false;
 auto a=input.impl_->matrix,ga=outputA.impl_->matrix,ua=outputB.impl_->matrix;
 if(a.rows<1||a.rows>2||a.columns!=2048||ga.rows!=a.rows||ua.rows!=a.rows||ga.columns!=6144||ua.columns!=6144||a.rowBytes!=2048*4||ga.rowBytes!=6144*4||ua.rowBytes!=6144*4)return false;
 auto pipeline=a.rows==1?impl_->kernels->dualB1:impl_->kernels->dualB2;
 if(!pipeline)return false;
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
 if(soleDualUnavailable.load())return false;
#endif
 if(!impl_->command)throw std::invalid_argument("Sole dual decode requires active batch");
 if(a.device!=impl_->device||ga.device!=impl_->device||ua.device!=impl_->device||g->buffer.device!=impl_->device||u->buffer.device!=impl_->device)
  throw std::invalid_argument("Sole dual decode device mismatch");
 if(ga.data==ua.data||ga.data==a.data||ua.data==a.data||ga.data==g->buffer||ga.data==u->buffer||ua.data==g->buffer||ua.data==u->buffer)
  throw std::invalid_argument("Sole dual decode output alias");
 if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))throw std::logic_error("Sole dual decode uninitialized or foreign pending input");
 for(const auto* out:{&outputA,&outputB})if(out->impl_->pending&&!impl_->pendingOutputs.count(out->impl_.get()))throw std::logic_error("Sole dual decode foreign pending output");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
 if(soleDualFailNext.exchange(false))throw std::runtime_error("Injected sole dual predispatch failure");
#endif
 struct Params{uint64_t rows,inner,columns,aStride,oStride;};Params p{a.rows,2048,6144,2048,6144};
 auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Cannot create sole dual encoder");
 [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:g->buffer offset:0 atIndex:1];
 [encoder setBuffer:ga.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder setBuffer:u->buffer offset:0 atIndex:4];[encoder setBuffer:ua.data offset:0 atIndex:5];
 [encoder dispatchThreadgroups:MTLSizeMake(6144/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
 for(const auto* out:{&outputA,&outputB}){out->impl_->ready=false;out->impl_->pending=true;if(impl_->pendingOutputs.insert(out->impl_.get()).second)impl_->outputs.push_back(out->impl_);}
 soleDualPairs.fetch_add(1);soleDualProjections.fetch_add(2);return true;
 }
#endif
}
std::vector<uint32_t> MetalContext::SoleDualProjectionReadWords(const MetalBF16Weight& first,const MetalBF16Weight& second,size_t start,size_t count){
#ifndef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
 (void)first;(void)second;(void)start;(void)count;throw std::logic_error("Sole dual word reader disabled");
#else
 @autoreleasepool {
 if(impl_->command||impl_->submitted.count)throw std::logic_error("Sole dual reader requires closed drained batch");
 for(const auto* item:{&first,&second}){auto w=item->impl_;if(!w||!w->soleInterleaved||!w->buffer||w->buffer.device!=impl_->device||w->rows!=6144||w->columns!=2048||w->buffer.length!=6144ull*2048*2)throw std::invalid_argument("Sole dual reader requires exact same-device sole weights");}
 const size_t total=6144*2048/16;if(!count||count>8*1024*1024/32||start>total||count>total-start)throw std::invalid_argument("Sole dual reader range");
 if(!impl_->kernels->dualRead)throw std::runtime_error("Sole dual reader unavailable");
 auto output=[impl_->device newBufferWithLength:count*32*4 options:MTLResourceStorageModeShared];if(!output)throw std::runtime_error("Sole dual reader allocation failed");
 auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];if(!command||!encoder)throw std::runtime_error("Sole dual reader command failed");uint64_t p[2]={start,count};
 [encoder setComputePipelineState:impl_->kernels->dualRead];[encoder setBuffer:first.impl_->buffer offset:0 atIndex:0];[encoder setBuffer:second.impl_->buffer offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:p length:sizeof(p) atIndex:3];
 [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Sole dual reader GPU failed");
 std::vector<uint32_t> result(count*32);std::memcpy(result.data(),output.contents,result.size()*4);return result;
 }
#endif
}
}
''')
needle='        if(op.kind==Kind::MatMul) {\n            if(slots_[op.b].bf16.Valid())'
block='''#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE
        if(!profileOps&&i+1<ops_.size()&&(packedOutput||residentOps_[i+1])){
            const auto& next=ops_[i+1];
            auto projection=[&](const Op& value){return (value.kind==Kind::BF16MatVec||(value.kind==Kind::MatMul&&value.transpose))&&slots_[value.b].bf16.Valid();};
            if(projection(op)&&projection(next)&&op.a==next.a&&op.out!=next.out&&
               gpu.EncodeSoleDualProjectionDecode(a,slots_[op.b].bf16,slots_[next.b].bf16,out,slots_[next.out].matrix)){
                ++i;continue;
            }
        }
#endif
''';change(g,needle,block+needle)
change(e,'    std::string SoleStatsJson() {','''    std::string DualProjectionStatsJson(){const auto p=MetalContext::DualProjectionStats();return nlohmann::json{{"encoded_pairs",p.encodedPairs},{"represented_projections",p.representedProjections}}.dump();}
    std::string SoleStatsJson() {''')
needle='        APISET().AddFunc<0>("sole_interleaved_stats_json",&GarnetMetalAPI::SoleStatsJson);';change(e,needle,needle+'\n        APISET().AddFunc<0>("sole_dual_projection_stats_json",&GarnetMetalAPI::DualProjectionStatsJson);')
needle='        info+="; sole_direct_interleaved_prefill";\n#endif';change(e,needle,needle+'\n#ifdef GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE\n        info+="; sole_dual_projection_decode";\n#endif')
change(runtime,'            sole_before=json.loads(garnet.sole_interleaved_stats_json())','            sole_before=json.loads(garnet.sole_interleaved_stats_json())\n            dual_before=json.loads(garnet.sole_dual_projection_stats_json())')
change(runtime,'            sole_after=json.loads(garnet.sole_interleaved_stats_json())','            sole_after=json.loads(garnet.sole_interleaved_stats_json())\n            dual_after=json.loads(garnet.sole_dual_projection_stats_json())')
change(runtime,"                result['timings']['sole_storage_before']=sole_before","                result['timings']['dual_projection_before']=dual_before\n                result['timings']['dual_projection_after']=dual_after\n                result['timings']['dual_projection_delta']={key:dual_after[key]-dual_before[key] for key in ['encoded_pairs','represented_projections']}\n                result['timings']['sole_storage_before']=sole_before")
append('cmake/GarnetMetal.cmake','''
option(GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE "Share activation loads for exact sole gate/up decode, retaining separate complete outputs" OFF)
if(GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE)
 if(NOT GARNET_METAL_SOLE_INTERLEAVED_MLP OR NOT GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL)
  message(FATAL_ERROR "Sole dual decode requires sole MLP storage and direct prefill")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE=1)
endif()
add_executable(garnet_sole_dual_test test2026/metal/sole_dual_projection_test.cpp)
target_link_libraries(garnet_sole_dual_test PRIVATE garnet_metal)
add_test(NAME metal_sole_dual_reference COMMAND garnet_sole_dual_test "${CMAKE_CURRENT_BINARY_DIR}/sole_dual_projection_words.bin")
''')
# Generate exact original per-projection arithmetic, adding a second set of independent accumulators.
sole=(old/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0]
start=sole.index('kernel void sole_specialized_bf16_decode(');end=sole.index('kernel void sole_bf16_restore(',start);original=sole[start:end].rstrip();dual=original;shaderops=[]
def sc(a,z,count=1):
 global dual
 assert dual.count(a)==count,(a,dual.count(a));dual=dual.replace(a,z);shaderops.append(dict(before=a,after=z,count=count))
sc('sole_specialized_bf16_decode','sole_dual_specialized_decode')
sc('constant BF16Params& p [[buffer(3)]],','constant BF16Params& p [[buffer(3)]],device const ushort* upweight [[buffer(4)]],device float* upout [[buffer(5)]],')
sc('float4 sum0=0.0f,sum1=0.0f;','float4 sum0=0.0f,sum1=0.0f;float4 up0=0.0f,up1=0.0f;')
a='''            uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
            ushort4 words[4]={as_type<ushort4>(pair0.xy),as_type<ushort4>(pair0.zw),as_type<ushort4>(pair1.xy),as_type<ushort4>(pair1.zw)};'''
sc(a,'            SoleDualWords gateWords=sole_dual_words(weight,packed),upWords=sole_dual_words(upweight,packed);')
sc('uint4(words[j])','uint4(gateWords.words[j])')
a='                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;';sc(a,a+'\n                float4 uw=as_type<float4>(uint4(upWords.words[j])<<16);up0+=x[j]*uw;if(ulong(fixed_rows)==2)up1+=y[j]*uw;')
# Keep the complete original odd-K fallback algebra too, though exact admitted K2048 always takes vector path.
a='''        if(column+3<ulong(fixed_columns))w.w=as_type<float>(uint(weight[(column+3)*ulong(fixed_inner)+k])<<16);''';up='''        float4 uw=0.0f;
        uw.x=as_type<float>(uint(upweight[column*ulong(fixed_inner)+k])<<16);
        if(column+1<ulong(fixed_columns))uw.y=as_type<float>(uint(upweight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(column+2<ulong(fixed_columns))uw.z=as_type<float>(uint(upweight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(column+3<ulong(fixed_columns))uw.w=as_type<float>(uint(upweight[(column+3)*ulong(fixed_inner)+k])<<16);''';sc(a,a+'\n'+up)
a='        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;';sc(a,a+'\n        up0+=x*uw;if(ulong(fixed_rows)==2)up1+=y*uw;')
a='    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));';sc(a,a+'\n    up0=float4(simd_sum(up0.x),simd_sum(up0.y),simd_sum(up0.z),simd_sum(up0.w));\n    up1=float4(simd_sum(up1.x),simd_sum(up1.y),simd_sum(up1.z),simd_sum(up1.w));')
a='out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];';sc(a,a+'upout[column+j]=up0[j];if(ulong(fixed_rows)==2)upout[ulong(fixed_oStride)+column+j]=up1[j];')
inverse=dual
for op in reversed(shaderops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==original
helper='''struct SoleDualWords {ushort4 words[4];};
inline SoleDualWords sole_dual_words(device const ushort* weight,ulong packed){
 uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
 SoleDualWords value;value.words[0]=as_type<ushort4>(pair0.xy);value.words[1]=as_type<ushort4>(pair0.zw);value.words[2]=as_type<ushort4>(pair1.xy);value.words[3]=as_type<ushort4>(pair1.zw);return value;
}
'''
probe='''
kernel void sole_dual_read_pair(device const ushort* first [[buffer(0)]],device const ushort* second [[buffer(1)]],device uint* output [[buffer(2)]],constant ulong2& p [[buffer(3)]],uint id [[thread_position_in_grid]]){
 ulong i=id;if(i>=p.y)return;ulong packed=(p.x+i)*16;SoleDualWords g=sole_dual_words(first,packed),u=sole_dual_words(second,packed);
 for(uint k=0;k<4;++k)for(uint c=0;c<4;++c){output[i*32+k*4+c]=uint(g.words[k][c])<<16;output[i*32+16+k*4+c]=uint(u.words[k][c])<<16;}
}
'''
(s/'src/backends/metal/sole_dual_projection_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n#pragma once\nnamespace Garnet {\ninline constexpr const char* kSoleDualProjectionShader=R"SOLEDUAL(\n'+helper+dual+'\n'+probe+')SOLEDUAL";\n}\n')
(w/'dual-shader-inverse-proof.json').write_text(json.dumps(dict(original_source=original,original_sha256=hashlib.sha256(original.encode()).hexdigest(),dual_source=dual,dual_sha256=hashlib.sha256(dual.encode()).hexdigest(),operations=shaderops,original_per_projection_arithmetic_and_stores_inverse_exact=True,probe_uses_same_integer_vector_load_helper=True),indent=2)+'\n')
for rel,records in ops.items():
 inverse=(s/rel).read_text()
 for op in reversed(records):
  if op.get('append'):assert inverse.endswith(op['after']);inverse=inverse[:-len(op['after'])]
  else:assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse==(old/rel).read_text(),rel
 if rel.endswith('.py'):ast.parse((s/rel).read_text())
(w/'seed-source-manifest.json').write_text(json.dumps(seed,indent=2)+'\n');(w/'source-transformations.json').write_text(json.dumps(ops,indent=2)+'\n');shutil.copytree(b/'work/worldsense-direct-interleaved-prefill/build-profile',w/'build-profile')
print('WS66 isolated host/graph/telemetry/shader implementation prepared; new native test and final source freeze still pending, no build/model/speed')
