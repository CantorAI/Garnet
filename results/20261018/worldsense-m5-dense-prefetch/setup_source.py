"""Implement prospective WS60 in a new immutable-seed experiment directory."""
from pathlib import Path
import json, hashlib, shutil, datetime, ast
w=Path(__file__).resolve().parent
r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-isolated-primary/source'
s=r/'out/research/worldsense-dense-prefetch/source'
m=json.loads((w.parent/'worldsense-isolated-primary/correctness-source-manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(m)==1718 and all(sha(old/k)==v for k,v in m.items())
assert not s.exists();s.mkdir(parents=True)
for k in m:
 p=s/k;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/k,p)
(w/'seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def change(rel,a,b):
 p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a[:100],t.count(a));p.write_text(t.replace(a,b,1))
header='''// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Original per-output BF16 expansion, accumulation and SIMD reduction order.
// Only immutable activation/weight reads are scheduled one iteration ahead.
inline constexpr const char* kDensePrefetchShader=R"DENSELOOK(
struct DenseLookaheadPacket { float4 x; ushort4 b0,b1,b2,b3; };
inline DenseLookaheadPacket dense_lookahead_load(device const float* a,device const ushort* weight,ulong column,ulong k) {
    DenseLookaheadPacket value;
    value.x=*(device const float4*)(a+k);
    value.b0=*(device const ushort4*)(weight+column*ulong(fixed_inner)+k);
    value.b1=ushort4(0);value.b2=ushort4(0);value.b3=ushort4(0);
    if(fixed_no_tail||column+1<ulong(fixed_columns))value.b1=*(device const ushort4*)(weight+(column+1)*ulong(fixed_inner)+k);
    if(fixed_no_tail||column+2<ulong(fixed_columns))value.b2=*(device const ushort4*)(weight+(column+2)*ulong(fixed_inner)+k);
    if(fixed_no_tail||column+3<ulong(fixed_columns))value.b3=*(device const ushort4*)(weight+(column+3)*ulong(fixed_inner)+k);
    return value;
}
kernel void dense_prefetched_bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    ulong k=ulong(lane)*4;
    DenseLookaheadPacket current=dense_lookahead_load(a,weight,column,k);
    float4 sum0=0.0f;
    for(;k<ulong(fixed_inner);k+=128) {
        DenseLookaheadPacket next={float4(0.0f),ushort4(0),ushort4(0),ushort4(0),ushort4(0)};
        if(k+128<ulong(fixed_inner))next=dense_lookahead_load(a,weight,column,k+128);
        for(uint j=0;j<4;++j) {
            float4 w=as_type<float4>(uint4(current.b0[j],current.b1[j],current.b2[j],current.b3[j])<<16);
            sum0+=current.x[j]*w;
        }
        current=next;
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    if(lane==0)for(uint j=0;j<4&&(fixed_no_tail||column+j<ulong(fixed_columns));++j)out[column+j]=sum0[j];
}
)DENSELOOK";
}
'''
(s/'src/backends/metal/dense_prefetch_shader.h').write_text(header)
rel='src/backends/metal/metal_context.mm'
change(rel,'#include "body_shape_shader.h"','#include "body_shape_shader.h"\n#ifdef GARNET_METAL_DENSE_PREFETCH\n#include "dense_prefetch_shader.h"\n#endif')
change(rel,'PrimaryCounters primaryCounters;','PrimaryCounters primaryCounters;\n#ifdef GARNET_METAL_DENSE_PREFETCH\nstd::atomic<uint64_t> densePrefetchEncoded{0};\n#endif')
change(rel,'    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> bodyShape;','    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> bodyShape;\n#ifdef GARNET_METAL_DENSE_PREFETCH\n    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> densePrefetchBodyShape;\n#endif')
change(rel,'        source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];','        source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];\n#ifdef GARNET_METAL_DENSE_PREFETCH\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kDensePrefetchShader]];\n#endif')
change(rel,'        auto found=selectedCache.find(key);if(found!=selectedCache.end())return found->second!=nil;','''#ifdef GARNET_METAL_DENSE_PREFETCH
        if(!primary&&a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256&&o.columns<100000) {
            auto& lookahead=impl_->kernels->densePrefetchBodyShape;
            if(lookahead.find(key)==lookahead.end()&&lookahead.size()<128) {
                uint32_t values[5];for(size_t i=0;i<5;++i)values[i]=uint32_t(key[i]);
                auto constants=[MTLFunctionConstantValues new];
                for(NSUInteger i=0;i<5;++i)[constants setConstantValue:&values[i] type:MTLDataTypeUInt atIndex:i];
                bool noTail=o.columns%16==0;[constants setConstantValue:&noTail type:MTLDataTypeBool atIndex:5];
                NSError* error=nil;
                auto function=[impl_->kernels->library newFunctionWithName:@"dense_prefetched_bf16_decode" constantValues:constants error:&error];
                id<MTLComputePipelineState> pipeline=nil;
                if(function)pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
                if(pipeline&&(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128))pipeline=nil;
                lookahead.emplace(key,pipeline);
            }
        }
#endif
        auto found=selectedCache.find(key);if(found!=selectedCache.end())return found->second!=nil;''')
change(rel,'size_t MetalContext::LastBodyShapeDispatches() const {return impl_->bodyShapeDispatches;}','''uint64_t MetalContext::DensePrefetchEncoded() {
#ifdef GARNET_METAL_DENSE_PREFETCH
 return densePrefetchEncoded.load();
#else
 return 0;
#endif
}
size_t MetalContext::LastBodyShapeDispatches() const {return impl_->bodyShapeDispatches;}''')
change(rel,'        size_t decodeColumns=16;','        size_t decodeColumns=16;\n#ifdef GARNET_METAL_DENSE_PREFETCH\n        bool usingDensePrefetch=false;\n#endif')
change(rel,'            } else ++impl_->bodyShapeMisses;','''            } else ++impl_->bodyShapeMisses;
#ifdef GARNET_METAL_DENSE_PREFETCH
            if(!primary&&a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256) {
                auto lookahead=impl_->kernels->densePrefetchBodyShape.find(key);
                if(lookahead!=impl_->kernels->densePrefetchBodyShape.end()&&lookahead->second) {
                    pipeline=lookahead->second;usingDensePrefetch=true;
                    threads=128;decodeColumns=16;
                }
            }
#endif''')
needle='        [encoder dispatchThreadgroups:a.rows<=2?MTLSizeMake((o.columns+decodeColumns-1)/decodeColumns,1,1):MTLSizeMake((o.columns+tileColumns-1)/tileColumns,(o.rows+tileRows-1)/tileRows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];'
change(rel,needle,needle+'\n#ifdef GARNET_METAL_DENSE_PREFETCH\n        if(usingDensePrefetch)++densePrefetchEncoded;\n#endif')
change('src/backends/metal/metal_context.h','    size_t LastBodyShapeDispatches() const;','    static uint64_t DensePrefetchEncoded();\n    size_t LastBodyShapeDispatches() const;')
rel='src/entry/metal_main.cpp'
change(rel,'    std::string PrimaryStatsJson() {','    std::string DensePrefetchStatsJson() {return nlohmann::json{{"encoded",MetalContext::DensePrefetchEncoded()}}.dump();}\n    std::string PrimaryStatsJson() {')
change(rel,'        APISET().AddFunc<0>("primary_storage_stats_json",&GarnetMetalAPI::PrimaryStatsJson);','        APISET().AddFunc<0>("primary_storage_stats_json",&GarnetMetalAPI::PrimaryStatsJson);\n        APISET().AddFunc<0>("dense_prefetch_stats_json",&GarnetMetalAPI::DensePrefetchStatsJson);')
change(rel,'        return info;\n    }\n    // Graphs own','#ifdef GARNET_METAL_DENSE_PREFETCH\n        info+="; dense_raw_decode_prefetch";\n#endif\n        return info;\n    }\n    // Graphs own')
rel='runtime/metal_serving.py'
change(rel,'_models = {}',"_native_dense_prefetch = 'dense_raw_decode_prefetch' in garnet.backend_info()\n_models = {}")
change(rel,'            before=json.loads(garnet.primary_storage_stats_json())','            before=json.loads(garnet.primary_storage_stats_json())\n            dense_before=json.loads(garnet.dense_prefetch_stats_json())[\'encoded\'] if _native_dense_prefetch else 0')
change(rel,"                result['timings']['primary_storage_before']=before","                result['timings']['dense_prefetch_encoded_delta']=(json.loads(garnet.dense_prefetch_stats_json())['encoded']-dense_before) if _native_dense_prefetch else 0\n                result['timings']['primary_storage_before']=before")
ast.parse((s/rel).read_text())
p=s/'cmake/GarnetMetal.cmake'
p.write_text(p.read_text()+'''
option(GARNET_METAL_DENSE_PREFETCH "Ordered one-step lookahead for raw2048/6144 dense body decode" OFF)
if(GARNET_METAL_DENSE_PREFETCH)
 if(NOT GARNET_METAL_BODY_SHAPE)
  message(FATAL_ERROR "Dense prefetch requires body-shape specialization")
 endif()
 foreach(policy GARNET_METAL_NATIVE_BFLOAT_BODY GARNET_METAL_INTERLEAVED_BF16_BODY GARNET_METAL_COALESCED_BF16_BODY)
  if(${policy})
   message(FATAL_ERROR "Dense prefetch excludes later raw body override: ${policy}")
  endif()
 endforeach()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DENSE_PREFETCH=1)
endif()
add_executable(garnet_dense_prefetch_test test2026/metal/dense_prefetch_test.cpp)
target_link_libraries(garnet_dense_prefetch_test PRIVATE garnet_metal)
add_test(NAME metal_dense_prefetch_reference COMMAND garnet_dense_prefetch_test ${CMAKE_BINARY_DIR}/dense_prefetch_projection_words.bin)
''')
t=(s/'test2026/metal/primary_projection_test.cpp').read_text()
t=t.replace('auto s=MetalContext::PrimaryStats();return s.decodeEncoded+s.headEncoded+s.prefillEncoded;','return MetalContext::DensePrefetchEncoded();')
t=t.replace('gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner)','gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,false)')
t=t.replace('#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS\n  eligible=columns>256&&columns%4==0&&inner%128==0;','#ifdef GARNET_METAL_DENSE_PREFETCH\n  eligible=rows==1&&(inner==2048||inner==6144)&&columns>256&&columns<100000;')
t=t.replace('require(w.PrimaryReady()==eligible,"Primary admission mismatch");','require(!w.PrimaryReady(),"Dense prefetch unit requires forced raw storage");')
t=t.replace('#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS\n require(widePasses>0', '#ifdef GARNET_METAL_DENSE_PREFETCH\n require(widePasses>0')
t=t.replace('Primary numerical/reference/poison/input/tail272/fallback/small/vocab/queued owner/Abort recovery PASS','Forced-raw dense prefetch numerical/reference/poison/input/tail272/fallback/small/vocab/queued owner/Abort recovery PASS')
(s/'test2026/metal/dense_prefetch_test.cpp').write_text(t)
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()}
changed=sorted(k for k,v in manifest.items()if m.get(k)!=v)
assert len(manifest)==1720 and len(changed)==7
assert all(manifest[k]==v for k,v in m.items()if k not in changed)
assert sha(s/'src/backends/metal/body_shape_shader.h')==m['src/backends/metal/body_shape_shader.h']
assert sha(s/'test2026/metal/primary_projection_test.cpp')==m['test2026/metal/primary_projection_test.cpp']
(w/'implementation-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(w/'implementation-audit.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed_files=1718,source_files=1720,changed_files=changed,all_other_source_exact54=True,original_body_shader_and_projection_test_exact=True,no_build_or_model_run_yet=True),indent=2)+'\n')
print('WS60 isolated source1720/seven deltas implemented; original tests/shaders/CUDA retained')
