from pathlib import Path
import shutil,re,json,time,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/Garnet');s=Path('/Users/shawn/CantorAI/out/research/worldsense-coalesced-prefill/source');s.mkdir(parents=True,exist_ok=False)
for d in ['src','cmake','runtime','xModel']:shutil.copytree(r/d,s/d,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(r/'CMakeLists.txt',s/'CMakeLists.txt');(s/'third_party').mkdir();shutil.copytree(r/'third_party/nlohmann',s/'third_party/nlohmann');(s/'test2026/metal').mkdir(parents=True)
for p in (r/'test2026/metal').iterdir():
 if p.is_file()and p.suffix in ['.cpp','.mm','.h','.py']:shutil.copy2(p,s/'test2026/metal'/p.name)
p=s/'src/backends/metal/metal_context.mm';text=p.read_text();kernels=[]
for name,k in [('gemm_16_64_64',64),('gemm_32_64_32',32)]:
 start=text.index('kernel void '+name+'(');end=text.index('\nkernel void ',start+1);part=text[start:end]
 part=part.replace('struct ArgmaxParams {ulong rows,columns,iStride,oStride;};','')
 part=part.replace(name,'coalesced_'+name).replace('right['+str(k*64)+']','right['+str(k*65)+']')
 old='for(uint i=tid;i<'+str(k*64)+';i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}'
 new='for(uint i=tid;i<'+str(k*64)+';i+=256){ulong k=base+i%'+str(k)+',column=colBase+i/'+str(k)+';right[(i%'+str(k)+')*65+i/'+str(k)+']=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}'
 assert old in part;part=part.replace(old,new).replace('right+offset*64+simd*8,64','right+offset*65+simd*8,65');kernels.append(part)
(s/'src/backends/metal/coalesced_prefill_shader.h').write_text('// SPDX-License-Identifier: Apache-2.0\n// Original FP32 MMA order; coalesced BF16 reads and padded shared transpose.\n#pragma once\nnamespace Garnet { inline constexpr const char* kCoalescedPrefillShader=R"COALESCED(\n'+''.join(kernels)+')COALESCED"; }\n')
text=text.replace('#include "exact_half_codec.h"','#include "exact_half_codec.h"\n#ifdef GARNET_METAL_COALESCED_PREFILL\n#include "coalesced_prefill_shader.h"\n#endif')
text=text.replace('    id<MTLComputePipelineState> argmax;','''#ifdef GARNET_METAL_COALESCED_PREFILL
    id<MTLComputePipelineState> coalescedPrefill16,coalescedPrefill32;
#endif
    id<MTLComputePipelineState> argmax;''')
text=text.replace('    size_t coalescedBlockDispatches=0;','    size_t coalescedBlockDispatches=0;')
# Append independent members before a stable existing counter declaration.
needle='    size_t globalHalfAlphaDispatches=0;';assert needle in text;text=text.replace(needle,'    size_t coalescedPrefillDispatches=0;bool coalescedPrefillEnabled=true;\n'+needle)
text=text.replace('impl_->coalescedBlockDispatches=0;','impl_->coalescedPrefillDispatches=0;impl_->coalescedBlockDispatches=0;')
needle='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];';assert needle in text
text=text.replace(needle,needle+'''
#ifdef GARNET_METAL_COALESCED_PREFILL
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kCoalescedPrefillShader]];
#endif''')
needle='        impl_->kernels->bf16Gemm16Wide=';idx=text.index(needle);text=text[:idx]+'''#ifdef GARNET_METAL_COALESCED_PREFILL
        impl_->kernels->coalescedPrefill16=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"coalesced_gemm_16_64_64"] error:&error];
        impl_->kernels->coalescedPrefill32=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"coalesced_gemm_32_64_32"] error:&error];
        if(!impl_->kernels->coalescedPrefill16||!impl_->kernels->coalescedPrefill32)throw std::runtime_error("Cannot create coalesced prefill pipelines");
#endif
'''+text[idx:]
needle='size_t MetalContext::LastCoalescedBlockDispatches() const';idx=text.index(needle);text=text[:idx]+'''size_t MetalContext::LastCoalescedPrefillDispatches() const {return impl_->coalescedPrefillDispatches;}
bool MetalContext::SetCoalescedPrefill(bool enabled) {
    if(impl_->command)throw std::logic_error("Change prefill policy outside a command batch");
#ifdef GARNET_METAL_COALESCED_PREFILL
    impl_->coalescedPrefillEnabled=enabled;return true;
#else
    (void)enabled;return false;
#endif
}
'''+text[idx:]
needle='        auto widePipeline=a.rows>16?impl_->kernels->bf16Gemm32Wide:impl_->kernels->bf16Gemm16Wide;';assert needle in text;text=text.replace(needle,needle+'''
#ifdef GARNET_METAL_COALESCED_PREFILL
        if(impl_->coalescedPrefillEnabled)widePipeline=a.rows>16?impl_->kernels->coalescedPrefill32:impl_->kernels->coalescedPrefill16;
#endif''')
needle='        auto pipeline=a.rows<=2?impl_->kernels->bf16Decode:wide?widePipeline:impl_->kernels->bf16Gemm;';assert needle in text;text=text.replace(needle,'''#ifdef GARNET_METAL_COALESCED_PREFILL
        if(wide&&impl_->coalescedPrefillEnabled)++impl_->coalescedPrefillDispatches;
#endif
'''+needle);p.write_text(text)
p=s/'src/backends/metal/metal_context.h';text=p.read_text().replace('    size_t LastCoalescedBlockDispatches() const;','    size_t LastCoalescedBlockDispatches() const;\n    size_t LastCoalescedPrefillDispatches() const;\n    bool SetCoalescedPrefill(bool enabled);');p.write_text(text)
p=s/'src/backends/metal/decoder_session.h';text=p.read_text().replace('    size_t BodyShapePipelineCount()', '    size_t BodyShapePipelineCount()')
needle='    double LastInferenceMs() const';idx=text.index(needle);text=text[:idx]+'''    size_t coalescedPrefillDispatches_=0;
    size_t LastCoalescedPrefillDispatches() const {return coalescedPrefillDispatches_;}
'''+text[idx:];text=text.replace('        APISET().AddProp("coalesced_block_dispatches"','        APISET().AddProp("coalesced_prefill_dispatches",&MetalDecoderSession::LastCoalescedPrefillDispatches);\n        APISET().AddProp("coalesced_block_dispatches"');p.write_text(text)
p=s/'src/backends/metal/decoder_session.cpp';text=p.read_text().replace('coalescedBlockDispatches_=0;','coalescedPrefillDispatches_=0;coalescedBlockDispatches_=0;').replace('coalescedBlockDispatches_=gpu_->LastCoalescedBlockDispatches();','coalescedBlockDispatches_=gpu_->LastCoalescedBlockDispatches();coalescedPrefillDispatches_=gpu_->LastCoalescedPrefillDispatches();');p.write_text(text)
p=s/'src/entry/metal_main.cpp';text=p.read_text();idx=text.index('#ifdef GARNET_METAL_BODY_SHAPE');text=text[:idx]+'''#ifdef GARNET_METAL_COALESCED_PREFILL
        info+="; coalesced_prefill";
#endif
'''+text[idx:];p.write_text(text)
p=s/'xModel/qwen35/vl_0_8b/metal_decoder.py';text=p.read_text().replace("        if 'mlp_prefill_graph' in garnet.backend_info():", "        if 'coalesced_prefill' in garnet.backend_info():self.prefill_profile['coalesced_prefill_dispatches']=0\n        if 'mlp_prefill_graph' in garnet.backend_info():");p.write_text(text)
p=s/'cmake/GarnetMetal.cmake';text=p.read_text().replace('add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)','''add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)
option(GARNET_METAL_COALESCED_PREFILL "Coalesce original BF16 wide-tile loads with padded shared transpose" OFF)
if(GARNET_METAL_COALESCED_PREFILL)
    if(NOT GARNET_METAL_BODY_SHAPE OR GARNET_METAL_EXACT_HALF_PREFILL OR GARNET_METAL_DENSE_KEY_SPLIT_K)
        message(FATAL_ERROR "Coalesced prefill screen requires original BF16 storage/body profile")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_COALESCED_PREFILL=1)
endif()''');p.write_text(text)
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-04','status':'prepared, uncompiled/unmeasured','declared_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hypothesis':'Coalesced contiguous-K BF16 loads plus 65-column threadgroup transpose reduce global transaction dispersion/bank conflicts for existing 16x64x64 and 32x64x32 FP32 MMA tiles. Same values, K reduction order and original decode path.','tradeoffs':'Extra 256/128 bytes of shared memory per wide tile; transpose stores and SIMD stride may cost latency. No extra model weight copies. Small-row/small-output fallback unchanged.','gates':{'native_wide_tile_fp32_bit_equality':True,'fp64_original_envelopes':True,'full_tokens_and_states_bit_equal':True,'4b_ttft_min_gain':.10,'4b_total_min_gain':.05,'4b_and_text_decode_max_regression':.02,'text_total_ttft_max_regression':.02},'sequence':'After WS-M5-03 terminal: correctness tests then fresh WorldSense ABBA on unchanged cases. Fresh confirmation only on pass. No performance timing outside WorldSense.'},indent=2));print('Prepared coalesced-prefill prototype; no GPU execution')
