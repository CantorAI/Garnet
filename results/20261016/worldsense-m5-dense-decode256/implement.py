from pathlib import Path
import shutil,json,hashlib
old=Path('/Users/shawn/CantorAI/out/research/worldsense-private-execution/source');new=Path('/Users/shawn/CantorAI/out/research/worldsense-dense-decode256/source');assert not new.exists();shutil.copytree(old,new)
p=new/'src/backends/metal/body_shape_shader.h';original=p.read_text();a=original.index('kernel void specialized_bf16_decode(');b=original.index('kernel void specialized_bf16_decode_small(',a);body=original[a:b];wide=body.replace('specialized_bf16_decode(','specialized_bf16_decode_dense256(').replace('ulong(group)*16+simd*4','ulong(group)*32+simd*4');assert wide!=body and wide.replace('specialized_bf16_decode_dense256(','specialized_bf16_decode(').replace('ulong(group)*32+simd*4','ulong(group)*16+simd*4')==body
(new/'src/backends/metal/dense_decode256_shader.h').write_text('// Exact per-output SIMD32 arithmetic from body_shape_shader.h; scheduling only.\n#pragma once\nnamespace Garnet { inline constexpr const char* kDenseDecode256Shader=R"DENSE256(\n'+wide+')DENSE256"; }\n')
p=new/'src/backends/metal/metal_context.mm';s=p.read_text();needle='#include "body_shape_shader.h"';assert needle in s;s=s.replace(needle,needle+'\n#ifdef GARNET_METAL_DENSE_DECODE256\n#include "dense_decode256_shader.h"\n#endif')
needle='    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> bodyShape;';assert needle in s;s=s.replace(needle,'    struct BodyShapeEntry {id<MTLComputePipelineState> pipeline=nil;bool wide=false;};\n    std::map<std::array<size_t,5>,BodyShapeEntry> bodyShape;')
needle='    size_t bodyShapeDispatches=0,bodyShapeMisses=0;';assert needle in s;s=s.replace(needle,needle+'\n    size_t denseDecode256Encoded=0;')
needle='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];';assert needle in s;s=s.replace(needle,needle+'\n#ifdef GARNET_METAL_DENSE_DECODE256\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kDenseDecode256Shader]];\n#endif')
s=s.replace('return found->second!=nil;','return found->second.pipeline!=nil;') if 'return found->second!=nil;'in s else s
start=s.index('        uint32_t dimensions[5];',s.index('bool MetalContext::PrepareBF16BodyShape'));end=s.index('\n    }\n#else',start)
s=s[:start]+'''        auto create=[&](bool wide)->id<MTLComputePipelineState> {
            uint32_t dimensions[5];for(size_t i=0;i<5;++i)dimensions[i]=uint32_t(key[i]);
            auto constants=[MTLFunctionConstantValues new];
            for(NSUInteger i=0;i<5;++i)[constants setConstantValue:&dimensions[i] type:MTLDataTypeUInt atIndex:i];
            bool noTail=o.columns%(wide?32:(small?4:16))==0;
            [constants setConstantValue:&noTail type:MTLDataTypeBool atIndex:5];
            NSError* error=nil;
            NSString* name=wide?@"specialized_bf16_decode_dense256":(small?@"specialized_bf16_decode_small":@"specialized_bf16_decode");
            auto function=[impl_->kernels->library newFunctionWithName:name constantValues:constants error:&error];
            id<MTLComputePipelineState> pipeline=nil;
            if(function)pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
            if(pipeline&&(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<(wide?256:(small?64:128))))pipeline=nil;
            return pipeline;
        };
        bool wide=false;id<MTLComputePipelineState> pipeline=nil;
#ifdef GARNET_METAL_DENSE_DECODE256
        if(a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256) {pipeline=create(true);wide=pipeline!=nil;}
#endif
        if(!pipeline)pipeline=create(false);
        cache.emplace(key,MetalPipelines::BodyShapeEntry{pipeline,wide});return pipeline!=nil;'''+s[end:]
needle='size_t MetalContext::LastBodyShapeMisses() const {return impl_->bodyShapeMisses;}';assert needle in s;s=s.replace(needle,needle+'\nsize_t MetalContext::DenseDecode256EncodedTotal() const {return impl_->denseDecode256Encoded;}')
needle='''            if(found!=impl_->kernels->bodyShape.end()&&found->second) {
                pipeline=found->second;++impl_->bodyShapeDispatches;''';assert needle in s;s=s.replace(needle,'''            if(found!=impl_->kernels->bodyShape.end()&&found->second.pipeline) {
                pipeline=found->second.pipeline;++impl_->bodyShapeDispatches;
#ifdef GARNET_METAL_DENSE_DECODE256
                if(found->second.wide){threads=256;decodeColumns=32;}
#endif''')
# Count only after the actual encoder dispatch call, including subsequently aborted batches.
needle='''        [encoder dispatchThreadgroups:a.rows<=2?MTLSizeMake((o.columns+decodeColumns-1)/decodeColumns,1,1):MTLSizeMake((o.columns+tileColumns-1)/tileColumns,(o.rows+tileRows-1)/tileRows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];''';assert needle in s;s=s.replace(needle,needle+'''\n#ifdef GARNET_METAL_DENSE_DECODE256
        if(threads==256&&decodeColumns==32&&a.rows==1&&(a.columns==2048||a.columns==6144)&&o.columns>256&&o.columns<100000)++impl_->denseDecode256Encoded;
#endif''');p.write_text(s)
p=new/'src/backends/metal/metal_context.h';s=p.read_text();needle='    size_t LastBodyShapeMisses() const;';assert needle in s;s=s.replace(needle,needle+'\n    size_t DenseDecode256EncodedTotal() const;');p.write_text(s)
p=new/'src/backends/metal/decoder_session.h';s=p.read_text();needle='    size_t ScopedExecutionRuns() const;';assert needle in s;s=s.replace(needle,needle+'\n    size_t DenseDecode256EncodedTotal() const;');needle='        APISET().AddProp("scoped_execution_runs",&MetalDecoderSession::ScopedExecutionRuns);';assert needle in s;s=s.replace(needle,needle+'\n        APISET().AddProp("dense_decode256_encoded_total",&MetalDecoderSession::DenseDecode256EncodedTotal);');p.write_text(s)
p=new/'src/backends/metal/decoder_session.cpp';s=p.read_text();s+='\nnamespace Garnet {size_t MetalDecoderSession::DenseDecode256EncodedTotal() const {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->DenseDecode256EncodedTotal():0;}}\n';p.write_text(s)
p=new/'runtime/metal_serving.py';s=p.read_text();needle="        stages['private_execution_runs']";at=s.index(needle);s=s[:at]+"        stages['dense_decode256_encoded_total'] = self.decoder.session.dense_decode256_encoded_total\n"+s[at:];p.write_text(s)
p=new/'src/entry/metal_main.cpp';s=p.read_text();needle='#ifdef GARNET_METAL_DENSE_PREFIX_CACHE';at=s.index(needle);s=s[:at]+'#ifdef GARNET_METAL_DENSE_DECODE256\n        info+="; dense_decode256";\n#endif\n'+s[at:];p.write_text(s)
p=new/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+'''\noption(GARNET_METAL_DENSE_DECODE256 "Eight SIMD groups for inner2048/6144 body decode" OFF)
if(GARNET_METAL_DENSE_DECODE256)
 if(NOT GARNET_METAL_BODY_SHAPE)
  message(FATAL_ERROR "Dense decode256 requires body shape specialization")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DENSE_DECODE256=1)
endif()
''')
print('Isolated WS27 source written; unbuilt/untested')
