"""Prepare an isolated native-only exact-half shared-storage prototype; no measurements."""
from pathlib import Path
import re
s=Path(__file__).resolve().parent/'source';p=s/'src/backends/metal/metal_context.mm';text=p.read_text()
# Derive original strict FP32 dot/reduction kernels, changing only immutable
# weight-word interpretation. Integer conversion preserves half subnormals.
def convert_kernel(kernel,special):
 kernel=kernel.replace('specialized_bf16_decode','specialized_shared_half_decode').replace('bf16_decode','shared_half_decode')
 kernel=kernel.replace('constant BF16Params& p [[buffer(3)]],','constant SharedHalfParams& p [[buffer(3)]], device const int* scales [[buffer(4)]],')
 if special:
  kernel=kernel.replace('weight+column*ulong(fixed_inner)+k','weight+column*ulong(fixed_wStride)+k')
  for j in [1,2,3]:kernel=kernel.replace('weight+(column+'+str(j)+')*ulong(fixed_inner)+k','weight+(column+'+str(j)+')*ulong(fixed_wStride)+k')
  kernel=kernel.replace('weight[column*ulong(fixed_inner)+k]','weight[column*ulong(fixed_wStride)+k]')
  for j in [1,2,3]:kernel=kernel.replace('weight[(column+'+str(j)+')*ulong(fixed_inner)+k]','weight[(column+'+str(j)+')*ulong(fixed_wStride)+k]')
 else:
  kernel=kernel.replace('weight+column*p.inner+k','weight+column*p.wStride+k')
  for j in [1,2,3]:kernel=kernel.replace('weight+(column+'+str(j)+')*p.inner+k','weight+(column+'+str(j)+')*p.wStride+k')
  kernel=kernel.replace('weight[column*p.inner+k]','weight[column*p.wStride+k]')
  for j in [1,2,3]:kernel=kernel.replace('weight[(column+'+str(j)+')*p.inner+k]','weight[(column+'+str(j)+')*p.wStride+k]')
 old='float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);'
 new='float4 w=float4(shared_half_value(b0[j],scales[column]), column+1<p.columns?shared_half_value(b1[j],scales[column+1]):0.0f, column+2<p.columns?shared_half_value(b2[j],scales[column+2]):0.0f, column+3<p.columns?shared_half_value(b3[j],scales[column+3]):0.0f);'
 assert old in kernel;kernel=kernel.replace(old,new)
 kernel=re.sub(r'as_type<float>\(uint\((weight\[[^\]]+\])\)<<16\)',lambda m:'shared_half_value('+m[1]+',scales['+('column' if '(column+' not in m[1] else 'column+'+re.search(r'\(column\+(\d)\)',m[1])[1])+'])',kernel)
 return kernel
native=text[text.index('kernel void bf16_decode('):text.index('kernel void bf16_decode_small(')]
body=(s/'src/backends/metal/body_shape_shader.h').read_text();body=body[body.index('kernel void specialized_bf16_decode('):body.index('kernel void specialized_bf16_decode_small(')]
shader='''// Derived from the original strict FP32 BF16 dot/reduction kernels.
#pragma once
namespace Garnet {
inline constexpr const char* kSharedHalfShader=R"SHAREDHALF(
struct SharedHalfParams { ulong rows,inner,columns,aStride,oStride,wStride; };
constant uint fixed_wStride [[function_constant(6)]];
inline float shared_half_value(ushort h,int scale) {
    uint sign=uint(h&0x8000)<<16, fraction=uint(h&1023), exponent=uint(h>>10)&31;
    if(exponent) return as_type<float>(sign | (uint(int(exponent)+112-scale)<<23) | (fraction<<13));
    if(!fraction)return as_type<float>(sign);
    uint top=31-clz(fraction);
    return as_type<float>(sign | (uint(int(top)+103-scale)<<23) | ((fraction-(1u<<top))<<(23-top)));
}
'''+convert_kernel(native,False)+convert_kernel(body,True)+')SHAREDHALF";\n}\n'
(s/'src/backends/metal/shared_half_shader.h').write_text(shader)
text=text.replace('#include "body_shape_shader.h"','#include "body_shape_shader.h"\n#ifdef GARNET_METAL_SHARED_EXACT_HALF\n#include "shared_half_shader.h"\n#endif')
assert '#include "shared_half_shader.h"'in text
text=text.replace('    bool globalHalf=false;','    bool sharedExactHalf=false;\n    bool globalHalf=false;')
text=text.replace('    id<MTLComputePipelineState> argmax;','''#ifdef GARNET_METAL_SHARED_EXACT_HALF
    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> sharedHalfBodyShape;
    id<MTLComputePipelineState> sharedHalfDecode;
#endif
    id<MTLComputePipelineState> argmax;''')
text=text.replace('source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];','''source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];
#ifdef GARNET_METAL_SHARED_EXACT_HALF
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kSharedHalfShader]];
#endif''')
text=text.replace('impl_->kernels->bf16Decode=[impl_->device', '''#ifdef GARNET_METAL_SHARED_EXACT_HALF
        impl_->kernels->sharedHalfDecode=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"shared_half_decode"] error:&error];
        if(!impl_->kernels->sharedHalfDecode)throw std::runtime_error("Cannot create shared exact-half decoder");
#endif
        impl_->kernels->bf16Decode=[impl_->device''')
text=text.replace('if(!floatPrefill&&bytes<=16*1024*1024&&impl_->kernels->halfRescale)', '''#ifdef GARNET_METAL_SHARED_EXACT_HALF
        const size_t halfLimit=64*1024*1024;
#else
        const size_t halfLimit=16*1024*1024;
#endif
        if(!floatPrefill&&bytes<=halfLimit&&impl_->kernels->halfRescale)''')
needle='if(!value->exactHalf||!value->halfScales)throw std::runtime_error("Exact half metadata allocation failed");'
assert needle in text;text=text.replace(needle,needle+'''
#ifdef GARNET_METAL_SHARED_EXACT_HALF
                    // Immutable shared layout: MPS and strict FP32 decode use
                    // the same exact words; the old BF16 allocation is released.
                    value->buffer=encoded;value->sharedExactHalf=true;
#endif''')
# Two body-shape caches, since identical dimensions can have different layouts.
text=text.replace('auto& cache=impl_->kernels->bodyShape;', '''auto& cache=impl_->kernels->bodyShape;
#ifdef GARNET_METAL_SHARED_EXACT_HALF
        auto& selectedCache=w->sharedExactHalf?impl_->kernels->sharedHalfBodyShape:cache;
#else
        auto& selectedCache=cache;
#endif''')
# Scope preparation replacement to avoid touching unrelated cache uses.
start=text.index('bool MetalContext::PrepareBF16BodyShape(');end=text.index('size_t MetalContext::BodyShapePipelineCount()',start);part=text[start:end]
part=part.replace('cache.find(key)','selectedCache.find(key)').replace('cache.end()','selectedCache.end()').replace('cache.size()','selectedCache.size()').replace('cache.emplace(key,pipeline)','selectedCache.emplace(key,pipeline)')
needle='NSError* error=nil;\n        auto function='
assert needle in part;part=part.replace(needle,'''#ifdef GARNET_METAL_SHARED_EXACT_HALF
        uint32_t weightStride=w->sharedExactHalf?uint32_t(w->exactHalf.rowBytes/2):uint32_t(w->columns);
        [constants setConstantValue:&weightStride type:MTLDataTypeUInt atIndex:6];
        NSString* functionName=w->sharedExactHalf?@"specialized_shared_half_decode":small?@"specialized_bf16_decode_small":@"specialized_bf16_decode";
#else
        NSString* functionName=small?@"specialized_bf16_decode_small":@"specialized_bf16_decode";
#endif
        NSError* error=nil;
        auto function=''')
part=part.replace('newFunctionWithName:small?@"specialized_bf16_decode_small":@"specialized_bf16_decode"','newFunctionWithName:functionName');text=text[:start]+part+text[end:]
needle='        // Wide tiles share more input/weight loading at chunk sizes 16/32.'
assert needle in text;text=text.replace(needle,'''#ifdef GARNET_METAL_SHARED_EXACT_HALF
        if(w->sharedExactHalf) {
            if(a.rows>2)throw std::logic_error("Shared exact-half prefill failed to use MPS");
            auto pipeline=impl_->kernels->sharedHalfDecode;
            if(o.columns<100000) {
                const std::array<size_t,5> key{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
                std::shared_lock<std::shared_mutex> lock(impl_->kernels->bodyShapeMutex);
                auto found=impl_->kernels->sharedHalfBodyShape.find(key);
                if(found!=impl_->kernels->sharedHalfBodyShape.end()&&found->second) {pipeline=found->second;++impl_->bodyShapeDispatches;}
                else ++impl_->bodyShapeMisses;
            }
            struct SharedParams {uint64_t rows,inner,columns,aStride,oStride,wStride;};
            SharedParams p{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4,w->exactHalf.rowBytes/2};
            auto encoder=[impl_->command computeCommandEncoder];
            if(!encoder)throw std::runtime_error("Cannot create shared exact-half projection encoder");
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];
            [encoder setBuffer:w->buffer offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];
            [encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder setBuffer:w->halfScales offset:w->halfScalesOffset atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake((o.columns+15)/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;
            if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
            return;
        }
#endif
'''+needle)
p.write_text(text)
p=s/'cmake/GarnetMetal.cmake';text=p.read_text();text='''option(GARNET_METAL_SHARED_EXACT_HALF "Share lossless half storage across prefill and strict FP32 decode" OFF)
if(GARNET_METAL_SHARED_EXACT_HALF)
    if(NOT GARNET_METAL_EXACT_HALF_PREFILL OR NOT GARNET_METAL_GLOBAL_HALF_ALPHA OR NOT GARNET_METAL_BODY_SHAPE)
        message(FATAL_ERROR "Shared exact half requires exact-half, common-alpha and body shape")
    endif()
    if(GARNET_METAL_FUSED_HEAD_ARGMAX OR GARNET_METAL_JOINED_PREFILL OR GARNET_METAL_JOINED_GLOBAL_MLP OR GARNET_METAL_INTERLEAVED_BF16_BODY OR GARNET_METAL_NATIVE_BFLOAT_BODY OR GARNET_METAL_DENSE_KEY_SPLIT_K OR GARNET_METAL_SMALL_OUTPUT_DECODE)
        message(FATAL_ERROR "Shared exact half is an isolated storage policy")
    endif()
endif()
'''+text
text=text.replace('add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)','''add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)
if(GARNET_METAL_SHARED_EXACT_HALF)
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_SHARED_EXACT_HALF=1)
endif()''');p.write_text(text)
# Expose an immutable layout query to the native numerical/ownership tests.
p=s/'src/backends/metal/metal_context.h';text=p.read_text().replace('    bool GlobalHalfReady() const;','    bool GlobalHalfReady() const;\n    bool SharedExactHalfReady() const;');p.write_text(text)
p=s/'src/backends/metal/metal_context.mm';text=p.read_text().replace('bool MetalBF16Weight::GlobalHalfReady() const','bool MetalBF16Weight::SharedExactHalfReady() const {return impl_&&impl_->sharedExactHalf;}\nbool MetalBF16Weight::GlobalHalfReady() const');p.write_text(text)
p=s/'src/entry/metal_main.cpp';text=p.read_text();idx=text.index('#ifdef GARNET_METAL_BODY_SHAPE',text.index('std::string BackendInfo')if 'std::string BackendInfo'in text else 0);text=text[:idx]+'''#ifdef GARNET_METAL_SHARED_EXACT_HALF
        result += "; shared_exact_half";
#endif
'''+text[idx:];p.write_text(text)
print('Prepared isolated shared-half storage prototype; uncompiled, unvalidated')
