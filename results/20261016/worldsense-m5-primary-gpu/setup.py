from pathlib import Path
import shutil,json,hashlib,datetime,re
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-cached-matrix-metadata/source';s=r/'out/research/worldsense-primary-gpu/source';assert not s.exists();shutil.copytree(old,s,ignore=shutil.ignore_patterns('__pycache__','.git','results'))
shutil.copy2(w.parent/'worldsense-primary-blocks/native-design.txt',w/'native-design-before-implementation.txt')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-37','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':'WS34 source with lazy/cached-metadata OFF; no WS35 paused diagnostic. Original WS10 kernel profile.','hypothesis':'Sole-primary lossless compact/raw BF16 weights may reduce actual live-weight storage; decode/prefill/header read overhead may regress. CPU feasibility is not speed proof.','admission':'DefaultOFF primary flag;rows>256/rows%4==0/K%128==0/nonfloat/nonvisionBF16 with byte savings. Original small/F32/unsupported fallback unchanged;398actual large matrices.','gates':'Native GPU original bit reconstruction, unchanged FP32word+FP64 bounds, full32VLM28textstates/tokens/vision/prefix/capacity/actualgraph/fault/ownership first. Then freshWorldSense8resource>=10%Metal>=5%footprint,20speedbothmodelsTTFTtotaldecodeSkillwall<=2%regression,48fullfourworkers/twoindependentlypassingpairs beforepromotion.','accounting':'Nativepacking/setup/temporarymemory cost recorded. Before/after native storage counter collection included in outer TTFT/total; counters encoded not completed GPU. No CPython images/inference, CUDA unchanged, four files/no camera, restoreWS10normal afterevery stage.','selection':'WS36 CPU feasible23.4%Bf16byte reduction; no timing-selected policy or inference retry.'},indent=2))
shutil.copy2(r/'out/research/worldsense-primary-blocks/cpu-source/primary_block_codec.h',s/'src/backends/metal/primary_block_codec.h')
# Deterministically clone original kernels with weight-read substitutions only.
mm=s/'src/backends/metal/metal_context.mm';original=mm.read_text();body=(s/'src/backends/metal/body_shape_shader.h').read_text()
def kernel(text,name):
 a=text.index('kernel void '+name+'(');brace=text.index('{',a);depth=1;i=brace+1
 while depth:
  if text[i]=='{':depth+=1
  elif text[i]=='}':depth-=1
  i+=1
 return text[a:i]
helper='''inline ushort primary_word(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    ulong block=(row/4)*(inner/128)+k/128,meta=metadata[block],at=(meta>>9)*256,index=(k%128)*4+row%4;
    if(meta&256ul){uint sm=payload[at+index],tag=(payload[at+512+index/2]>>((index%2)*4))&15u;
        return ushort((sm&127u)|((sm&128u)<<8)|(((uint(meta)&255u)+tag)<<7));}
    return ushort(payload[at+index*2])|(ushort(payload[at+index*2+1])<<8);
}
inline ushort4 primary_row4(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    return ushort4(primary_word(payload,metadata,row,k,inner),primary_word(payload,metadata,row,k+1,inner),primary_word(payload,metadata,row,k+2,inner),primary_word(payload,metadata,row,k+3,inner));
}
kernel void primary_expand_words(device const uchar* payload [[buffer(0)]],device const ulong* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 if(ulong(id)>=count)return;ulong i=start+id;output[id]=uint(primary_word(payload,metadata,i/inner,i%inner,inner))<<16;
}
'''
variants=[];audit=[]
for name in ['bf16_decode','bf16_gemm','gemm_16_64_64','gemm_32_64_32','specialized_bf16_decode']:
 src=kernel(body if name.startswith('specialized')else original,name);t=src;substitutions=[]
 def replace(a,b):
  global t
  n=t.count(a)
  if n:t=t.replace(a,b);substitutions.append((a,b,n))
 replace('kernel void '+name+'(', 'kernel void primary_'+name+'(')
 replace('device const ushort* weight [[buffer(1)]]','device const uchar* payload [[buffer(1)]],device const ulong* metadata [[buffer(4)]]')
 inner='ulong(fixed_inner)'if name.startswith('specialized')else'p.inner'
 for row in ['column','(column+1)','(column+2)','(column+3)']:
  replace('*(device const ushort4*)(weight+'+row+'*'+inner+'+k)','primary_row4(payload,metadata,'+row+',k,'+inner+')')
  replace('weight['+row+'*'+inner+'+k]','primary_word(payload,metadata,'+row+',k,'+inner+')')
 assert not re.search(r'\bweight\b',t),name
 recovered=t
 for a,b,n in reversed(substitutions):assert recovered.count(b)==n,(name,b);recovered=recovered.replace(b,a)
 assert recovered==src,name
 variants.append(t);audit.append({'name':name,'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'transformed_sha256':hashlib.sha256(t.encode()).hexdigest(),'inverse_substitution_recovers_original_exactly':True,'substitutions':substitutions})
(s/'src/backends/metal/primary_block_shader.h').write_text('#pragma once\nnamespace Garnet {inline constexpr const char* kPrimaryBlockShader=R"PRIMARY(\n'+helper+'\n'.join(variants)+'\n)PRIMARY";}\n')
(w/'shader-substitution-audit.json').write_text(json.dumps(audit,indent=2))
t=original.replace('#include "metal_context.h"','#include "metal_context.h"\n#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS\n#include "primary_block_codec.h"\n#include "primary_block_shader.h"\n#include <atomic>\n#endif',1)
needle='namespace Garnet {\nstruct MetalMatrix::Impl'
globals='''namespace Garnet {
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
namespace {
struct PrimaryCounters {std::atomic<uint64_t> weights{0},original{0},payload{0},metadata{0},allocated{0},uploads{0},decode{0},prefill{0},head{0};};
PrimaryCounters primaryCounters;
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
std::atomic<int> primaryFailure{0};
void PrimaryMaybeFail(int stage){int expected=stage;if(primaryFailure.compare_exchange_strong(expected,0))throw std::runtime_error("Injected primary weight allocation failure");}
#endif
}
#endif
struct MetalMatrix::Impl'''
assert needle in t;t=t.replace(needle,globals,1)
needle='    id<MTLBuffer> buffer;'
fields='''    id<MTLBuffer> buffer;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
    id<MTLBuffer> primaryPayload,primaryMetadata;
    uint64_t primaryOriginalBytes=0,primaryPayloadBytes=0,primaryMetadataBytes=0,primaryAllocatedBytes=0;
    bool primaryCharged=false;
    ~Impl(){if(primaryCharged){primaryCounters.weights.fetch_sub(1);primaryCounters.original.fetch_sub(primaryOriginalBytes);primaryCounters.payload.fetch_sub(primaryPayloadBytes);primaryCounters.metadata.fetch_sub(primaryMetadataBytes);primaryCounters.allocated.fetch_sub(primaryAllocatedBytes);}}
#endif'''
assert t.count(needle)==1;t=t.replace(needle,fields)
idx=t.index('bool MetalBF16Weight::ExactHalfReady()');t=t[:idx]+'''bool MetalBF16Weight::PrimaryReady() const {
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 return impl_&&impl_->primaryCharged&&impl_->primaryPayload&&impl_->primaryMetadata&&!impl_->buffer;
#else
 return false;
#endif
}
MetalPrimaryStats MetalContext::PrimaryStats() {
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
 return {primaryCounters.weights.load(),primaryCounters.original.load(),primaryCounters.payload.load(),primaryCounters.metadata.load(),primaryCounters.allocated.load(),primaryCounters.uploads.load(),primaryCounters.decode.load(),primaryCounters.prefill.load(),primaryCounters.head.load()};
#else
 return {};
#endif
}
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
void MetalContext::PrimaryTestFailAfter(int stage){if(stage<0||stage>2)throw std::invalid_argument("Invalid primary failure stage");primaryFailure.store(stage);}
#endif
'''+t[idx:]
needle='    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> bodyShape;';assert needle in t;t=t.replace(needle,needle+'''
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> primaryBodyShape;
#endif''')
needle='    id<MTLComputePipelineState> bf16Decode,bf16DecodeSmall,bf16Gemm,bf16Gemm16Wide,bf16Gemm32Wide;';assert needle in t;t=t.replace(needle,needle+'''
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
    id<MTLComputePipelineState> primaryDecode,primaryGemm,primaryGemm16Wide,primaryGemm32Wide,primaryExpand;
#endif''')
needle='        auto library=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];';assert needle in t;t=t.replace(needle,'''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];
#endif
'''+needle,1)
needle='        impl_->kernels->rms=[impl_->device newComputePipelineStateWithFunction:';idx=t.index(needle)
pipeline='''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
'''
for field,name in [('primaryDecode','bf16_decode'),('primaryGemm','bf16_gemm'),('primaryGemm16Wide','gemm_16_64_64'),('primaryGemm32Wide','gemm_32_64_32'),('primaryExpand','expand_words')]:
 pipeline+='        impl_->kernels->'+field+'=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"primary_'+name+'"] error:&error];\n        if(!impl_->kernels->'+field+')throw std::runtime_error("Cannot create primary '+name+' pipeline");\n'
pipeline+='#endif\n';t=t[:idx]+pipeline+t[idx:]
# Pack before ever allocating a raw buffer; unsupported/non-saving inputs remain original.
needle='        value->buffer=[impl_->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];';assert t.count(needle)==1
pack='''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"Primary metadata uses little endian");
        if(!floatPrefill&&!visionPrefill&&rows>256&&PrimaryBlocks::Eligible(rows,columns)) {
            const size_t count=rows/4*(columns/128);
            if(count>SIZE_MAX/8)throw std::overflow_error("Primary metadata count overflow");
            std::vector<uint8_t> payload;payload.reserve(bytes);std::vector<uint64_t> metadata;metadata.reserve(count);
            std::array<uint16_t,PrimaryBlocks::Words> words;auto source=static_cast<const uint8_t*>(data);
            for(size_t group=0;group<rows/4;++group)for(size_t block=0;block<columns/128;++block) {
                for(size_t k=0;k<128;++k)for(size_t j=0;j<4;++j){const size_t pos=((group*4+j)*columns+block*128+k)*2;words[k*4+j]=uint16_t(source[pos])|uint16_t(source[pos+1])<<8;}
                const auto encoded=PrimaryBlocks::Encode(words,payload.size());metadata.push_back(encoded.metadata);
                payload.insert(payload.end(),encoded.data.begin(),encoded.data.begin()+encoded.bytes);
            }
            if(metadata.size()!=count)throw std::logic_error("Primary block count mismatch");
            const auto metaBytes=metadata.size()*sizeof(uint64_t);
            if(payload.size()<bytes&&metaBytes<bytes-payload.size()) {
                if(payload.size()>impl_->device.maxBufferLength||metaBytes>impl_->device.maxBufferLength)throw std::runtime_error("Primary buffer exceeds device limit");
                value->primaryPayload=[impl_->device newBufferWithBytes:payload.data() length:payload.size() options:MTLResourceStorageModeShared];
                if(!value->primaryPayload)throw std::runtime_error("Primary payload allocation failed");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
                PrimaryMaybeFail(1);
#endif
                value->primaryMetadata=[impl_->device newBufferWithBytes:metadata.data() length:metaBytes options:MTLResourceStorageModeShared];
                if(!value->primaryMetadata)throw std::runtime_error("Primary metadata allocation failed");
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
                PrimaryMaybeFail(2);
#endif
                value->primaryOriginalBytes=bytes;value->primaryPayloadBytes=payload.size();value->primaryMetadataBytes=metaBytes;
                value->primaryAllocatedBytes=value->primaryPayload.allocatedSize+value->primaryMetadata.allocatedSize;
                value->primaryCharged=true;primaryCounters.weights.fetch_add(1);primaryCounters.original.fetch_add(bytes);primaryCounters.payload.fetch_add(payload.size());primaryCounters.metadata.fetch_add(metaBytes);primaryCounters.allocated.fetch_add(value->primaryAllocatedBytes);primaryCounters.uploads.fetch_add(1);
                return MetalBF16Weight(value);
            }
        }
#endif
'''
t=t.replace(needle,pack+needle,1)
needle='        add(w->buffer,"bf16");if(w->exactHalf)';assert needle in t;t=t.replace(needle,'''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        add(w->primaryPayload,"primary_bf16_payload");add(w->primaryMetadata,"primary_bf16_metadata");
#endif
'''+needle)
# Both preparation and actual encode use the selected immutable storage/device.
for start,end in [('bool MetalContext::PrepareBF16BodyShape(', 'size_t MetalContext::BodyShapePipelineCount()'),('void MetalContext::EncodeBF16MatVec(', 'MetalMatrix MetalContext::Upload(')]:
 a=t.index(start);b=t.index(end,a);part=t[a:b];needle='auto w=weight.impl_;'
 # Actual method declares the same text at end of a comma-separated native declaration.
 assert needle in part;part=part.replace(needle,needle+'''
        id<MTLBuffer> storage=w->buffer;bool primary=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        primary=w->primaryCharged;if(primary)storage=w->primaryPayload;
#endif''',1);part=part.replace('w->buffer.device!=impl_->device','storage.device!=impl_->device')
 if start.startswith('bool'):
  part=part.replace('auto& cache=impl_->kernels->bodyShape;','''auto& cache=impl_->kernels->bodyShape;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        auto& selectedCache=primary?impl_->kernels->primaryBodyShape:cache;
#else
        auto& selectedCache=cache;
#endif''').replace('auto found=cache.find(key);if(found!=cache.end())','auto found=selectedCache.find(key);if(found!=selectedCache.end())').replace('if(cache.size()>=128)','if(selectedCache.size()>=128)').replace('cache.emplace(key,pipeline)','selectedCache.emplace(key,pipeline)')
  part=part.replace('auto function=[impl_->kernels->library newFunctionWithName:small?@"specialized_bf16_decode_small":@"specialized_bf16_decode" constantValues:constants error:&error];','''NSString* functionName=small?@"specialized_bf16_decode_small":@"specialized_bf16_decode";
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        if(primary)functionName=@"primary_specialized_bf16_decode";
#endif
        auto function=[impl_->kernels->library newFunctionWithName:functionName constantValues:constants error:&error];''')
 else:
  needle='        size_t threads=wide?256:128,tileRows=wide&&a.rows>16?32:16,tileColumns=wide?64:32;';assert needle in part;part=part.replace(needle,'''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        if(primary){pipeline=a.rows<=2?impl_->kernels->primaryDecode:wide?(a.rows>16?impl_->kernels->primaryGemm32Wide:impl_->kernels->primaryGemm16Wide):impl_->kernels->primaryGemm;}
#endif
'''+needle)
  part=part.replace('auto found=impl_->kernels->bodyShape.find(key);\n            if(found!=impl_->kernels->bodyShape.end()', '''auto& cache=impl_->kernels->bodyShape;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
            auto& selectedCache=primary?impl_->kernels->primaryBodyShape:cache;
#else
            auto& selectedCache=cache;
#endif
            auto found=selectedCache.find(key);
            if(found!=selectedCache.end()''')
  # The accepted profile disables old auxiliary BF16 layouts.
  needle='        [encoder setBuffer:w->buffer offset:0 atIndex:1];\n#endif\n        [encoder setBuffer:o.data';assert needle in part
  part=part.replace(needle,'''        [encoder setBuffer:storage offset:0 atIndex:1];
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        if(primary){[encoder setBuffer:w->primaryMetadata offset:0 atIndex:4];if(a.rows>2)primaryCounters.prefill.fetch_add(1);else if(o.columns>=100000)primaryCounters.head.fetch_add(1);else primaryCounters.decode.fetch_add(1);}
#endif
#endif
        [encoder setBuffer:o.data''')
 t=t[:a]+part+t[b:]
# Pipeline metadata count includes both disjoint caches.
needle='return impl_->kernels->bodyShape.size();';assert needle in t;t=t.replace(needle,'''return impl_->kernels->bodyShape.size()
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        +impl_->kernels->primaryBodyShape.size()
#endif
        ;''',1)
# Bounded GPU readback is correctness-only, never exposed to model serving.
t+='''
namespace Garnet {
std::vector<uint32_t> MetalContext::PrimaryReadWords(const MetalBF16Weight& weight,size_t start,size_t count) {
#ifndef GARNET_METAL_PRIMARY_BF16_BLOCKS
 (void)weight;(void)start;(void)count;throw std::logic_error("Primary weight support disabled");
#else
 @autoreleasepool {
  if(impl_->command||!weight.PrimaryReady()||!count||count>8*1024*1024)throw std::invalid_argument("Primary readback requires packed weight, closed batch and bounded count");
  auto w=weight.impl_;if(w->primaryPayload.device!=impl_->device||w->primaryMetadata.device!=impl_->device||start>w->rows*w->columns||count>w->rows*w->columns-start)throw std::invalid_argument("Primary readback device/range mismatch");
  id<MTLBuffer> output=[impl_->device newBufferWithLength:count*4 options:MTLResourceStorageModeShared];if(!output)throw std::runtime_error("Primary readback allocation failed");
  auto command=[impl_->queue commandBuffer];auto encoder=[command computeCommandEncoder];if(!encoder)throw std::runtime_error("Primary readback encoder failed");
  uint64_t inner=w->columns,offset=start,n=count;
  [encoder setComputePipelineState:impl_->kernels->primaryExpand];[encoder setBuffer:w->primaryPayload offset:0 atIndex:0];[encoder setBuffer:w->primaryMetadata offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];[encoder setBytes:&inner length:8 atIndex:3];[encoder setBytes:&offset length:8 atIndex:4];[encoder setBytes:&n length:8 atIndex:5];
  [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];[command commit];[command waitUntilCompleted];if(command.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("Primary readback GPU failed");
  std::vector<uint32_t> result(count);std::memcpy(result.data(),output.contents,count*4);return result;
 }
#endif
}
}
'''
mm.write_text(t)
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('    bool ExactHalfReady() const;','    bool PrimaryReady() const;\n    bool ExactHalfReady() const;',1);idx=t.index('class MetalContext {');t=t[:idx]+'''struct MetalPrimaryStats {
 uint64_t activeWeights=0,originalBytes=0,payloadBytes=0,metadataBytes=0,allocatedBytes=0,uploads=0,decodeEncoded=0,prefillEncoded=0,headEncoded=0;
};
'''+t[idx:];needle='    MetalContext();';t=t.replace(needle,'''    static MetalPrimaryStats PrimaryStats();
    std::vector<uint32_t> PrimaryReadWords(const MetalBF16Weight&,size_t start,size_t count);
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
    static void PrimaryTestFailAfter(int stage);
#endif
'''+needle,1);p.write_text(t)
# Native JSON counters and inclusive serialized serving wrapper.
p=s/'src/entry/metal_main.cpp';t=p.read_text();idx=t.index('    std::string MemoryStatsJson()');t=t[:idx]+'''    std::string PrimaryStatsJson() {
        const auto p=MetalContext::PrimaryStats();
        return nlohmann::json{{"active_weights",p.activeWeights},{"original_bytes",p.originalBytes},{"payload_bytes",p.payloadBytes},{"metadata_bytes",p.metadataBytes},{"allocated_bytes",p.allocatedBytes},{"uploads",p.uploads},{"decode_encoded",p.decodeEncoded},{"prefill_encoded",p.prefillEncoded},{"head_encoded",p.headEncoded}}.dump();
    }
'''+t[idx:];needle='        APISET().AddFunc<0>("memory_stats_json"';assert needle in t;t=t.replace(needle,'        APISET().AddFunc<0>("primary_storage_stats_json",&GarnetMetalAPI::PrimaryStatsJson);\n'+needle,1);needle='#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS';idx=t.index(needle);t=t[:idx]+'''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        info+="; primary_bf16_blocks";
#endif
'''+t[idx:];p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text();needle="            return json.dumps(_models[name].infer(prompt, image, int(limit)))";assert needle in t;t=t.replace(needle,'''            outer_started=time.monotonic()
            before=json.loads(garnet.primary_storage_stats_json())
            model_started=time.monotonic()
            result=_models[name].infer(prompt,image,int(limit))
            after=json.loads(garnet.primary_storage_stats_json())
            if result.get('status')=='ok':
                result['ttft_seconds'] += model_started-outer_started
                result['timings']['primary_storage_before']=before
                result['timings']['primary_storage_after']=after
                result['timings']['primary_encoded_delta']={key:after[key]-before[key] for key in ['decode_encoded','prefill_encoded','head_encoded']}
                result['total_seconds']=time.monotonic()-outer_started
            return json.dumps(result)''');p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();t+='''
option(GARNET_METAL_PRIMARY_BF16_BLOCKS "Sole-primary lossless BF16 compact/raw blocks" OFF)
option(GARNET_METAL_PRIMARY_TEST_FAULTS "Correctness-only primary weight allocation failure" OFF)
if(GARNET_METAL_PRIMARY_BF16_BLOCKS)
 if(NOT GARNET_METAL_BODY_SHAPE)
  message(FATAL_ERROR "Initial primary block profile requires body-shape support")
 endif()
 foreach(policy GARNET_METAL_EXACT_HALF_PREFILL GARNET_METAL_GLOBAL_HALF_ALPHA GARNET_METAL_NATIVE_BFLOAT_BODY GARNET_METAL_CONSTANT_BF16_HEAD GARNET_METAL_SINGLE_CODER_HEAD GARNET_METAL_INTERLEAVED_BF16_BODY GARNET_METAL_COALESCED_BF16_BODY GARNET_METAL_FUSED_HEAD_ARGMAX GARNET_METAL_SCREENED_HEAD GARNET_METAL_FLOAT_DENSE_KEY_PREFILL GARNET_METAL_JOINED_PREFILL GARNET_METAL_JOINED_GLOBAL_MLP GARNET_METAL_MLP_PREFILL_GRAPH)
  if(${policy})
   message(FATAL_ERROR "Primary block profile excludes raw-only auxiliary policy: ${policy}")
  endif()
 endforeach()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_BF16_BLOCKS=1)
endif()
if(GARNET_METAL_PRIMARY_TEST_FAULTS)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS)
  message(FATAL_ERROR "Primary test faults require primary weights")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_TEST_FAULTS=1)
endif()
''';p.write_text(t)
# Actual source closure will be frozen after final correctness, before timing.
print('WS37 native implementation prepared; no build or performance claim',flush=True)
