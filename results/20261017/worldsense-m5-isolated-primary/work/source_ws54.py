from pathlib import Path
import hashlib,json,shutil,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=root/'work/worldsense-isolated-primary';seed=r/'out/research/worldsense-wide-cached-prefill/source';s=r/'out/research/worldsense-isolated-primary/source';old=root/'work/worldsense-wide-cached-prefill';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(w/'protocol.json')['experiment']=='WS-M5-54'and not s.exists();m=load(old/'correctness-source-manifest.json');assert len(m)==1717 and all(sha(seed/k)==h for k,h in m.items());(w/'ws53-seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n');s.mkdir(parents=True)
for rel in m:(s/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/rel,s/rel)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();t+='''
option(GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY "Compile all packed kernels in the existing isolated prefill library" OFF)
if(GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY)
  message(FATAL_ERROR "Full primary isolation requires primary weights and isolated vector prefill")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=1)
endif()
''';p.write_text(t)
constants=(s/'src/backends/metal/body_shape_shader.h').read_text().split('R"BODYSHAPE(',1)[1].split('kernel void ',1)[0]
preamble='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+constants
(s/'src/backends/metal/primary_isolation_shader.h').write_text('#pragma once\nnamespace Garnet {inline constexpr const char* kPrimaryIsolatedPreamble=R"IPRIMARY('+preamble+')IPRIMARY";}\n')
p=s/'src/backends/metal/metal_context.mm';t=p.read_text();needle='#include "primary_block_shader.h"';assert t.count(needle)==1;t=t.replace(needle,needle+'\n#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n#include "primary_isolation_shader.h"\n#endif',1)
needle='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];';assert t.count(needle)==1;t=t.replace(needle,'#ifndef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n'+needle+'\n#endif',1)
needle='        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryPrefillMinimalPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];';assert t.count(needle)==1
addition='''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];
        prefillSource=[prefillSource stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];
#else
'''+needle+'\n#endif';t=t.replace(needle,addition,1)
needle='        impl_->kernels->argmax=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"argmax_rows"] error:&error];';assert t.count(needle)==1
addition='''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        id<MTLLibrary> primaryLibrary=library;
#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        primaryLibrary=prefillLibrary;
#endif
#endif
''';t=t.replace(needle,addition+needle,1)
names=['primary_bf16_decode','primary_bf16_gemm','primary_gemm_16_64_64','primary_gemm_32_64_32','primary_expand_words','primary_expand_vector_words']
for name in names:
 needle='[library newFunctionWithName:@"'+name+'"]';assert t.count(needle)==1;t=t.replace(needle,'[primaryLibrary newFunctionWithName:@"'+name+'"]',1)
needle='        auto function=[impl_->kernels->library newFunctionWithName:functionName constantValues:constants error:&error];';assert t.count(needle)==1
t=t.replace(needle,'''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:impl_->kernels->library;
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];
#else
'''+needle+'\n#endif',1);p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='        return info;';assert t.count(needle)==1;t=t.replace(needle,'#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n        info+="; primary_full_isolated_library";\n#endif\n'+needle,1);p.write_text(t)
current={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};changed=[k for k,h in current.items()if m.get(k)!=h];expected=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/primary_isolation_shader.h','src/entry/metal_main.cpp'];assert len(current)==1718 and sorted(changed)==expected
base=load(old/'seed-source-manifest.json');overlay=[k for k,h in current.items()if base.get(k)!=h];assert len(overlay)==10
for name in ['starting-source-manifest.json','correctness-source-manifest.json']:(w/name).write_text(json.dumps(current,indent=2)+'\n')
(w/'implementation-audit.json').write_text(json.dumps({'source_files':1718,'changed_files':overlay,'seed_files_changed':changed,'all_original_shader_codec_math_test_Model_wrapper_CUDA_unchanged':True,'libraries':'Existing main and vectorPrefillLibrary fields only; no third retained library','PSO_count_unchanged':True,'scope':'Source inventory only; no native/model/speed result yet.'},indent=2)+'\n')
(w/'implementation-identity.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'design_sha256':sha(w/'native-design-before-implementation.txt'),'source_created_after_design':True,'preamble_sha256':hashlib.sha256(preamble.encode()).hexdigest()},indent=2)+'\n');print('WS54 source frozen',len(current),'canonical overlays',len(overlay),'changed from53',len(changed))
