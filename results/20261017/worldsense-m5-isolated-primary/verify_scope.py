from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-isolated-primary/source';seed=r/'out/research/worldsense-wide-cached-prefill/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();old=json.loads((w/'ws53-seed-source-manifest.json').read_text());m=json.loads((w/'correctness-source-manifest.json').read_text());assert len(old)==1717 and len(m)==1718 and all(sha(seed/k)==h for k,h in old.items())and all(sha(s/k)==h for k,h in m.items())
changes={k for k,h in m.items()if old.get(k)!=h};assert changes=={'cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/primary_isolation_shader.h','src/entry/metal_main.cpp'}
mm=(s/'src/backends/metal/metal_context.mm').read_text();original=(seed/'src/backends/metal/metal_context.mm').read_text()
def replace(a,b):
 global mm
 assert mm.count(a)==1,a;mm=mm.replace(a,b,1)
replace('#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n#include "primary_isolation_shader.h"\n#endif\n','')
line='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];'
replace('#ifndef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n'+line+'\n#endif',line)
line='        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryPrefillMinimalPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];'
replace('''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];
        prefillSource=[prefillSource stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];
#else
'''+line+'\n#endif',line)
replace('''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        id<MTLLibrary> primaryLibrary=library;
#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        primaryLibrary=prefillLibrary;
#endif
#endif
''','')
for name in ['primary_bf16_decode','primary_bf16_gemm','primary_gemm_16_64_64','primary_gemm_32_64_32','primary_expand_words','primary_expand_vector_words']:
 replace('[primaryLibrary newFunctionWithName:@"'+name+'"]','[library newFunctionWithName:@"'+name+'"]')
line='        auto function=[impl_->kernels->library newFunctionWithName:functionName constantValues:constants error:&error];'
replace('''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:impl_->kernels->library;
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];
#else
'''+line+'\n#endif',line)
assert mm==original,'Changes beyond source composition/lookup routing'
header=(s/'src/backends/metal/primary_isolation_shader.h').read_text();preamble=header.split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0];constants=(s/'src/backends/metal/body_shape_shader.h').read_text().split('R"BODYSHAPE(',1)[1].split('kernel void ',1)[0];assert preamble=='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+constants
assert preamble.count('function_constant(')==6 and 'kernel void'not in preamble and 'primary_quad_kword'not in preamble
entry=(s/'src/entry/metal_main.cpp').read_text();marker='#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n        info+="; primary_full_isolated_library";\n#endif\n';assert entry.count(marker)==1 and entry.replace(marker,'',1)==(seed/'src/entry/metal_main.cpp').read_text()
cmake=(s/'cmake/GarnetMetal.cmake').read_text();prefix=(seed/'cmake/GarnetMetal.cmake').read_text();assert cmake.startswith(prefix);addition=cmake[len(prefix):];assert 'OFF)'in addition and 'NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY'in addition
report={'wide_admission_scope_verified':True,'full_context_recovered_by_only_library_composition_and_lookup_inverse':True,'minimal_BF16Params_and_six_function_constants_exact':True,'all_other_source_including_tests_shaders_codec_graph_session_Model_wrapper_CUDA_exact_WS53':True,'same_existing_library_members':True,'dynamic_primary_specialization_uses_secondary_raw_uses_main':True,'source_files':1718,'source_delta_from_WS53':4,'canonical_overlays':10,'scope':'Source identity/route proof only; full native/model/actual program and WorldSense correctness/performance remain required.'};(w/'wide-scope-proof.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
