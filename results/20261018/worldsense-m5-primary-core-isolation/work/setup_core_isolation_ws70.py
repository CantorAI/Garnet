"""Implement only prospective ordinary-core partition/counters and new native test."""
from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-primary-core-isolation';old=r/'out/research/worldsense-request-weight-residency/source';s=r/'out/research/worldsense-primary-core-isolation/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();seed=load(w/'seed-source-manifest.json');protocol=load(w/'protocol.json')
assert protocol['state']=='DESIGN_BEFORE_SOURCE'and sha(w/'native-design-before-implementation.txt')==protocol['design_sha256']and not s.exists()
shutil.copytree(old,s,ignore=shutil.ignore_patterns('__pycache__','.git'));assert {str(p.relative_to(s)):sha(p)for p in s.rglob('*')if p.is_file()}==seed
transforms={}
def edit(rel,operations):
 p=s/rel;text=p.read_text();record=[]
 for before,after in operations:
  assert text.count(before)==1,(rel,before[:120],text.count(before));text=text.replace(before,after);record.append(dict(before=before,after=after,count=1))
 inverse=text
 for op in reversed(record):assert inverse.count(op['after'])==1;inverse=inverse.replace(op['after'],op['before'])
 assert inverse==p.read_text();p.write_text(text);transforms[rel]=record
mm='src/backends/metal/metal_context.mm'
edit(mm,[
 ('#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n#include "primary_isolation_shader.h"','#if defined(GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY) || defined(GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY)\n#include "primary_isolation_shader.h"'),
 ('vectorDecode{0},vectorHead{0};};','vectorDecode{0},vectorHead{0},coreDecode{0},corePrefill{0},coreHead{0};};'),
 ('primaryCounters.vectorHead.load()};','primaryCounters.vectorHead.load(),primaryCounters.coreDecode.load(),primaryCounters.corePrefill.load(),primaryCounters.coreHead.load()};'),
 ('    id<MTLLibrary> vectorPrefillLibrary=nil;','    id<MTLLibrary> vectorPrefillLibrary=nil;\n#ifdef GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY\n    id<MTLLibrary> primaryCoreLibrary=nil;\n#endif'),
 ('#ifndef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];','#if !defined(GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY) && !defined(GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY)\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];'),
 ('        id<MTLLibrary> primaryLibrary=library;','''        id<MTLLibrary> primaryLibrary=library;
#ifdef GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY
        NSString* coreSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];
        auto coreLibrary=[impl_->device newLibraryWithSource:coreSource options:compileOptions error:&error];
        if(!coreLibrary)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        impl_->kernels->primaryCoreLibrary=coreLibrary;
        impl_->kernels->prefillProgram=coreSource.UTF8String;
        primaryLibrary=coreLibrary;
#endif'''),
 ('''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:impl_->kernels->library;
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];
#else''','''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:impl_->kernels->library;
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];
#elif defined(GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY)
        auto sourceLibrary=primary?impl_->kernels->primaryCoreLibrary:impl_->kernels->library;
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];
#else'''),
 ('        if(primary){[encoder setBuffer:w->primaryMetadata offset:0 atIndex:4];if(a.rows>2)primaryCounters.prefill.fetch_add(1);else if(o.columns>=100000){primaryCounters.head.fetch_add(1);primaryCounters.vectorHead.fetch_add(1);}else{primaryCounters.decode.fetch_add(1);primaryCounters.vectorDecode.fetch_add(1);}}','''        if(primary){[encoder setBuffer:w->primaryMetadata offset:0 atIndex:4];if(a.rows>2)primaryCounters.prefill.fetch_add(1);else if(o.columns>=100000){primaryCounters.head.fetch_add(1);primaryCounters.vectorHead.fetch_add(1);}else{primaryCounters.decode.fetch_add(1);primaryCounters.vectorDecode.fetch_add(1);}}
#ifdef GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY
        if(primary){if(a.rows>2)primaryCounters.corePrefill.fetch_add(1);else if(o.columns>=100000)primaryCounters.coreHead.fetch_add(1);else primaryCounters.coreDecode.fetch_add(1);}
#endif''')])
edit('src/backends/metal/metal_context.h',[('vectorDecodeEncoded=0,vectorHeadEncoded=0;','vectorDecodeEncoded=0,vectorHeadEncoded=0,coreLibraryDecodeEncoded=0,coreLibraryPrefillEncoded=0,coreLibraryHeadEncoded=0;')])
edit('src/entry/metal_main.cpp',[
 ('{"vector_head_encoded",p.vectorHeadEncoded}}.dump();','{"vector_head_encoded",p.vectorHeadEncoded},{"core_library_decode_encoded",p.coreLibraryDecodeEncoded},{"core_library_prefill_encoded",p.coreLibraryPrefillEncoded},{"core_library_head_encoded",p.coreLibraryHeadEncoded}}.dump();'),
 ('        info+="; primary_full_isolated_library";\n#endif','        info+="; primary_full_isolated_library";\n#endif\n#ifdef GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY\n        info+="; primary_core_isolated_library";\n#endif')])
extra='''
option(GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY "Isolate unchanged ordinary primary kernels from raw MAIN without vector prefill" OFF)
if(GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_WIDE_PRIMARY_STORAGE OR NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS OR NOT GARNET_METAL_BODY_SHAPE)
  message(FATAL_ERROR "Primary core isolation requires primary/wide/lazy/body-shape")
 endif()
 if(GARNET_METAL_PRIMARY_VECTOR_PREFILL OR GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY OR GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY OR GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY)
  message(FATAL_ERROR "Primary core isolation excludes vector/full/prefill isolation and raw request residency")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY=1)
endif()
add_executable(garnet_primary_core_isolation_test test2026/metal/primary_core_isolation_test.cpp)
target_link_libraries(garnet_primary_core_isolation_test PRIVATE garnet_metal)
target_include_directories(garnet_primary_core_isolation_test PRIVATE third_party)
add_test(NAME metal_primary_core_isolation_reference COMMAND garnet_primary_core_isolation_test "${CMAKE_CURRENT_BINARY_DIR}/primary_core_projection_words.bin")
'''
p=s/'cmake/GarnetMetal.cmake';original=p.read_text();p.write_text(original+extra);transforms['cmake/GarnetMetal.cmake']=[dict(before='',after=extra,append=True)]
cpp=w/'primary_core_isolation_test.cpp';assert cpp.is_file();shutil.copy2(cpp,s/'test2026/metal/primary_core_isolation_test.cpp')
(w/'source-transformations.json').write_text(json.dumps(transforms,indent=2)+'\n')
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};assert len(manifest)==1721;delta=sorted(k for k,v in manifest.items()if seed.get(k)!=v);assert delta==sorted(list(transforms)+['test2026/metal/primary_core_isolation_test.cpp'])
assert all(manifest[k]==v for k,v in seed.items()if k not in transforms)
proof=dict(seed_files=1720,source_files=1721,changed_or_added=delta,all_original_shaders_codec_graph_session_runtime_xModel_image_CUDA_engine_and_original_tests_byte_preserved=True,all_transformed_original_text_inverse_exact=True,ordinary_primary_fragment_and_existing_minimal_preamble_unchanged=True,defaultOFF_old_FULL_vector_request_implementations_preserved=True,new_actual_encoded_route_fields_only_no_math_or_weight_change=True)
(w/'source-inverse-proof.json').write_text(json.dumps(proof,indent=2)+'\n');(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
oldwork=w.parent/'worldsense-packed-request-residency'
shutil.copytree(oldwork/'build-profile',w/'build-profile')
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib']:shutil.copy2(oldwork/name,w/name)
build=(oldwork/'build.py').read_text().replace('worldsense-packed-request-residency','worldsense-primary-core-isolation').replace("'GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY':packed,'GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS':'ON'if role=='fault'else'OFF',","'GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY':packed,")
ast.parse(build);(w/'build.py').write_text(build)
state=dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),experiment='WS-M5-70',goal='active',goal_complete=False,promotion=False,state='SOURCE_READY_BUILD_NEXT',work=str(w),source=str(s),seed_source=str(old),seed_commit=protocol['seed_commit'],seed_files=1720,source_files=1721,changed_or_added_vs56=delta,design_sha256=protocol['design_sha256'],consumer_audit_sha256=protocol['audit_sha256'],source_implemented=True,no_build_or_model_dispatched=True,live_job=None,prior69='valid-reject-published',prior69_commit='4bcce222f7dfd7712fbceb548b87894ec8b64312',normal_selected='WS10',normal_final=load(w/'previous69-published-checkpoint.json')['normal_final'],next_actions=['Build sixroles with frozen1721source; do not mutate consumed sources/drivers/libraries whilelive.','Then original8native+newcoretest perrole54 total, full186 and complete newwords/program partition/counters, independentpostterminal audit.','Prepare/freeze all original full learned guards and verifiers before model calls, then actualWorldSense functional4/resource8/WS10speed20/PRIMARY20/RAW20/mirrored48/canonical gates. No speed claim orpromotion.']);(b/'work/worldsense-next.json').write_text(json.dumps(state,indent=2)+'\n')
print('WS70 five-path source delta implemented with exact inverse; all old shader/math/runtime/CUDA/engine/tests byte-preserved; six-role driver ready, no build/model yet.')
