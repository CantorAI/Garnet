"""Implement only prospective packed grouping/counter/diagnostic delta."""
from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-primary-decode-groups2';old=w.parent/'worldsense-minimal-full-prefill';seed=r/'out/research/worldsense-minimal-full-prefill/source';s=r/'out/research/worldsense-primary-decode-groups2/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();protocol=load(w/'protocol.json');m=load(w/'seed-source-manifest.json')
assert protocol['state']=='DESIGN_BEFORE_SOURCE'and sha(w/'native-design-before-implementation.txt')==protocol['design_sha256']and not s.exists();assert all(sha(seed/k)==v for k,v in m.items());shutil.copytree(seed,s,ignore=shutil.ignore_patterns('__pycache__','.git'))
ops={}
def edit(rel,changes):
 p=s/rel;original=p.read_text();text=original;record=[]
 for before,after in changes:
  assert text.count(before)==1,(rel,before,text.count(before));text=text.replace(before,after);record.append(dict(before=before,after=after,count=1))
 inverse=text
 for op in reversed(record):assert inverse.count(op['after'])==1;inverse=inverse.replace(op['after'],op['before'])
 assert inverse==original;p.write_text(text);ops[rel]=record
edit('src/backends/metal/metal_context.h',[('coreLibraryHeadEncoded=0;','coreLibraryHeadEncoded=0,decodeGroups2Encoded=0,headGroups2Encoded=0;')])
edit('src/backends/metal/metal_context.mm',[
 ('#include "primary_block_shader.h"','#include "primary_block_shader.h"\n#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2\n#include "primary_decode_groups2_shader.h"\n#endif'),
 ('corePrefill{0},coreHead{0};','corePrefill{0},coreHead{0},groups2Decode{0},groups2Head{0};'),
 ('primaryCounters.coreHead.load()};','primaryCounters.coreHead.load(),primaryCounters.groups2Decode.load(),primaryCounters.groups2Head.load()};'),
 ('        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];','''#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2
        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:PrimaryDecodeGroups2Program().c_str()]];
#else
        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];
#endif'''),
 ('        if(!impl_->kernels->primaryDecode)throw std::runtime_error("Cannot create primary bf16_decode pipeline");','''        if(!impl_->kernels->primaryDecode)throw std::runtime_error("Cannot create primary bf16_decode pipeline");
#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2
        if(impl_->kernels->primaryDecode.threadExecutionWidth!=32||impl_->kernels->primaryDecode.maxTotalThreadsPerThreadgroup<64)
            throw std::runtime_error("Packed groups2 requires a 32-lane 64-thread decode pipeline");
#endif'''),
 ('        bool noTail=o.columns%(small?4:16)==0;','''        bool noTail=o.columns%(small?4:16)==0;
#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2
        if(primary)noTail=o.columns%8==0;
#endif'''),
 ('        if(pipeline&&(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<(small?64:128)))pipeline=nil;','''        size_t requiredThreads=small?64:128;
#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2
        if(primary)requiredThreads=64;
#endif
        if(pipeline&&(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<requiredThreads))pipeline=nil;'''),
 ('        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads)','''#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2
        if(primary&&a.rows<=2){threads=64;decodeColumns=8;}
#endif
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads)'''),
 ('        [encoder dispatchThreadgroups:a.rows<=2?MTLSizeMake((o.columns+decodeColumns-1)/decodeColumns,1,1):MTLSizeMake((o.columns+tileColumns-1)/tileColumns,(o.rows+tileRows-1)/tileRows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];','''        [encoder dispatchThreadgroups:a.rows<=2?MTLSizeMake((o.columns+decodeColumns-1)/decodeColumns,1,1):MTLSizeMake((o.columns+tileColumns-1)/tileColumns,(o.rows+tileRows-1)/tileRows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2
        if(primary&&a.rows<=2){if(o.columns>=100000)primaryCounters.groups2Head.fetch_add(1);else primaryCounters.groups2Decode.fetch_add(1);}
#endif''')])
edit('src/entry/metal_main.cpp',[
 ('{"core_library_head_encoded",p.coreLibraryHeadEncoded}}.dump();','{"core_library_head_encoded",p.coreLibraryHeadEncoded},{"decode_groups2_encoded",p.decodeGroups2Encoded},{"head_groups2_encoded",p.headGroups2Encoded}}.dump();'),
 ('        info+="; primary_core_isolated_library";\n#endif','        info+="; primary_core_isolated_library";\n#endif\n#ifdef GARNET_METAL_PRIMARY_DECODE_GROUPS2\n        info+="; primary_decode_groups2";\n#endif')])
extra='''
option(GARNET_METAL_PRIMARY_DECODE_GROUPS2 "Use two SIMD groups only for sole-primary generic/specialized decode and heads" OFF)
if(GARNET_METAL_PRIMARY_DECODE_GROUPS2)
 if(NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_WIDE_PRIMARY_STORAGE OR NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS OR NOT GARNET_METAL_BODY_SHAPE OR NOT GARNET_METAL_PRIMARY_VECTOR_PREFILL OR NOT GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY OR NOT GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY)
  message(FATAL_ERROR "Packed groups2 requires primary/wide/lazy/body-shape/vector/prefill-isolation/full-isolation")
 endif()
 if(GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY OR GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY OR GARNET_METAL_PRIMARY_DECODE_GROUPS8)
  message(FATAL_ERROR "Packed groups2 excludes CORE, raw request residency and GROUPS8")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_PRIMARY_DECODE_GROUPS2=1)
endif()
add_executable(garnet_primary_decode_groups2_test test2026/metal/primary_decode_groups2_test.cpp)
target_link_libraries(garnet_primary_decode_groups2_test PRIVATE garnet_metal)
target_include_directories(garnet_primary_decode_groups2_test PRIVATE third_party)
add_test(NAME metal_primary_decode_groups2_reference COMMAND garnet_primary_decode_groups2_test "${CMAKE_CURRENT_BINARY_DIR}/primary_decode_groups2_words.bin")
'''
p=s/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+extra);ops['cmake/GarnetMetal.cmake']=[dict(before='',after=extra,append=True)]
for name,rel in [('primary_decode_groups2_shader.h','src/backends/metal/primary_decode_groups2_shader.h'),('primary_decode_groups2_test.cpp','test2026/metal/primary_decode_groups2_test.cpp')]:shutil.copy2(w/name,s/rel)
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};delta=sorted(k for k,v in manifest.items()if m.get(k)!=v);assert len(manifest)==1723 and len(delta)==6 and all(manifest[k]==v for k,v in m.items()if k not in ops)
(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'source-transformations.json').write_text(json.dumps(ops,indent=2)+'\n');(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1721,source_files=1723,changed_or_added=delta,all_changed_original_host_text_inverse_exact=True,original_primary_shader_FP_helpers_codec_graph_session_Model_runtime_xModels_images_CUDA_engine_and_original_tests_byte_preserved=True,selector_only_two_output_origins_with_host_noTail_and_dispatch_geometry_changes=True),indent=2)+'\n')
shutil.copytree(old/'build-profile',w/'build-profile')
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib']:shutil.copy2(old/name,w/name)
t=(old/'build.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups2');before=" flags['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']='OFF'";assert t.count(before)==1;t=t.replace(before,before+"\n flags['GARNET_METAL_PRIMARY_DECODE_GROUPS2']='ON'if role in ['on','fault']else'OFF'");t=t.replace('len(manifest)==1721','len(manifest)==1723');ast.parse(t);(w/'build.py').write_text(t)
p=b/'work/worldsense-next.json';d=load(p);d.update(state='SOURCE_READY_BUILD_NEXT',source_implemented=True,source_files=1723,changed_or_added_vs71=delta);p.write_text(json.dumps(d,indent=2)+'\n');print('WS73 sixpath1723source ready; host inverse exact/originalshader FP helpers unchanged; sixrole driver ready, no build/model yet.')
