from pathlib import Path
import json,hashlib,shutil,re,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-sole-interleaved-text';old=r/'out/research/worldsense-direct-interleaved-prefill/source';s=r/'out/research/worldsense-sole-interleaved-text/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((w/'protocol.json').read_text())['design_sha256']==sha(w/'native-design-before-implementation.txt');seed=json.loads((w/'read-only-consumer-audit.json').read_text())['seed_manifest'];assert len(seed)==1721 and all(sha(old/k)==v for k,v in seed.items());s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
ops={}
def change(rel,a,z,count=1):
 p=s/rel;t=p.read_text();assert t.count(a)==count,(rel,a,t.count(a));p.write_text(t.replace(a,z));ops.setdefault(rel,[]).append(dict(before=a,after=z,count=count))
def append(rel,z):
 p=s/rel;p.write_text(p.read_text()+z);ops.setdefault(rel,[]).append(dict(before='',after=z,count=1,append=True))
m='src/backends/metal/metal_context.mm';e='src/entry/metal_main.cpp';cm='cmake/GarnetMetal.cmake'
a='''        if(soleAvailable&&!floatPrefill&&!visionPrefill&&
           ((rows==6144&&columns==2048)||(rows==2048&&columns==6144))) {'''
z='''        const bool soleMlpShape=(rows==6144&&columns==2048)||(rows==2048&&columns==6144);
        bool soleShape=soleMlpShape;
#ifdef GARNET_METAL_SOLE_INTERLEAVED_TEXT
        soleShape=soleShape||(columns==2048&&(rows==2048||rows==1024||rows==151936));
#endif
        if(soleAvailable&&!floatPrefill&&!visionPrefill&&soleShape) {'''
change(m,a,z)
change(m,'w->buffer.length!=kSoleScratchBytes','w->buffer.length!=Elements(w->rows,w->columns)*sizeof(uint16_t)',3)
needle='        info+="; sole_direct_interleaved_prefill";\n#endif';change(e,needle,needle+'\n#ifdef GARNET_METAL_SOLE_INTERLEAVED_TEXT\n        info+="; sole_interleaved_text";\n#endif')
append(cm,'''
option(GARNET_METAL_SOLE_INTERLEAVED_TEXT "Extend sole direct BF16 layout to exact raw text attention and vocabulary weights" OFF)
if(GARNET_METAL_SOLE_INTERLEAVED_TEXT)
 if(NOT GARNET_METAL_SOLE_INTERLEAVED_MLP OR NOT GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL)
  message(FATAL_ERROR "Sole interleaved text requires sole MLP storage and direct prefill")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_SOLE_INTERLEAVED_TEXT=1)
endif()
add_executable(garnet_sole_text_test test2026/metal/sole_text_test.cpp)
target_link_libraries(garnet_sole_text_test PRIVATE garnet_metal)
add_test(NAME metal_sole_text_reference COMMAND garnet_sole_text_test "${CMAKE_CURRENT_BINARY_DIR}/sole_text_projection_words.bin")
''')
# Original 28-record MLP test remains byte-exact. Distinct ninth test keeps complete cases for each new shape.
test=(old/'test2026/metal/sole_interleaved_test.cpp').read_text();t=test
changes=[]
def tc(a,z,count=1):
 global t
 assert t.count(a)==count,(a,t.count(a));t=t.replace(a,z);changes.append(dict(before=a,after=z,count=count))
tc('#ifdef GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL\n direct=true;\n#endif\n#ifdef GARNET_METAL_SOLE_INTERLEAVED_MLP\n enabled=true;\n#endif','#ifdef GARNET_METAL_SOLE_INTERLEAVED_TEXT\n enabled=true;direct=true;\n#endif')
tc(''' for(size_t inner:{2048,6144}){
 size_t columns=inner==2048?6144:2048;size_t count=inner*columns;''',''' for(size_t columns:{2048,1024,151936}){
 size_t inner=2048;size_t count=inner*columns;''')
tc('prepared==(rows<=2)','prepared==(rows<=2&&columns<100000)')
tc('need(gpu.PrepareBF16BodyShape(a,bypass,one)&&gpu.PrepareBF16BodyShape(a,weight,two),"Separate cache prepare failed");','need(gpu.PrepareBF16BodyShape(a,bypass,one)==(columns<100000)&&gpu.PrepareBF16BodyShape(a,weight,two)==(columns<100000),"Separate cache/head admission failed");')
tc('need(cases==14&&records==28,"Missing projection coverage");','need(cases==21&&records==42,"Missing expanded text projection coverage");')
(s/'test2026/metal/sole_text_test.cpp').write_text(t);(w/'sole_text_test.cpp').write_text(t)
# Invert the new test to its original full MLP template, preserving math/limits/lifecycle word-for-word.
inverse=t
for op in reversed(changes):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==test
for rel,records in ops.items():
 inverse=(s/rel).read_text()
 for op in reversed(records):
  if op.get('append'):assert inverse.endswith(op['after']);inverse=inverse[:-len(op['after'])]
  else:assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse==(old/rel).read_text(),rel
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()};assert len(manifest)==1722 and all(manifest[k]==v for k,v in seed.items()if k not in ops)
for name,data in [('seed-source-manifest.json',seed),('correctness-source-manifest.json',manifest),('source-transformations.json',ops),('new-native-test-template-proof.json',dict(operations=changes,seed_sha256=hashlib.sha256(test.encode()).hexdigest(),current_sha256=hashlib.sha256(t.encode()).hexdigest(),original_FP64_bits_lifecycle_fault_limits_inverse_exact=True))]:(w/name).write_text(json.dumps(data,indent=2)+'\n')
proof=dict(seed_files=1721,current_files=1722,changed_or_added=sorted(k for k,v in manifest.items()if seed.get(k)!=v),all_shader_files_graph_session_xModel_image_CUDA_engine_original_eight_tests_unchanged=True,original_shader_literals_and_63_decode_graph_session_xModel_image_CUDA_original_tests_unchanged=True,original28_MLP_test_byte_exact=True)
(w/'source-inverse-proof.json').write_text(json.dumps(proof,indent=2)+'\n');shutil.copytree(b/'work/worldsense-direct-interleaved-prefill/build-profile',w/'build-profile')
print(json.dumps(proof))
