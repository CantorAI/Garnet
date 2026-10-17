from pathlib import Path
import ast, datetime, hashlib, json, re, shutil

w=Path(__file__).resolve().parent
r=Path('/Users/shawn/CantorAI')
old=r/'out/research/worldsense-primary-vector/source'
s=r/'out/research/worldsense-primary-kmajor/source'
assert not s.exists()
assert json.loads((w/'design-identity.json').read_text())['design_sha256']==hashlib.sha256((w/'native-design-before-implementation.txt').read_bytes()).hexdigest()
seed=json.loads((w.parent/'worldsense-primary-vector/starting-source-manifest.json').read_text())
assert all(hashlib.sha256((old/rel).read_bytes()).hexdigest()==digest for rel,digest in seed.items())
shutil.copytree(old,s,ignore=shutil.ignore_patterns('__pycache__','.git','results'))
(w/'protocol.json').write_text(json.dumps({
    'experiment':'WS-M5-39','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'seed':'Frozen WS38 plus direct K-major load reconstruction and a compiled capability marker only.',
    'hypothesis':'Avoid row-vector reconstruction/transposition before original FP32 FMA loop. Compiler/register behavior may regress; no speed prediction.',
    'fixed':'Codec,64bit variable metadata,398matrix admission,primary-only ownership,128thread geometry,prefill,FP32 arithmetic,counters,profiles,budgets,CUDA.',
    'gates':'All original native/full-model/fault/dualGPUcorpus checks; NEW8resource10%Metal5%footprint; NEW20speedbothmodelsTTFTtotalderiveddecodeandSkillwall2%preservation; NEW48fourworker3frame/querytwopairs independentlypass beforepromotion.',
    'accounting':'Native/XLang3 image/inference; CPython orchestration only. No camera,competingGPU/build/archivecopyduringtiming. Preserve first/warm/failures/source/model/build/SDK identities. Restore WS10 after each phase.',
},indent=2))

p=s/'src/backends/metal/primary_block_shader.h';before=p.read_text()
a=before.index('struct PrimaryQuad ');b=before.index('// Correctness-only bounded readback',a)
helper=r'''struct PrimaryQuad { ushort4 k0,k1,k2,k3; };
inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    uint4 signMant=uint4(as_type<uchar4>(sm));
    uint4 exponents=uint4(tags,tags>>4,tags>>8,tags>>12)&15u;
    return ushort4((signMant&127u)|((signMant&128u)<<8)|((exponents+base)<<7));
}
inline PrimaryQuad primary_quad(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    ulong meta=metadata[(row/4)*(inner/128)+k/128],at=(meta>>9)*256,index=(k%128)*4;
    PrimaryQuad q;
    if(meta&256ul) {
        uint4 sm=*(device const uint4*)(payload+at+index);
        ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+index/2));
        uint base=uint(meta)&255u;
        q.k0=primary_quad_kword(sm.x,tags.x,base);q.k1=primary_quad_kword(sm.y,tags.y,base);
        q.k2=primary_quad_kword(sm.z,tags.z,base);q.k3=primary_quad_kword(sm.w,tags.w,base);
    } else {
        uint4 lo=*(device const uint4*)(payload+at+index*2),hi=*(device const uint4*)(payload+at+index*2+16);
        q.k0=as_type<ushort4>(lo.xy);q.k1=as_type<ushort4>(lo.zw);
        q.k2=as_type<ushort4>(hi.xy);q.k3=as_type<ushort4>(hi.zw);
    }
    return q;
}
'''
t=before[:a]+helper+before[b:]
old_reader='ushort4 values=row%4==0?q.r0:row%4==1?q.r1:row%4==2?q.r2:q.r3;\n    output[id]=uint(values[k%4])<<16;'
new_reader='ushort4 values=k%4==0?q.k0:k%4==1?q.k1:k%4==2?q.k2:q.k3;\n    output[id]=uint(values[row%4])<<16;'
assert t.count(old_reader)==1;t=t.replace(old_reader,new_reader,1)

def kernel(text,name):
    start=text.index('kernel void '+name+'(');end=text.index('{',start);depth=1;end+=1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return text[start:end]

audit=[]
for name in ['primary_bf16_decode','primary_specialized_bf16_decode']:
    original=kernel(before,name);changed=original
    old_load='ushort4 b0=quad.r0,b1=quad.r1,b2=quad.r2,b3=quad.r3;'
    new_load='ushort4 weights[4]={quad.k0,quad.k1,quad.k2,quad.k3};'
    old_convert='float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);'
    new_convert='float4 w=as_type<float4>(uint4(weights[j])<<16);'
    assert changed.count(old_load)==changed.count(old_convert)==1
    changed=changed.replace(old_load,new_load).replace(old_convert,new_convert)
    assert changed.replace(new_convert,old_convert).replace(new_load,old_load)==original
    t=t.replace(original,changed,1)
    audit.append({'kernel':name,'inverse_load_substitution_exact':True,'old_load':old_load,'new_load':new_load,'old_convert':old_convert,'new_convert':new_convert,'original_sha256':hashlib.sha256(original.encode()).hexdigest(),'new_sha256':hashlib.sha256(changed.encode()).hexdigest()})
for name in ['primary_expand_words','primary_bf16_gemm','primary_gemm_16_64_64','primary_gemm_32_64_32']:
    assert kernel(before,name)==kernel(t,name)
p.write_text(t)
(w/'kmajor-load-substitution-audit.json').write_text(json.dumps(audit,indent=2))
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='info+="; primary_bf16_blocks; primary_vector_decode";';assert t.count(needle)==1
p.write_text(t.replace(needle,'info+="; primary_bf16_blocks; primary_vector_decode; primary_kmajor_decode";',1))

# Final validated correctness drivers, with metadata labels corrected before use.
drivers=['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py']
for name in drivers:
    (w/name).write_text((w.parent/'worldsense-primary-vector'/name).read_text().replace('worldsense-primary-vector','worldsense-primary-kmajor'))
for name in ['fixture-1-8','fixture-2-64']:
    shutil.copytree(w.parent/'worldsense-primary-vector'/name,w/name)
p=w/'api_state_guard.py';t=p.read_text();needle="assert ('primary_vector_decode' in call.get('backend',''))==bool(expected);";assert needle in t
t=t.replace(needle,needle+"assert ('primary_kmajor_decode' in call.get('backend',''))==bool(expected);",1)
assert 'lazy_defaults_expected=expected' in t
t=t.replace('lazy_defaults_expected=expected','primary_expected=expected,lazy_defaults_expected=False')
p.write_text(t)
p=w/'linear_guard.py';t=p.read_text().replace('Four-input trained linear lazy-retirement/generic-resident parity','Four-input trained linear generic-resident parity');p.write_text(t)
(w/'guard-scope-note.json').write_text(json.dumps({'metadata_label_fix':'Before execution, replace inherited lazy_defaults_expected=expected with primary_expected=expected,lazy_defaults_expected=False; correct the old lazy retirement success-message label. Test assertions/inputs/outputs/numerical bounds unchanged except new K-major compiled capability check.','lazy_policy':'OFF both; old lazy-allocation fault block disabled. Primary stage1/2 faults execute in native C++ fault build; owner counters not driver leak proof.'},indent=2))

changed=[str(p.relative_to(s))for p in s.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.read_bytes()!=(old/p.relative_to(s)).read_bytes()]
assert sorted(changed)==['src/backends/metal/primary_block_shader.h','src/entry/metal_main.cpp'],changed
raw_strings=lambda p:re.findall(r'R"METAL\((.*?)\)METAL"',p.read_text(),re.S)
assert raw_strings(s/'src/backends/metal/metal_context.mm')==raw_strings(old/'src/backends/metal/metal_context.mm')
cuda=[p for p in old.rglob('*')if p.is_file()and('cuda'in str(p.relative_to(old)).lower()or p.suffix=='.cu')]
assert all(p.read_bytes()==(s/p.relative_to(old)).read_bytes()for p in cuda)
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'implementation-audit.json').write_text(json.dumps({'changed_files_from_WS38':changed,'CUDA_unchanged_files':len(cuda),'all_other_source_files_identical':True,'scalar_prefill_reader_kernels_identical':True,'codec_64bit_metadata_ownership_geometry_Model_wrapper_counters_unchanged':True,'inverse_decode_weight_load_substitution_exact':True,'scope':'Static source proof only; runtime correctness and speed unproven.'},indent=2))
manifest={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()and '__pycache__'not in p.parts}
(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2))
(w/'seed-source-verification.json').write_text(json.dumps({'all_WS38_seed_files_match_frozen_manifest':True,'count':len(seed),'seed_archive':'Garnet/results/20261016/worldsense-m5-primary-vector/starting-source-manifest.json'},indent=2))
print('WS39 two-file direct K-major delta prepared; static proof and driver parsing PASS',flush=True)
