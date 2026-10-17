from pathlib import Path
import ast, datetime, hashlib, json, re, shutil

w = Path(__file__).resolve().parent
r = Path('/Users/shawn/CantorAI')
old = r/'out/research/worldsense-primary-gpu/source'
s = r/'out/research/worldsense-primary-vector/source'
assert not s.exists()
assert json.loads((w/'design-identity.json').read_text())['design_sha256'] == hashlib.sha256((w/'native-design-before-implementation.txt').read_bytes()).hexdigest()
shutil.copytree(old, s, ignore=shutil.ignore_patterns('__pycache__', '.git', 'results'))
(w/'protocol.json').write_text(json.dumps({
    'experiment':'WS-M5-38', 'declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'seed':'Frozen WS37 source; distinct four-row/four-K vector decode load. Original scalar prefill, codec, ownership and FP32 arithmetic unchanged.',
    'gates':'All original native/full-model/fault/corpus gates plus vector GPU reconstruction first; then NEW8resource10%Metal5%footprint,NEW20speed2%preservationbothmodelsTTFTtotalderiveddecodeSkillwall,NEW48fullfourworkers/twopairs individuallypassing.',
    'accounting':'Native counters encoded not completedGPU; source/build/model/input identities and all first/warm/failures retained. No externalCPythonimages/inference;no cameras;preserveCUDA;restoreWS10normal.',
    'hypothesis':'Explicit vector loads and shared metadata may reduce decode reconstruction costs. Compiler may already optimize scalar loads; register pressure may regress. No speed prediction.',
},indent=2))

p = s/'src/backends/metal/primary_block_shader.h'
before = p.read_text()
helper = r'''
// Packed admission proves row%4==0, k%4==0, inner%128==0 and four valid rows.
// At every decode iteration k..k+3 stays in a single 128-K block.
struct PrimaryQuad { ushort4 r0,r1,r2,r3; };
inline ushort4 primary_quad_row(uint4 sm,uint4 tags,uint base,uint row) {
    uint4 bytes=(sm>>(row*8))&255u;
    return ushort4((bytes&127u)|((bytes&128u)<<8)|((uint4(base)+((tags>>(row*4))&15u))<<7));
}
inline PrimaryQuad primary_quad(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    ulong meta=metadata[(row/4)*(inner/128)+k/128],at=(meta>>9)*256,index=(k%128)*4;
    PrimaryQuad q;
    if(meta&256ul) {
        uint4 sm=*(device const uint4*)(payload+at+index);
        uint2 packedTags=*(device const uint2*)(payload+at+512+index/2);
        uint4 tags=uint4(packedTags.x&65535u,packedTags.x>>16,packedTags.y&65535u,packedTags.y>>16);
        uint base=uint(meta)&255u;
        q.r0=primary_quad_row(sm,tags,base,0);q.r1=primary_quad_row(sm,tags,base,1);
        q.r2=primary_quad_row(sm,tags,base,2);q.r3=primary_quad_row(sm,tags,base,3);
    } else {
        uint4 lo=*(device const uint4*)(payload+at+index*2),hi=*(device const uint4*)(payload+at+index*2+16);
        uint4 even=uint4(lo.x,lo.z,hi.x,hi.z),odd=uint4(lo.y,lo.w,hi.y,hi.w);
        q.r0=ushort4(even&65535u);q.r1=ushort4(even>>16);
        q.r2=ushort4(odd&65535u);q.r3=ushort4(odd>>16);
    }
    return q;
}
// Correctness-only bounded readback exercises the exact tile helper used by decode.
kernel void primary_expand_vector_words(device const uchar* payload [[buffer(0)]],device const ulong* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=count)return;
    ulong i=start+id,row=i/inner,k=i%inner;
    PrimaryQuad q=primary_quad(payload,metadata,row-row%4,k-k%4,inner);
    ushort4 values=row%4==0?q.r0:row%4==1?q.r1:row%4==2?q.r2:q.r3;
    output[id]=uint(values[k%4])<<16;
}
'''
t = before.replace('kernel void primary_expand_words(', helper+'\nkernel void primary_expand_words(',1)
audit = []
def kernel(text,name):
    a=text.index('kernel void '+name+'(');b=text.index('{',a);depth=1;i=b+1
    while depth:
        if text[i]=='{':depth+=1
        elif text[i]=='}':depth-=1
        i+=1
    return text[a:i]
for name in ['primary_bf16_decode','primary_specialized_bf16_decode']:
    original=kernel(before,name);modified=original
    inner='ulong(fixed_inner)' if 'specialized' in name else 'p.inner'
    begin=modified.index('ushort4 b0=primary_row4(')
    end=modified.index('            for(uint j=0;',begin)
    prior=modified[begin:end]
    replacement='PrimaryQuad quad=primary_quad(payload,metadata,column,k,'+inner+');\n            ushort4 b0=quad.r0,b1=quad.r1,b2=quad.r2,b3=quad.r3;\n'
    modified=modified[:begin]+replacement+modified[end:]
    assert modified.replace(replacement,prior)==original
    t=t.replace(original,modified,1)
    audit.append(dict(kernel=name,prior_load_block=prior,new_load_block=replacement,inverse_load_substitution_exact=True,original_sha256=hashlib.sha256(original.encode()).hexdigest(),modified_sha256=hashlib.sha256(modified.encode()).hexdigest()))
for name in ['primary_expand_words','primary_bf16_gemm','primary_gemm_16_64_64','primary_gemm_32_64_32']:
    assert kernel(before,name)==kernel(t,name)
p.write_text(t)
(w/'vector-load-substitution-audit.json').write_text(json.dumps(audit,indent=2))

p=s/'src/backends/metal/metal_context.h';t=p.read_text()
t=t.replace('headEncoded=0;','headEncoded=0,vectorDecodeEncoded=0,vectorHeadEncoded=0;',1)
assert 'vectorHeadEncoded=0' in t
t=t.replace('public:\n    static MetalPrimaryStats PrimaryStats();','    std::vector<uint32_t> PrimaryReadWordsInternal(const MetalBF16Weight&,size_t start,size_t count,bool vector);\npublic:\n    static MetalPrimaryStats PrimaryStats();',1)
t=t.replace('    std::vector<uint32_t> PrimaryReadWords(const MetalBF16Weight&,size_t start,size_t count);','    std::vector<uint32_t> PrimaryReadWords(const MetalBF16Weight&,size_t start,size_t count);\n    std::vector<uint32_t> PrimaryReadVectorWords(const MetalBF16Weight&,size_t start,size_t count);',1)
p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text()
t=t.replace('head{0};','head{0},vectorDecode{0},vectorHead{0};',1)
t=t.replace('primaryCounters.head.load()};','primaryCounters.head.load(),primaryCounters.vectorDecode.load(),primaryCounters.vectorHead.load()};',1)
t=t.replace('primaryGemm32Wide,primaryExpand;','primaryGemm32Wide,primaryExpand,primaryExpandVector;',1)
needle='        if(!impl_->kernels->primaryExpand)throw std::runtime_error("Cannot create primary expand_words pipeline");'
assert needle in t
t=t.replace(needle,needle+'\n        impl_->kernels->primaryExpandVector=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"primary_expand_vector_words"] error:&error];\n        if(!impl_->kernels->primaryExpandVector)throw std::runtime_error("Cannot create primary vector readback pipeline");',1)
needle='else if(o.columns>=100000)primaryCounters.head.fetch_add(1);else primaryCounters.decode.fetch_add(1);'
assert t.count(needle)==1
t=t.replace(needle,'else if(o.columns>=100000){primaryCounters.head.fetch_add(1);primaryCounters.vectorHead.fetch_add(1);}else{primaryCounters.decode.fetch_add(1);primaryCounters.vectorDecode.fetch_add(1);}',1)
needle='std::vector<uint32_t> MetalContext::PrimaryReadWords(const MetalBF16Weight& weight,size_t start,size_t count) {'
assert t.count(needle)==1
t=t.replace(needle,'''std::vector<uint32_t> MetalContext::PrimaryReadWords(const MetalBF16Weight& weight,size_t start,size_t count) {return PrimaryReadWordsInternal(weight,start,count,false);}
std::vector<uint32_t> MetalContext::PrimaryReadVectorWords(const MetalBF16Weight& weight,size_t start,size_t count) {return PrimaryReadWordsInternal(weight,start,count,true);}
std::vector<uint32_t> MetalContext::PrimaryReadWordsInternal(const MetalBF16Weight& weight,size_t start,size_t count,bool vector) {''',1)
t=t.replace('(void)weight;(void)start;(void)count;throw std::logic_error("Primary weight support disabled");','(void)weight;(void)start;(void)count;(void)vector;throw std::logic_error("Primary weight support disabled");',1)
t=t.replace('[encoder setComputePipelineState:impl_->kernels->primaryExpand];','[encoder setComputePipelineState:vector?impl_->kernels->primaryExpandVector:impl_->kernels->primaryExpand];',1)
p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text()
t=t.replace('{"head_encoded",p.headEncoded}}.dump();','{"head_encoded",p.headEncoded},{"vector_decode_encoded",p.vectorDecodeEncoded},{"vector_head_encoded",p.vectorHeadEncoded}}.dump();',1)
t=t.replace('info+="; primary_bf16_blocks";','info+="; primary_bf16_blocks; primary_vector_decode";',1)
assert 'vector_head_encoded' in t;p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text();t=t.replace("['decode_encoded','prefill_encoded','head_encoded']","['decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded']",1);p.write_text(t)

# Add vector checks without replacing any scalar/mathematical controls.
p=s/'test2026/metal/primary_storage_test.cpp';t=p.read_text()
t=t.replace('auto actual=gpu.PrimaryReadWords(w,0,bits.size());','auto vectorActual=gpu.PrimaryReadVectorWords(w,0,bits.size());auto actual=gpu.PrimaryReadWords(w,0,bits.size());need(actual==vectorActual,"Finite scalar/vector GPU words differ");',1)
t=t.replace('auto actual=gpu.PrimaryReadWords(w,0,mix.size());','auto vectorActual=gpu.PrimaryReadVectorWords(w,0,mix.size());auto actual=gpu.PrimaryReadWords(w,0,mix.size());need(actual==vectorActual,"Mixed scalar/vector GPU words differ");',1)
needle=' need(gpu.PrimaryReadWords(shared,7,100)[0]==uint32_t(bits[7])<<16,"Nonzero read offset mismatch");'
assert needle in t
t=t.replace(needle,needle+'''
 for(size_t start:{size_t(7),size_t(125),size_t(507),bits.size()-100}){auto v=gpu.PrimaryReadVectorWords(shared,start,100);for(size_t j=0;j<100;++j)need(v[j]==uint32_t(bits[start+j])<<16,"Vector partial/block/tail offset mismatch");}
 for(int which:{0,1,2}){bool bad=false;try{gpu.PrimaryReadVectorWords(shared,which==0?bits.size()-1:0,which==0?2:which==1?0:8*1024*1024+1);}catch(...){bad=true;}need(bad,"Vector reader invalid bound accepted");}
 gpu.Begin();{bool bad=false;try{gpu.PrimaryReadVectorWords(shared,0,16);}catch(...){bad=true;}need(bad,"Vector reader admitted an open batch");}gpu.Abort();
''',1)
p.write_text(t)

# Copy final WS37 drivers, never its old source-mutating generators.
drivers=['build.py','build_fault.py','build_corpus.py','primary_corpus_test.cpp','full_correctness.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','make_resident_fixtures.py','wait_cantor.py']
for name in drivers:
    text=(w.parent/'worldsense-primary-gpu'/name).read_text().replace('worldsense-primary-gpu','worldsense-primary-vector')
    (w/name).write_text(text)
for name in ['fixture-1-8','fixture-2-64']:
    shutil.copytree(w.parent/'worldsense-primary-gpu'/name,w/name)
p=w/'primary_corpus_test.cpp';t=p.read_text()
needle='auto actual=gpu.PrimaryReadWords(w,start,count);need(actual.size()==count,"GPU read size changed");'
assert needle in t
t=t.replace(needle,needle+'auto vectorActual=gpu.PrimaryReadVectorWords(w,start,count);need(actual==vectorActual,"Actual checkpoint scalar/vector GPU words differ");',1)
t=t.replace('{"GPU_original_FP32_words_checked",checked}', '{"GPU_original_FP32_words_checked",checked},{"GPU_vector_original_FP32_words_checked",checked}')
t=t.replace('{"GPU_original_FP32_words_checked",wordsChecked}', '{"GPU_original_FP32_words_checked",wordsChecked},{"GPU_vector_original_FP32_words_checked",wordsChecked}')
t=t.replace('full_actual_packed_corpus_GPU_bits_exact_not_inference_validated','full_actual_both_readers_packed_corpus_GPU_bits_exact_not_inference_validated')
p.write_text(t)
p=w/'api_state_guard.py';t=p.read_text();needle="assert ('primary_bf16_blocks' in call.get('backend',''))==bool(expected);";assert needle in t;t=t.replace(needle,needle+"assert ('primary_vector_decode' in call.get('backend',''))==bool(expected);",1)
needle="  assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])"
assert needle in t;t=t.replace(needle,needle+"\n  assert delta['vector_decode_encoded']==delta['decode_encoded'] and delta['vector_head_encoded']==delta['head_encoded']",1);p.write_text(t)
p=w/'full_correctness.py';t=p.read_text();needle=" print('ALL398PACKED_ACTUAL_WEIGHTS_GPU_WORDS PASS',flush=True)";assert needle in t;t=t.replace(needle," assert json.loads((w/'corpus-GPU-bits.json').read_text())['totals']['GPU_vector_original_FP32_words_checked']==5921308672\n"+needle,1)
t=t.replace("'all398actualpacked_weights_GPU_bit_reconstruction':True", "'all398actualpacked_weights_GPU_bit_reconstruction':True,'all398actualpacked_weights_vector_GPU_bit_reconstruction':True")
p.write_text(t)

changed=[]
for p in s.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.read_bytes() != (old/p.relative_to(s)).read_bytes():changed.append(str(p.relative_to(s)))
expected=['runtime/metal_serving.py','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/primary_block_shader.h','src/entry/metal_main.cpp','test2026/metal/primary_storage_test.cpp']
assert sorted(changed)==sorted(expected),changed
for rel in ['src/backends/metal/primary_block_codec.h','src/backends/metal/body_shape_shader.h']:
    assert (s/rel).read_bytes()==(old/rel).read_bytes()
for p in old.rglob('*'):
    if p.is_file() and ('cuda' in str(p.relative_to(old)).lower() or p.suffix=='.cu'):
        assert p.read_bytes()==(s/p.relative_to(old)).read_bytes()
get_model=lambda p:next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Model')
assert ast.dump(get_model(s/'runtime/metal_serving.py'),include_attributes=False)==ast.dump(get_model(old/'runtime/metal_serving.py'),include_attributes=False)
(w/'implementation-audit.json').write_text(json.dumps({'changed_files_from_WS37':changed,'entire_Model_class_AST_unchanged':True,'CUDA_codec_original_body_shader_unchanged':True,'prefill_scalar_reader_kernels_identical':True,'decode_inverse_load_substitution_exact':True,'scope':'Static invariants only, not runtime correctness or speed.'},indent=2))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
print('WS38 six-file vector decode delta prepared; source/driver parsing and static invariants PASS',flush=True)
