"""CPU-only source/flag/admission review before native tests; no speed claim."""
from pathlib import Path
import json,hashlib,re,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI')
s=r/'out/research/worldsense-dense-prefetch/source';old=r/'out/research/worldsense-isolated-primary/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((w/'correctness-source-manifest.json').read_text());seed=json.loads((w/'seed-source-manifest.json').read_text())
assert len(m)==1720 and len(seed)==1718 and all(sha(s/k)==v for k,v in m.items())
changed=sorted(k for k,v in m.items()if seed.get(k)!=v)
assert changed==sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/entry/metal_main.cpp','src/backends/metal/dense_prefetch_shader.h','test2026/metal/dense_prefetch_test.cpp'])
before=(old/'src/backends/metal/metal_context.mm').read_text();after=(s/'src/backends/metal/metal_context.mm').read_text()
literal=lambda t:re.findall(r'R"([A-Za-z0-9_]*)\((.*?)\)\1"',t,re.S)
assert literal(before)==literal(after)and len(literal(before))>=3
assert (s/'src/backends/metal/body_shape_shader.h').read_bytes()==(old/'src/backends/metal/body_shape_shader.h').read_bytes()
assert (s/'test2026/metal/primary_projection_test.cpp').read_bytes()==(old/'test2026/metal/primary_projection_test.cpp').read_bytes()
body=(s/'src/backends/metal/body_shape_shader.h').read_text();look=(s/'src/backends/metal/dense_prefetch_shader.h').read_text()
assert 'if(k+128<ulong(fixed_inner))next=dense_lookahead_load(a,weight,column,k+128);'in look
assert 'for(;k<ulong(fixed_inner);k+=128)'in look and 'ulong k=ulong(lane)*4;'in look
math='''float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;'''
normalized=look.replace('current.b0','b0').replace('current.b1','b1').replace('current.b2','b2').replace('current.b3','b3').replace('current.x','x')
space=lambda t:re.sub(r'\s+','',t)
assert space(math)in space(body)and space(math)in space(normalized)
reduce='sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));'
assert reduce in body and reduce in look
assert 'ulong column=ulong(group)*16+simd*4;'in body and 'ulong column=ulong(group)*16+simd*4;'in look
unit=(s/'test2026/metal/dense_prefetch_test.cpp').read_text();original=(old/'test2026/metal/primary_projection_test.cpp').read_text()
bound='double bound=rows<=2?1e-4+2e-6*l1:5e-6+5e-5*std::abs(ref);'
assert bound in unit and bound in original and 'require(cases==93&&passes==186' in unit
assert 'inner,false,false,false)'in unit and 'require(!w.PrimaryReady()'in unit
flags=['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS','DENSE_PREFETCH','PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','DIRECT_FLOAT_WEIGHT_UPLOAD','PRIMARY_VECTOR_PREFILL_K32_ONLY','CACHED_MATRIX_METADATA','PRIMARY_FULL_ISOLATED_LIBRARY']
observed={}
for mode in ['off','primary','on','fault']:
 b=r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode)
 cache={k:v for k,typ,v in re.findall(r'^(GARNET_METAL_\w+):(\w+)=(.*)$',(b/'CMakeCache.txt').read_text(),re.M)}
 expected={name:('OFF'if mode=='off'else'ON'if mode in ['on','fault']or i<4 else'OFF')for i,name in enumerate(flags)}
 assert all(cache['GARNET_METAL_'+k]==v for k,v in expected.items()),(mode,expected)
 assert cache['GARNET_METAL_PRIMARY_TEST_FAULTS']==('ON'if mode=='fault'else'OFF')
 observed[mode]=expected
(w/'source-proof.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_shader_literals_body_and_tests_unchanged=True,raw_shader_literal_blocks=len(literal(before)),source_files=1720,changed_files=changed,new_MSL_per_output_product_accumulation_and_SIMD_order_inverse_pass=True,new_lookahead_bounds_and_target_admission_source_review_pass=True,original_numeric_bounds_and_93shape186output_scope_retained=True,ten_method_flags_four_roles=observed,CUDA_xModel_graph_session_and_all_other_source_exact54=True,no_GPU_or_performance_claim=True),indent=2)+'\n')
print('WS60 source/literal/arithmetic-order/guard/numeric-bound/10flag4role audit PASS')
