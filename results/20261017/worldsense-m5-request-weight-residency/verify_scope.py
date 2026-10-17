"""Independent exact inverse of lifecycle additions; no performance claim."""
from pathlib import Path
import json,hashlib,ast
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-request-weight-residency/source';seed=r/'out/research/worldsense-isolated-primary/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((w/'ws54-seed-source-manifest.json').read_text());new=json.loads((w/'correctness-source-manifest.json').read_text())
assert len(old)==1718 and len(new)==1720
assert all(sha(seed/k)==h for k,h in old.items())and all(sha(s/k)==h for k,h in new.items())
changed={k for k,h in new.items()if old.get(k)!=h}
assert changed=={'cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/weight_residency_budget.h','src/entry/metal_main.cpp','test2026/metal/request_residency_test.cpp'}
def remove_blocks(text):
 lines=text.splitlines(keepends=True);out=[];depth=0
 for line in lines:
  if not depth and line.strip()=='#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY':depth=1;continue
  if depth:
   if line.lstrip().startswith(('#if ','#if\t','#ifdef ','#ifndef ')):depth+=1
   if line.strip()=='#endif':depth-=1
   continue
  out.append(line)
 assert depth==0
 return ''.join(out)
def equal(rel,text):assert text==(seed/rel).read_text(),rel
for rel in ['src/backends/metal/metal_context.mm','src/backends/metal/decoder_session.cpp']:
 text=(s/rel).read_text();anchor='\nnamespace Garnet {\n#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY\n';assert text.count(anchor)==1
 original,addition=text.split(anchor,1);assert 'Request' in addition
 original=remove_blocks(original)
 if rel.endswith('decoder_session.cpp'):original=original.replace('#include <exception>\n','',1)
 equal(rel,original)
rel='src/backends/metal/metal_context.h';text=(s/rel).read_text();a=text.index('    bool BeginRequestWeightResidency(');b=text.index('    void Begin();',a);equal(rel,text[:a]+text[b:])
rel='src/backends/metal/decoder_session.h';text=remove_blocks((s/rel).read_text()).replace('    mutable std::mutex mutex_;','    std::mutex mutex_;',1)
for start,end in [('    bool BeginRequestResidency(','    bool Build('),('        APISET().AddFunc<1>("begin_request_residency",','        APISET().AddFunc<0>("close",')]:
 a=text.index(start);b=text.index(end,a);text=text[:a]+text[b:]
equal(rel,text)
rel='src/entry/metal_main.cpp';equal(rel,remove_blocks((s/rel).read_text()))
cm=(s/'cmake/GarnetMetal.cmake').read_text();before=(seed/'cmake/GarnetMetal.cmake').read_text();assert cm.startswith(before)
tail=cm[len(before):];assert tail.count('option(')==2 and tail.count(' OFF)')==2 and 'if(NOT GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY)'in tail
assert json.loads((w/'inference-expression-audit.json').read_text())
# Independently recover the original inference AST from its wrapper.
a=ast.parse((seed/'runtime/metal_serving.py').read_text());b=ast.parse((s/'runtime/metal_serving.py').read_text())
def model(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Model')
aa=next(n for n in model(a).body if isinstance(n,ast.FunctionDef)and n.name=='infer');bb=next(n for n in model(b).body if isinstance(n,ast.FunctionDef)and n.name=='infer')
wrapped=next(n for n in bb.body if isinstance(n,ast.Try));assert wrapped.handlers==[] and wrapped.orelse==[] and len(wrapped.finalbody)==1
body=wrapped.body[1:];assert isinstance(wrapped.body[0],ast.If)
assert isinstance(body[-1],ast.Return)and isinstance(body[-2],ast.Assign)and body[-2].targets[0].id=='result'
body[-2:]=[ast.Return(value=body[-2].value)]
idx=bb.body.index(wrapped);assert [x.targets[0].id for x in bb.body[idx-2:idx]]==['request_started','result'];bb.body=bb.body[:idx-2]+body
assert ast.dump(aa,include_attributes=False)==ast.dump(bb,include_attributes=False)
model(b).body[model(b).body.index(next(n for n in model(b).body if isinstance(n,ast.FunctionDef)and n.name=='infer'))]=aa
b.body=[n for n in b.body if not(isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='_request_scope_residency'for x in n.targets))]
assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
budget=json.loads((w/'budget-source-identity.json').read_text());assert sha(s/'src/backends/metal/weight_residency_budget.h')==budget['sha256']
for mode in ['off','primary','on','fault']:
 for part in ['main','prefill']:assert(w/f'actual-{mode}-{part}-program.metal').read_bytes()==(w.parent/'worldsense-isolated-primary'/f'actual-{mode}-{part}-program.metal').read_bytes()
ctx=(s/'src/backends/metal/metal_context.mm').read_text();assert 'constexpr uint64_t kRequestResidencyLimit=uint64_t(4)*1024*1024*1024;'in ctx and 'constexpr uint64_t kRequestResidencyLimit=64*1024;'in ctx and 'seen.size()>2048'in ctx
assert json.loads((w/'request-guards-result.json').read_text())['all_extra_request_trained_guards_pass']
assert json.loads((w/'request-session-guard.json').read_text())['all_cases_pass']
assert json.loads((w/'request-full-session-guard.json').read_text())['real197masters_global4GiB_contention_close_readmission_reset']
result=dict(wide_admission_scope_verified=True,original_context_and_session_code_exact_after_lifecycle_inverse=True,original_Model_infer_AST_exact_after_lifecycle_inverse=True,all_other_serving_AST_exact=True,original_math_shaders_codecs_xModels_CUDA_tests_exact=True,all_four_actual_programs_byte_exact_WS54=True,unchanged_WS32_budget_helper=True,request_limits_4GiB_2048_and_test_only_64KiB=True,trained_request_coverage_generation_invalid_nested_close_reset_contention_pass=True,source_files=1720,canonical_overlays=15,scope='Exact source inverse and executed correctness guards; no speed or physical residency claim.')
(w/'wide-scope-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
