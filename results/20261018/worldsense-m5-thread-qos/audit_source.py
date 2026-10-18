"""Frozen WS59 versus WS54 numerical-source and local-boundary audit."""
from pathlib import Path
import hashlib,json,re
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-thread-qos/source');old=s.parents[1]/'worldsense-isolated-primary/source'
manifest=json.loads((w/'gateway-source-manifest.json').read_text());assert len(manifest)==1723
assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
oldfiles=json.loads((w.parent/'worldsense-isolated-primary/correctness-source-manifest.json').read_text());assert len(oldfiles)==1718;assert all(hashlib.sha256((old/k).read_bytes()).hexdigest()==h for k,h in oldfiles.items())
changed=sorted(k for k,h in manifest.items()if oldfiles.get(k)!=h);assert not set(oldfiles)-set(manifest)
expected=sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/entry/metal_main.cpp','src/entry/native_worker_support.cpp','src/entry/native_worker_support.h','src/entry/native_text_worker.cpp','src/entry/native_text_worker.h','runtime/native_text_worker.py']);assert changed==expected
p='runtime/metal_serving.py';v=(s/p).read_text();v=v.replace("_native_gateway_clock = 'native_text_worker' in garnet.backend_info()\n",'');v=v.replace('        result = dict(input_prompt_sha256=','        return dict(input_prompt_sha256=');v=v.replace("        if _native_gateway_clock:\n            result['native_model_start_monotonic'] = start\n            result['native_first_token_monotonic'] = first\n        return result\n",'');assert v==(old/p).read_text()
a=(old/'src/entry/metal_main.cpp').read_text();b=(s/'src/entry/metal_main.cpp').read_text()
a=a[a.index('class GarnetMetalAPI {'):];b=b[b.index('class GarnetMetalAPI {'):]
def method(text,name):
 m=re.search(r'    (?:std::string|X::Value|bool) '+name+r'\(',text);assert m,name;start=text.index('{',m.end());depth=1;i=start+1
 # Balance braces while removing C++ quoted strings/comments from the count.
 token=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|[{}]')
 for match in token.finditer(text,i):
  if match.group()=='{':depth+=1
  elif match.group()=='}':
   depth-=1
   if depth==0:return text[m.start():match.end()]
 raise AssertionError('unclosed '+name)
local={}
for name in ['BenchmarkPrefillPolicy','ServeModel','ServeStatus','InferJson']:
 assert method(a,name)==method(b,name+'Local').replace(name+'Local(',name+'(',1);local[name]=True
for name in ['Serving','PrimaryStatsJson','MemoryStatsJson','PrepareImage']:
 # Serving returns X::Module rather than one of the public result types.
 if name=='Serving':
  assert a[a.index('    X::Module Serving'):a.index('public:')]==b[b.index('    X::Module Serving'):b.index('public:')]
 elif re.search(r' '+name+r'\(',a):assert method(a,name)==method(b,name)
backend=method(b,'BackendInfo').replace('#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_QOS\n        info+="; native_text_worker_user_initiated_qos";\n#endif\n','').replace('#ifdef GARNET_METAL_NATIVE_TEXT_WORKER\n        info+="; native_text_worker";\n#endif\n','');assert backend==method(a,'BackendInfo')
proof=dict(source_files=len(manifest),seed_files=len(oldfiles),changed_files=changed,all_other_files_byte_exact=True,all_MSL_codecs_math_graph_session_xModel_original_tests_CUDA_unchanged=True,serving_inverse_only_existing_clock_metadata=True,original_local_boundaries_byte_exact=local,original_backend_info_inverse=True,scope='Source invariants only. New manager/transport/timing/lifecycle correctness requires actual execution; no throughput claim.')
(w/'gateway-source-inverse-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
shader=json.loads((w.parent/'worldsense-isolated-primary/independent-shader-proof.json').read_text());assert shader['CXX_materialized_OFF_primary_fragment_exact'];assert all(manifest[k]==oldfiles[k]for k in manifest if k.startswith('src/backends/metal/'))
(w/'independent-shader-proof.json').write_text(json.dumps(dict(CXX_materialized_OFF_primary_fragment_exact=True,all_shader_sources_byte_exact_WS54=True,inherited_proof_sha256=hashlib.sha256((w.parent/'worldsense-isolated-primary/independent-shader-proof.json').read_bytes()).hexdigest(),scope='Inherited verified shader expression proof plus complete frozen shader byte identity; actual constructor dumps checked by current native tests.'),indent=2)+'\n')
print('WS59 source1723/eightdelta/all original numerical/local bodies/serving inverse PASS')
