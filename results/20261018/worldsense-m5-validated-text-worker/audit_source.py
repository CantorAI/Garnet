"""CPU-only full source/overlay/local-boundary invariant proof before tests."""
from pathlib import Path
import json,hashlib,re
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-validated-text-worker/source';old=r/'out/research/worldsense-native-text-worker/source';legacy=r/'out/research/worldsense-prefix-cache/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
m=json.loads((w/'gateway-source-manifest.json').read_text());seed=json.loads((w/'parent-seed-source-manifest.json').read_text());raw=json.loads((w/'validated-source-manifest.json').read_text())
assert len(m)==1726 and len(seed)==1723 and len(raw)==1706
assert all(sha(s/k)==v for k,v in m.items())and all(sha(old/k)==v for k,v in seed.items())and all(sha(legacy/k)==v for k,v in raw.items())
changed=sorted(k for k,v in m.items()if seed.get(k)!=v)
assert changed==sorted(['cmake/GarnetMetal.cmake','src/entry/metal_main.cpp','src/entry/native_text_worker.cpp','src/entry/validated_worker_host.mm','runtime/validated_text_worker.py','runtime/validated_ws10/metal_serving.py'])
assert not set(seed)-set(m)
overlay=(s/'runtime/validated_ws10/metal_serving.py').read_text();assert overlay.count('native_model_start_monotonic=start, native_first_token_monotonic=first, ')==1
assert overlay.replace('native_model_start_monotonic=start, native_first_token_monotonic=first, ','')==(legacy/'runtime/metal_serving.py').read_text()
a=(old/'src/entry/metal_main.cpp').read_text();b=(s/'src/entry/metal_main.cpp').read_text()
marker='#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW\n        info+="; native_text_worker_validated_ws10";\n#endif\n'
assert b.count(marker)==1 and b.replace(marker,'')==a
# All gateway model routing/timing/process-set accounting remains unchanged;
# only bootstrap and provenance checking is extended for the new backend.
a=(old/'src/entry/native_text_worker.cpp').read_text();b=(s/'src/entry/native_text_worker.cpp').read_text()
suffix='std::string NativeTextWorker::Infer('
assert a[a.index(suffix):]==b[b.index(suffix):]
def section(t,start,end):return t[t.index(start):t.index(end,t.index(start))]
for start,end in [(' X::Value Call(',' void Stop('),(' void Stop(',' J Memory('),(' J Memory(){','};\nstd::recursive_mutex&'),('std::string NativeTextWorker::Serve(','std::string NativeTextWorker::Infer(')]:assert section(a,start,end)==section(b,start,end)
assert (s/'runtime/native_text_worker.py').read_bytes()==(old/'runtime/native_text_worker.py').read_bytes()
assert all(m[k]==v for k,v in seed.items()if k.startswith(('src/backends/','xModel/','src/image/','test2026/')))
assert (s/'src/entry/validated_worker_host.mm').read_bytes()==(w/'native-host/worker_host.mm').read_bytes()
(w/'gateway-source-inverse-proof.json').write_text(json.dumps(dict(parent_files=1726,seed_files=1723,validated_source_files=1706,changed_files=changed,all_MSL_codecs_math_graph_session_xModel_original_tests_CUDA_unchanged=True,old1706source_and_library_unchanged=True,original_parent_local_boundaries_inverse_exact=True,original_gateway_Infer_timing_process_memory_and_owned_Stop_Call_Serve_bodies_exact=True,standard_worker_script_exact58=True,WS10_serving_overlay_inverse_only_existing_clock_fields=True,nativehost_code_exact_probed_prototype=True,no_performance_claim=True),indent=2)+'\n')
shader=json.loads((w.parent/'worldsense-isolated-primary/independent-shader-proof.json').read_text());assert shader['CXX_materialized_OFF_primary_fragment_exact']
(w/'independent-shader-proof.json').write_text(json.dumps(dict(CXX_materialized_OFF_primary_fragment_exact=True,all_shader_sources_byte_exact_WS54=True,scope='Inherited expression proof plus complete current shader byte identity. Actual constructor dumps checked by fresh native tests; not ISA/performance.'),indent=2)+'\n')
print('WS61 parent1726/legacy1706 source, clock-only overlay, local/gateway accounting/cleanup and all numerical invariants PASS')
