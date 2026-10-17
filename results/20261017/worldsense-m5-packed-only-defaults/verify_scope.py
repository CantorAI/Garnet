"""Exact source inverse of the packed-graph-only retirement admission guard."""
from pathlib import Path
import json,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-only-defaults/source';seed=r/'out/research/worldsense-isolated-primary/source';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((w/'ws54-seed-source-manifest.json').read_text());b=json.loads((w/'correctness-source-manifest.json').read_text());assert len(a)==len(b)==1718;assert all(sha(seed/k)==h for k,h in a.items())and all(sha(s/k)==h for k,h in b.items())
assert {k for k,h in b.items()if a.get(k)!=h}=={'cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/entry/metal_main.cpp'}
p='src/backends/metal/compiled_graph.cpp';text=(s/p).read_text();start=text.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS\n');end=text.index('#endif\n',start)+len('#endif\n');addition=text[start:end]
assert 'slot.bf16.PrimaryReady()'in addition and 'lazyDefaultSlots_.clear();lazyDefaultPlanReady_=true;return;'in addition
assert 'std::none_of(slots_.begin(),slots_.end(),'in addition
assert text[:start]+text[end:]==(seed/p).read_text()
anchor='if(!ready_||outputParts_.size()!=3||inputs_.size()<3||residentOps_.size()!=ops_.size())return;'
assert text.index(anchor)<start<text.index('std::vector<MetalBufferInfo> infos;',start)
p='src/entry/metal_main.cpp';text=(s/p).read_text();addition='#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS\n        info+="; primary_only_lazy_defaults";\n#endif\n';assert text.count(addition)==1 and text.replace(addition,'',1)==(seed/p).read_text()
p='cmake/GarnetMetal.cmake';text=(s/p).read_text();before=(seed/p).read_text();assert text.startswith(before);tail=text[len(before):];assert tail.count('option(')==1 and ' OFF)'in tail
assert 'NOT GARNET_METAL_PRIMARY_BF16_BLOCKS OR NOT GARNET_METAL_WIDE_PRIMARY_STORAGE OR NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS'in tail
for mode in ['off','primary','on','fault']:
 for part in ['main','prefill']:assert(w/f'actual-{mode}-{part}-program.metal').read_bytes()==(w.parent/'worldsense-isolated-primary'/f'actual-{mode}-{part}-program.metal').read_bytes()
result=dict(wide_admission_scope_verified=True,original_graph_code_recovered_by_only_packed_slot_admission_inverse=True,guard_after_original_validation_before_original_alias_weight_plan=True,all_math_shaders_codecs_loader_session_serving_xModel_CUDA_original_tests_exact=True,all_four_actual_programs_byte_exact_WS54=True,default_OFF_requires_primary_wide_lazy=True,actual_slots_not_model_name_or_global_primary_totals=True,source_files=1718,canonical_overlays=10,source_delta_from_WS54=3,scope='Exact source/program identity only; trained/API/policy and WorldSense evidence remain required.')
(w/'wide-scope-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
