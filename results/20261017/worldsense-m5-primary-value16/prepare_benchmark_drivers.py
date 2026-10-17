"""Prepare unchanged prospective WorldSense gates; never dispatch here."""
from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-primary-u32'
for rel in ['freeze.py','prepare_cohort.py','restore_normal.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']:
    p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text((seed/rel).read_text().replace('worldsense-primary-u32','worldsense-primary-value16'))
for rel in ['analyze_resource.py','speed/analyze.py','confirmation/analyze.py']:
    p=w/rel;t=p.read_text();needle=" if ('primary_u32_metadata'in call['backend'])!=enabled:return False";assert t.count(needle)==1
    p.write_text(t.replace(needle,needle+"\n if ('primary_value16_fold'in call['backend'])!=enabled:return False",1))
p=w/'freeze.py';t=p.read_text();needle="=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON')}";assert t.count(needle)==1
t=t.replace(needle,"=={'GARNET_METAL_PRIMARY_BF16_BLOCKS':('OFF','ON'),'GARNET_METAL_PRIMARY_VALUE16_FOLD':('OFF','ON')}",1)
t=t.replace("(w/'starting-source-manifest.json').write_text", "assert manifest==json.loads((w/'correctness-source-manifest.json').read_text()),'Source changed since native correctness'\n(w/'starting-source-manifest.json').write_text",1)
t=t.replace('only default-OFF primaryBF16 storage flag differs','only default-OFF primaryBF16 and value16fold flags differ');p.write_text(t)
p=w/'audit_gates.py';t=p.read_text().replace("old=w.parent/'worldsense-primary-kmajor'","old=w.parent/'worldsense-primary-u32'").replace(".replace('worldsense-primary-value16','worldsense-primary-kmajor')",".replace('worldsense-primary-value16','worldsense-primary-u32')")
needle="   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_u32_metadata','')if mode=='on'else bad['backend']+' primary_u32_metadata';assert not namespace['native_route'](bad,mode=='on')";assert t.count(needle)==1
t=t.replace(needle,needle+"\n   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_value16_fold','')if mode=='on'else bad['backend']+' primary_value16_fold';assert not namespace['native_route'](bad,mode=='on')",1);p.write_text(t)
p=w/'prepare_cohort.py';t=p.read_text();needle="assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())";assert t.count(needle)==1
extra="""
assert json.loads((w/'compiler-IR-diagnostic.json').read_text())['performance_eligible']is True
assert json.loads((w/'materialized-shader-proof.json').read_text())['on_to_off_helper_replacement_recovers_entire_shader']
assert json.loads((w/'correctness-terminal.json').read_text())['new_compact_case_executed_on_and_fault']
for mode in ['on','fault']:
    log=r/('out/build/garnet-metal-worldsense-primary-value16-'+mode)/'Testing/Temporary/LastTest.log'
    assert 'Compact base/tag/signmant1044480words BOTH GPU readers exact PASS'in log.read_text()
"""
p.write_text(t.replace(needle,needle+extra,1))
p=w/'confirmation/prepare.py';t=p.read_text();needle="assert json.loads((p/'speed/analysis.json').read_text())['passed']";assert t.count(needle)==1
p.write_text(t.replace(needle,needle+"\nassert json.loads((p/'production-speed/analysis.json').read_text())['passed']",1))

# Use the already tested actualWS10 screen scaffold. Replace only its route
# predicate with the full-primary/value16 one, preserving all numeric logic.
d=w/'production-speed';d.mkdir()
for name in ['prepare.py','run_stage.py','analyze.py']:
    t=(w.parent/'worldsense-primary-narrow-attention/production-speed'/name).read_text().replace('worldsense-primary-narrow-attention','worldsense-primary-value16')
    (d/name).write_text(t)
text=(w/'speed/analyze.py').read_text();tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');new=ast.get_source_segment(text,fn)
needle=" t=call['timings'];s=t['primary_storage_after'];delta=t['primary_encoded_delta']";assert new.count(needle)==1
guard=" if not enabled:\n  return 'dense_prefix_cache'in call['backend']and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_value16_fold','primary_raw_narrow_vocab','primary_raw_narrow_attention'])and not any(k in call['timings']for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])\n"
new=new.replace(needle,guard+needle,1)
p=d/'analyze.py';text=p.read_text();tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');old=ast.get_source_segment(text,fn);assert text.count(old)==1;p.write_text(text.replace(old,new,1))
for p in w.rglob('*.py'):ast.parse(p.read_text(),str(p))
(w/'benchmark-preparation-scope.json').write_text(json.dumps(dict(no_worker_dispatched=True,unchanged_original8_20_48numeric_and_golden_rules=True,additional_actualWS10speed20_required=True,source_since_correctness_and_new_GPU_compact_case_required=True),indent=2))
print('Future8/20/additionalactual20/48drivers prepared; source/newcase/IR guards required before dispatch',flush=True)
