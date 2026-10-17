from pathlib import Path
import ast
here=Path(__file__).resolve().parent
s=(here/'archive_isolated_primary.py').read_text().replace('WS54','WS57').replace('WS-M5-54','WS-M5-57').replace('worldsense-isolated-primary','worldsense-packed-only-defaults').replace('worldsense-m5-isolated-primary','worldsense-m5-packed-only-defaults').replace('prospective53 order','prospective57 order')
needle="                    routes.append(route);count+=1";assert s.count(needle)==1;s=s.replace(needle,"                    for key in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:\n                        route[key+'_available']=key in t;route[key]=t.get(key,'')\n"+needle,1)
a=s.index("for name in ['archive_isolated_primary.py'");b=s.index("(d/'file-manifest.json')",a)
s=s[:a]+'''for name in ['archive_packed_only_defaults.py','verify_packed_only_defaults.py','design_ws57.py','source_ws57.py','full_drivers_ws57.py','drivers_ws57.py','archive_drivers_ws57.py','finalize_ws57.py','posthoc_ws57.py']:
    p=Path(__file__).parent/name
    if p.exists():copy(p,d/'work'/name)
'''+s[b:];ast.parse(s);(here/'archive_packed_only_defaults.py').write_text(s)
# Preserve the independent archive/golden/metric/CSV audit, replacing only the
# new source inverse, policy-counter admission and terminal stage identities.
s=(here/'verify_isolated_primary.py').read_text().replace('WS54','WS57').replace('WS-M5-54','WS-M5-57').replace('worldsense-isolated-primary','worldsense-packed-only-defaults').replace('worldsense-m5-isolated-primary','worldsense-m5-packed-only-defaults')
s=s.replace("['full_context_recovered_by_only_library_composition_and_lookup_inverse']","['original_graph_code_recovered_by_only_packed_slot_admission_inverse']").replace("['raw_and_storage_control_programs_byte_exact_WS53']","['raw_and_storage_control_programs_byte_exact_WS54']")
a=s.index('# Reconstruct immutable53 context');b=s.index("for mode in ['on','fault']:\n assert 'Direct float upload",a)
# Reuse the immutable-seed reconstruction/replay mechanism already validated56,
# with the new3file inverse and exact57 script instead of request lifecycle code.
t=(here/'verify_request_weight_residency.py').read_text();start=t.index('# Independently reconstruct');end=t.index("assert len(load(d/'flag-dependency-proof.json'))",start);replay=t[start:end].replace('WS56','WS57').replace('worldsense-request-weight-residency','worldsense-packed-only-defaults')
s=s[:a]+replay+"assert len(load(d/'flag-dependency-proof.json'))==9 and all(x['expected_rejection']for x in load(d/'flag-dependency-proof.json').values())\n"+s[b:]
needle="                    source=env['control_source']";at=s.index(needle);s=s[:at]+"                    assert ('primary_only_lazy_defaults'in c['backend'])==packed\n"+s[at:]
s=s.replace("                            assert tt['lazy_default_retired_bytes']>0 and tt['lazy_default_planned_slots']>0", "                            assert (tt['lazy_default_retired_bytes']>0)==(c['model_id']=='Qwen3.5-4B') and (tt['lazy_default_planned_slots']>0)==(c['model_id']=='Qwen3.5-4B')")
needle="with(d/'skill-requests.csv').open()as f:rows=list(csv.DictReader(f))";at=s.index(needle)
extra='''    if name=='execution-routes.csv':
        for row in rows:
            identity=tuple(row[k]for k in ['artifact','category','sample_index','phase','model_id']);_,call,_=raw_calls[identity]
            for key in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
                assert row[key+'_available']==str(key in call['timings'])and row[key]==str(call['timings'].get(key,''))
'''
s=s[:at]+extra+s[at:]
point=s.index("for name in ['calls.csv','execution-routes.csv']:")
t=(here/'verify_request_weight_residency.py').read_text();a=t.index('# Recompute descriptive system counters');b=t.index("for name in ['calls.csv','execution-routes.csv']:",a);post=t[a:b]
post=post.replace("  if label.startswith('candidate'):assert summary['request_counters']==[c['timings']['request_weight_residency']for c in warm]\n",'')
s=s[:point]+"if(d/'production-speed/analysis.json').exists():\n"+''.join(' '+line+'\n'for line in post.splitlines())+s[point:]
a=s.index("decision=load(d/'decision.json')");s=s[:a]+'''decision=load(d/'decision.json');assert decision['stage_outcomes']==outcomes
reached=[]
for stage in load(d/'protocol.json')['stage_order']:
 keys=([stage]if stage in outcomes else sorted(k for k in outcomes if k.startswith(stage+':')))
 if not keys:break
 reached+=keys;terminal=load(d/(stage+'-terminal.json'));assert terminal['terminal_exit_code']==0 and terminal['controller_session']>0
 if not all(outcomes[k]for k in keys):break
assert set(outcomes)==set(reached),'No later stage after a valid failure'
assert decision['prototype_confirmed']==(len(expected)==116 and all(outcomes.values())and len(outcomes)==6)
assert not decision['promote'],'Canonical production acceptance is a separate required archive'
assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
print(json.dumps(dict(files=len(manifest),source_files=len(closure),source_overlay=10,benchmark_calls=len(expected),native_requests=len(parents),stage_outcomes=outcomes,exact_outputs=True,selective_retirement_verified=True,decision_consistent=True,scope='Independent source/word/output/policy/CSV/metric audit; original recorded numerical scope retained.')))
'''
ast.parse(s);(here/'verify_packed_only_defaults.py').write_text(s)
print('WS57 evidence packaging/audit scripts prepared; not executed during timing.')
