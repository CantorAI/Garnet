from pathlib import Path
import ast,json
w=Path(__file__).resolve().parent;seed=w.parent/'worldsense-vector-prefill'
files=['freeze.py','prepare_cohort.py','run_resource.py','analyze_resource.py','audit_gates.py','dependency_closure.py','verify_correctness_terminal.py','speed/prepare.py','speed/run_stage.py','speed/analyze.py','primary-speed/prepare.py','primary-speed/run_stage.py','primary-speed/analyze.py','production-speed/prepare.py','production-speed/run_stage.py','production-speed/analyze.py','confirmation/prepare.py','confirmation/run_stage.py','confirmation/analyze.py']
for rel in files:
 p=w/rel;assert not p.exists(),rel;p.parent.mkdir(parents=True,exist_ok=True)
 t=(seed/rel).read_text().replace('worldsense-vector-prefill','worldsense-isolated-vector-prefill')
 if rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
  a=" if ('primary_vector_prefill'in call['backend'])!=enabled:return False";assert t.count(a)==1
  t=t.replace(a,a+"\n if ('primary_prefill_isolated_library'in call['backend'])!=enabled:return False",1)
  t=t.replace("'primary_u32_metadata','primary_vector_prefill','lazy_resident_defaults'","'primary_u32_metadata','primary_vector_prefill','primary_prefill_isolated_library','lazy_resident_defaults'")
 if rel=='freeze.py':
  a="'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON')";assert t.count(a)==2;t=t.replace(a,a+",'GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY':('OFF','ON')")
  t=t.replace('only primary and vector-prefill flags differ in raw/candidate; prior-primary differs from candidate only by vector-prefill','raw/candidate differ in primary, vector-prefill and isolated-library flags; prior-primary differs by vector and isolation')
 if rel=='primary-speed/prepare.py':
  a="{'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON')}";assert t.count(a)==1;t=t.replace(a,"{'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON'),'GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY':('OFF','ON')}",1)
  t=t.replace('Only PRIMARY_VECTOR_PREFILL differs; both full-primary and eager','Vector-prefill and isolated-library flags differ; both full-primary and eager')
 if rel=='verify_correctness_terminal.py':
  t=t.replace('controller53303','controller70993').replace("'controller_session':53303","'controller_session':70993")
  a="  assert ('primary_vector_prefill'in b['call']['backend'])==(mode=='on')";assert t.count(a)==1;t=t.replace(a,a+"\n  assert ('primary_prefill_isolated_library'in b['call']['backend'])==(mode=='on')",1)
  a="(w/'correctness-terminal.json').write_text";assert t.count(a)==1
  t=t.replace(a,"program=json.loads((w/'actual-program-proof.json').read_text());assert program['actual_legacy_main_priorprimary_candidate_fault_byte_exact']and program['actual_candidate_prefill_equals_main_plus_fragment']\n"+a,1)
 if rel=='audit_gates.py':
  a="   for field in ['lazy_default_retired_bytes'";assert t.count(a)==1
  t=t.replace(a,"   bad=copy.deepcopy(c);bad['backend']=bad['backend'].replace('primary_prefill_isolated_library','')if enabled else bad['backend']+' primary_prefill_isolated_library';assert not ns['native_route'](bad,enabled,prior)\n"+a,1)
  # Original quantitative/golden/source comparison remains against immutable
  # u32 guard templates, while our new native predicate is excluded there.
 ast.parse(t);p.write_text(t)
(w/'benchmark-preparation-scope.json').write_text(json.dumps({'no_worker_dispatched':True,'raw_candidate_policy_differences':3,'matched_primary_policy_differences':2,'original_quantitative_golden_source_AST_to_be_audited':True,'same_predeclared5percentVLM_TTFT_and2percent_preservation_gates':True,'actual_main_program_invariant_required':True,'pinned_external_profiles_in_dependencies':True},indent=2)+'\n')
print('WS48 resource/matched/raw/actual/full-confirmation prospective drivers prepared; declared2-flag matched contrast, no benchmark dispatched.')
