"""Audit the original resource cohort after explicit source-path-only repair."""
from pathlib import Path
import json, hashlib, datetime
w=Path(__file__).resolve().parent.parent/'worldsense-validated-text-worker';r=Path('/Users/shawn/CantorAI')
saved=w/'failed-resource-analysis-provenance'
rels=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']
changes={}
for rel in rels:
 old_path="source/'runtime/metal_serving.py'"
 if rel=='production-speed/analyze.py':
  old_path="(source if expected else Path(environment['control_source']))/'runtime/metal_serving.py'"
  new_path="(source if expected else Path(environment['control_source']))/('runtime/validated_ws10/metal_serving.py'if expected and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')"
 else:
  cond="mode=='on'"if rel=='analyze_resource.py'else('True'if rel=='primary-speed/analyze.py'else'bool(expected)')
  new_path="source/('runtime/validated_ws10/metal_serving.py'if "+cond+" and c['model_id']=='Qwen3-1.7B'else'runtime/metal_serving.py')"
 current=(w/rel).read_text();before=(saved/rel).read_text()
 assert current.count(new_path)==(2 if rel=='confirmation/analyze.py'else 1)
 assert current.replace(new_path,old_path)==before,rel
 changes[str((w/rel).resolve())]=dict(before=hashlib.sha256(before.encode()).hexdigest(),after=hashlib.sha256(current.encode()).hexdigest(),only_exact_model_role_source_path_selector_changed=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=json.loads((saved/'driver-identities-before-dispatch.json').read_text())['files']
for path,value in frozen.items():
 if path in changes:assert value==changes[path]['before']and sha(Path(path))==changes[path]['after']
 else:assert sha(Path(path))==value,path
assert sha(w/'cohort-manifest.json')==sha(saved/'cohort-manifest.json')
assert sha(w/'environment.json')==sha(saved/'environment.json')
for key,root in [('parent_source',r/'out/research/worldsense-validated-text-worker/source'),('legacy_source',r/'out/research/worldsense-prefix-cache/source')]:
 d=json.loads((w/'learned-input-identities-before.json').read_text())[key]
 assert all(sha(root/p)==value for p,value in d.items())
for model,files in json.loads((w/'model-assets.json').read_text()).items():
 for rel,info in files.items():assert sha(r/'out/models'/model/rel)==info['sha256']
analysis=json.loads((w/'resource-analysis.json').read_text());assert analysis['passed']and len(analysis['checks'])==8
assert all(c['exact']and c['native_admission']for c in analysis['checks'])
assert analysis['reduction']['metal_allocated_bytes']>=.10 and analysis['reduction']['task_physical_footprint_bytes']>=.05
assert json.loads((w/'dependencies-after-resource.json').read_text())['all_aliases_and_file_bytes_unchanged']
assert json.loads((w/'cohort-native-identity-audit.json').read_text())['exact_two_child_calls_no_hidden_retry']
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
(w/'resource-independent-audit.json').write_text(json.dumps(dict(
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_resource_session=8059,
 terminal_exit_code=1,terminal_failure='Original single-source analyzer provenance assertion after workers completed successfully.',
 postterminal_provenance_only_repair_inverse_exact=changes,
 no_new_inference_or_timing_retry=True,all_other_original_frozen_drivers_unchanged=True,
 full_sources_models_and_recorded_cohort_unchanged=True,
 original_eight_calls_exact_native_resource_gate_pass=True,
 reduction=analysis['reduction'],speed_result=False),indent=2)+'\n')
print('Original WS61 eight resource calls PASS; source-path-only postprocessing repair independently exact; no new inference')
