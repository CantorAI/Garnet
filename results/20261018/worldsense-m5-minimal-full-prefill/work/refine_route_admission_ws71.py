"""Static pre-admission review: validate all profile markers even for legacy control."""
from pathlib import Path
import shutil,json,datetime,hashlib,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-minimal-full-prefill';a=w/'pre-admission-route-review';a.mkdir(exist_ok=False)
target=d/'prepare_worldsense_ws71.py';audit=d/'audit_routes_ws71.py'
for p in [target,audit]:shutil.copy2(p,a/p.name)
before=" if 'primary_core_isolated_library'in call['backend']:return False"
after=""" for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library']:
  if (cap in call['backend'])!=enabled:return False
 for cap in ['direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata']:
  if cap in call['backend']:return False
 if 'primary_core_isolated_library'in call['backend']:return False"""
t=target.read_text();assert t.count(before)==1;t=t.replace(before,after);ast.parse(t);target.write_text(t)
old="['primary_core_isolated_library','packed_request_weight_residency','request_scope_weight_residency','primary_simd_metadata']"
new="['primary_core_isolated_library','packed_request_weight_residency','request_scope_weight_residency','primary_simd_metadata','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','primary_full_isolated_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata']"
t=audit.read_text();assert t.count(old)==1;t=t.replace(old,new);ast.parse(t);audit.write_text(t)
(a/'review.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_any_native_route_audit_or_pipeline_dispatch=True,issue='Original legacy-control native_route returns before later profile marker loop. Validate required vector/FULL and excluded extras before that return, and add all eight legacy marker corruptions to existing negative cases.',changes=[dict(path=str(target),before=before,after=after),dict(path=str(audit),before=old,after=new)],all_original_metric_golden_threshold_AST_untouched=True,no_inference_source_test_model_build_or_timing_change=True),indent=2)+'\n')
print('Pre-admission static legacy-control marker review strengthened; original negatives retained plus8legacy markers. No native/model/timing execution.')
