"""Keep every original numerical gate, enforce exact selective graph policy."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-selective-defaults';old=b/'work/worldsense-minimal-full-prefill';slug='worldsense-minimal-selective-defaults';seed='worldsense-minimal-full-prefill';load=lambda p:json.loads(p.read_text())
assert load(w/'learned-independent-audit.json')['selective_raw_text_retirement_zero_and_VLM_positive_all_calls']and load(w/'admission-complete.json')['all_expected_config_rejections']
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];checks={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in files:
 original=(old/rel).read_text();t=original.replace(seed,slug)
 before=" t0=call['timings']";after=""" t0=call['timings']
 if ('primary_only_lazy_defaults'in call['backend'])!=(enabled or primary_control):return False
 if any(cap in call['backend']for cap in ['primary_decode_groups2','primary_decode_groups8']):return False
 if 'primary_storage_after'in t0:
  for key in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
   if type(t0.get(key))is not int or t0[key]<0:return False""";assert t.count(before)==1;t=t.replace(before,after)
 before=" if packed:\n  if t['lazy_default_retired_bytes']<=0";after=" if packed and call['model_id']=='Qwen3.5-4B':\n  if t['lazy_default_retired_bytes']<=0";assert t.count(before)==1;t=t.replace(before,after)
 ast.parse(t);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(t)
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(t.replace(slug,seed))),include_attributes=False),rel;checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py','controller.py','dispatch_prepared_resource.py']:
 t=(old/name).read_text().replace(seed,slug)
 if name=='freeze.py':
  before="['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']";after="['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']";assert t.count(before)==1;t=t.replace(before,after)
  t=t.replace('WS71 existing minimalFULL+vector profile, ZERO1721sourcechanges, RAW6flags and matchedPRIMARY3flags.','WS74 existing minimalFULL+vector profile plus exact57 selective graph retirement, three changed files. RAW7flags and matchedPRIMARY3flags; both PRIMARY/ON selective.').replace('source/sixflagRAWcontrast','source/sevenflagRAWcontrast').replace('WS71','WS74')
 ast.parse(t);assert not(w/name).exists();(w/name).write_text(t)
print('WS74 five original metric ASTs unchanged outside native_route; strict rawtext0/VLMpositive/selectivecap/corezero; no dispatch.')
