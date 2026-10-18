"""Keep every original numerical gate; enforce exact owned-local admission."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-owned-storage-local';old=b/'work/worldsense-minimal-selective-defaults';slug='worldsense-owned-storage-local';seed='worldsense-minimal-selective-defaults';load=lambda p:json.loads(p.read_text())
assert load(w/'learned-independent-audit.json')['selective_raw_text_retirement_zero_and_VLM_positive_all_calls']and load(w/'admission-complete.json')['all_expected_config_rejections']
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];checks={};changes={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in files:
 original=(old/rel).read_text();t=original.replace(seed,slug)
 before=" t0=call['timings']";after=before+"\n if ('owned_storage_local'in call['backend'])!=(enabled or primary_control):return False"
 assert t.count(before)==1;t=t.replace(before,after);ast.parse(t)
 p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(t)
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(t.replace(slug,seed))),include_attributes=False),rel
 assert t.replace(after,before).replace(slug,seed)==original;checks[rel]=True;changes[rel]=dict(before=before,after=after,seed_sha256=hashlib.sha256(original.encode()).hexdigest())
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,inverse_operations=changes,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py','controller.py','dispatch_prepared_resource.py']:
 t=(old/name).read_text().replace(seed,slug)
 if name=='freeze.py':
  before="['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']";after=before[:-1]+",'GARNET_METAL_OWNED_STORAGE_LOCAL']"
  assert t.count(before)==1;t=t.replace(before,after)
  t=t.replace('WS74 existing minimalFULL+vector profile plus exact57 selective graph retirement, three changed files. RAW7flags and matchedPRIMARY3flags; both PRIMARY/ON selective.','WS77 minimalFULL/vector/selective plus externally-retained selected-buffer local; four changed or added paths. RAW8flags and matchedPRIMARY3flags; PRIMARY/ON owned/selective.').replace('WS74','WS77').replace('sevenflagRAWcontrast','eightflagRAWcontrast')
 ast.parse(t);assert not(w/name).exists();(w/name).write_text(t)
print('WS77 all five original metric ASTs unchanged outside strict new owned-local native admission; no dispatch.')
