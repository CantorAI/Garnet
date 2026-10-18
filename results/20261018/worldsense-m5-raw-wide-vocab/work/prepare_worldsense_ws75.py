"""Preserve five original quantitative ASTs; require actual raw-wide head admission."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-raw-wide-vocab';old=b/'work/worldsense-minimal-selective-defaults';slug='worldsense-raw-wide-vocab';seed='worldsense-minimal-selective-defaults';load=lambda p:json.loads(p.read_text())
assert load(w/'learned-independent-audit.json')['actual_loader200_head_zero_and_original_goldens_pass']and load(w/'admission-complete.json')['all_expected_config_rejections']
files=['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py'];checks={}
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in files:
 original=(old/rel).read_text();t=original.replace(seed,slug)
 before=" t0=call['timings']";after=""" t0=call['timings']
 if ('primary_raw_wide_vocab'in call['backend'])!=enabled:return False
 if enabled and 'primary_storage_after'in t0:
  for group in ['primary_storage_before','primary_storage_after']:
   if not isinstance(t0.get(group),dict):return False
   for key in ['head_encoded','vector_head_encoded']:
    if t0[group].get(key)!=0:return False""";assert t.count(before)==1;t=t.replace(before,after)
 before='  if active!=201:return False';after="  if active!=(200 if enabled else 201):return False";assert t.count(before)==1;t=t.replace(before,after)
 before="  if not all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']):return False";after="""  if enabled:
   if delta['decode_encoded']<=0 or delta['prefill_encoded']<=0 or delta['head_encoded']!=0 or delta['vector_head_encoded']!=0:return False
  elif not all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded']):return False""";assert t.count(before)==1;t=t.replace(before,after)
 before="  if active not in [0,201]or any(v!=0 for v in delta.values()):return False";after="  if active not in [0,200 if enabled else 201]or any(v!=0 for v in delta.values()):return False";assert t.count(before)==1;t=t.replace(before,after)
 before=" return s['original_bytes']==8401715200 and s['payload_bytes']+s['metadata_bytes']==6402163712 and s['metadata_bytes']==32819200 and 0<s['allocated_bytes']<8401715200";after=" wanted=(7130316800,5433064960,27852800)if enabled else(8401715200,6402163712,32819200)\n return(s['original_bytes'],s['payload_bytes']+s['metadata_bytes'],s['metadata_bytes'])==wanted and 0<s['allocated_bytes']<wanted[0]";assert t.count(before)==1;t=t.replace(before,after)
 ast.parse(t);p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(t)
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse(t.replace(slug,seed))),include_attributes=False),rel;checks[rel]=True
(w/'acceptance-gate-audit.json').write_text(json.dumps(dict(all_five_original_golden_numeric_metric_gate_AST_unchanged=True,files=checks,only_native_admission_changed=True,full_route_mutation_audit_pending=True,driver_sha256={rel:hashlib.sha256((w/rel).read_bytes()).hexdigest()for rel in files}),indent=2)+'\n')
for name in ['freeze.py','dependency_closure.py','controller.py','dispatch_prepared_resource.py']:
 t=(old/name).read_text().replace(seed,slug)
 if name=='freeze.py':
  before="['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']";after="['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB']";assert t.count(before)==1;t=t.replace(before,after)
  before="['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']";assert t.count(before)==1;t=t.replace(before,"['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_RAW_WIDE_VOCAB']")
  t=t.replace('WS74 existing minimalFULL+vector profile plus exact57 selective graph retirement, three changed files. RAW7flags and matchedPRIMARY3flags; both PRIMARY/ON selective.','WS75 semantic rawWIDEvocabulary on exact74seed,1723files/sevenpathdelta. RAW8flags and matchedPRIMARY4flags; bothPRIMARY/ONselective, onlyONrawhead. OriginalrawGPUdiagnostic proof outsideproductionshader programs.').replace('WS74 source/sevenflagRAWcontrast/threeflagPRIMARYcontrast','WS75 source/eightflagRAWcontrast/fourflagPRIMARYcontrast')
 if name=='dependency_closure.py':
  before="    assert all(p.is_absolute()and p.is_file()for p in paths)";after="""    # Supplemental actual-candidate200/rawhead proof is additional to the full
    # originalCOMPAT201 and direct proofs, not a replacement dependency closure.
    extra=['/usr/bin/c++','-std=c++17','-DGARNET_METAL_PRIMARY_RAW_WIDE_VOCAB=1','-I',str(source/'src/backends/metal'),'-I',str(source/'src/model'),'-I',str(source/'third_party'),'-M','-MT','raw_wide_corpus',str(w/'raw_wide_corpus_test.cpp')]
    extra_text=subprocess.check_output(extra,text=True);(w/'raw-wide-corpus-preprocessor-dependencies.txt').write_text(extra_text)
    paths.update(Path(name)for name in dependencies(extra_text));paths.update([w/'raw_wide_corpus_test.cpp',r/'out/build/garnet-metal-worldsense-raw-wide-vocab-on/raw_wide_corpus_test',r/'out/build/garnet-metal-worldsense-raw-wide-vocab-compat56/primary_corpus_test'])
    assert all(p.is_absolute()and p.is_file()for p in paths)
""";assert t.count(before)==1;t=t.replace(before,after).replace('actual candidate full five-reader corpus','originalCOMPAT201 plus actualcandidate200/rawhead complete proof corpora')
  before="'preprocessor_command':cmd,";assert t.count(before)==1;t=t.replace(before,"'preprocessor_command':cmd,'supplemental_preprocessor_command':extra,")
 ast.parse(t);assert not(w/name).exists();(w/name).write_text(t)
print('WS75 originalfive quantitative ASTs preserved; strictactual200/headzero/cap/selective/compat201/legacy admission and additional candidateproof dependencies prepared; no dispatch.')
