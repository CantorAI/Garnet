"""Predeclare and generate an actual deployed-WS10 versus canonical-ON comparison."""
from pathlib import Path
import ast,hashlib,json,shutil
w=Path(__file__).resolve().parent;p=w/'production';p.mkdir(exist_ok=True)
old=w.parent/'worldsense-primary-u32';repo=Path('/Users/shawn/CantorAI/Garnet')
control=Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source')

route='''def native_route(call,enabled):
 t=call['timings']
 if not enabled:
  return 'dense_prefix_cache'in call['backend'] and not any(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata']) and 'primary_storage_after'not in t and 'primary_encoded_delta'not in t
 s=t['primary_storage_after'];delta=t['primary_encoded_delta']
 if not all(k in call['backend']for k in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata']):return False
 if t['lazy_default_retired_bytes']!=0 or t['lazy_default_rehydrated_bytes']!=0 or t['memory_after_decode']['task_info_status']!=0:return False
 expected={197:(3440902144,2623650048,13441024),201:(8401715200,6402163712,32819200),398:(11842617344,9025813760,46260224)}
 if s['active_weights']not in expected:return False
 original,retained,meta=expected[s['active_weights']]
 return s['original_bytes']==original and s['payload_bytes']+s['metadata_bytes']==retained and s['metadata_bytes']==meta and 0<s['allocated_bytes']<original and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
'''
t=(old/'confirmation/analyze.py').read_text()
fn=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');lines=t.splitlines(True);lines[fn.lineno-1:fn.end_lineno]=[route];t=''.join(lines)
t=t.replace("source=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-u32/source');source_hashes=json.loads((w.parent/'starting-source-manifest.json').read_text())","source=Path('/Users/shawn/CantorAI/Garnet');source_hashes=json.loads((w.parent/'canonical-source-before-build.json').read_text())")
needle="assert manifest[label]['library_sha256']==environment['libraries']['on'if expected else'off']and manifest[label]['serving_sha256']==source_hashes['runtime/metal_serving.py']";assert needle in t
t=t.replace(needle,"assert manifest[label]['library_sha256']==environment['libraries']['on'if expected else'off']and manifest[label]['serving_sha256']==(source_hashes['runtime/metal_serving.py']if expected else environment['control_serving_sha256'])",1)
t=t.replace("assert Path(c['timings']['serving_source']).resolve()==source/'runtime/metal_serving.py'","assert Path(c['timings']['serving_source']).resolve()==(source if expected else Path(environment['control_source']))/'runtime/metal_serving.py'")
needle="mem={label:max(r['call']['timings']['memory_after_decode']['metal_allocated_bytes']for r in rr)for label,rr in rows.items()}";assert needle in t
t=t.replace(needle,"mem={label:(max(r['call']['timings']['memory_after_decode']['metal_allocated_bytes']for r in rr)if label.startswith('candidate')else None)for label,rr in rows.items()}")
t=t.replace("memory_reduction=1-mem[labels[1]]/mem[labels[0]];wall_gain=1-wall[labels[1]]/wall[labels[0]]","memory_reduction=None;wall_gain=1-wall[labels[1]]/wall[labels[0]]")
assert 'and memory_reduction>=.10' in t;t=t.replace('and memory_reduction>=.10','')
t=t.replace("scope='Fresh three-frame and query-plan confirmation pair. Warm metrics separate from first; memory maxima sampled after decode over all returned calls, not peaks. Native Skill wall includes complete native request; memory snapshots count Metal resources, not free RAM or a peak.'","scope='Additional canonical-source acceptance versus actual deployedWS10. Existing full prototype resource and two-pair confirmation gates remain prerequisites. Same exact/golden/budget/2%TTFTtotaldecodeSkillwall gates; oldWS10memory telemetry unavailable and represented as null, not fabricated. Candidate native memory snapshots diagnostic only. First/warm separate.'")
assert "out/research/worldsense-primary-u32/source"not in t
ast.parse(t);(p/'analyze.py').write_text(t)

# Keep exact benchmark inputs and native controller behavior; only worker library/source map changes.
shutil.copy2(old/'confirmation/cases.json',p/'cases.json')
t=(old/'confirmation/controller.py').read_text().replace('work/worldsense-primary-u32/confirmation','work/worldsense-primary-u32-integration/production')
needle="library=root/('out/build/garnet-metal-worldsense-primary-u32-'+('on'if label.startswith('candidate')else'off'))/'libgarnet.dylib'";assert needle in t
t=t.replace(needle,"library=root/('out/build/garnet-metal-worldsense-primary-u32-integration-on'if label.startswith('candidate')else'out/build/garnet-metal-worldsense-prefix-cache')/'libgarnet.dylib'",1)
needle="sha(root/'out/research/worldsense-primary-u32/source/runtime/metal_serving.py')";assert needle in t
t=t.replace(needle,"sha((root/'Garnet'if label.startswith('candidate')else root/'out/research/worldsense-prefix-cache/source')/'runtime/metal_serving.py')",1)
ast.parse(t);(p/'controller.py').write_text(t)

# Freeze/verification and normal-restoration helpers will run only after canonical correctness passes.
t=(old/'dependency_closure.py').read_text().replace('worldsense-primary-u32','worldsense-primary-u32-integration').replace("source=r/'out/research/worldsense-primary-u32-integration/source'","source=r/'Garnet'")
(w/'dependency_closure.py').write_text(t)
(w/'restore_normal.py').write_text((old/'restore_normal.py').read_text().replace('worldsense-primary-u32','worldsense-primary-u32-integration'))
(w/'production-comparison-protocol.json').write_text(json.dumps({'stage':'Before any production-comparison timing','control':'Actual immutable deployedWS10 library/source,not new instrumentedOFF build','candidate':'Actual canonicalGarnet sourceON build; portable diagnostic output path only normal-runtime change from validatedprototype','inputs':'Exact3frames/query and output/golden/budgets unchanged','order':['control-full-1','candidate-full-1','candidate-full-2','control-full-2'],'calls':48,'gates':'Bothpairs independently alloutputs/input/config/source/library/SDK valid and bothmodelsTTFTtotalderiveddecode/Skillwall within2%regression','memory_scope':'Oldproduction lacks memory API. No memory numbers fabricated; prototype fresh10%Metal/5%footprint and twofullmemory gates remain required/satisfied. Thisadditionalproductioncomparison is latency/integration acceptance,notreplacementmemoryevidence.','source_immutability':'Canonical1715filemanifest; fulloldproduction source identity/library pinned; no source mutation/build/GPUtests/archivecopies duringtiming','promoted':False},indent=2))
print('Actual-WS10 canonical comparison drivers prepared; no worker dispatched',flush=True)
