"""Canonical raw evidence and reconstructable source closures; no auth artifacts."""
from pathlib import Path
import json,hashlib,shutil,subprocess,csv,re
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-cooperative-prefill';r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';d=repo/'results/20261018/worldsense-m5-cooperative-prefill';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
assert load(w/'decision.json')['decision']=='valid-reject'
d.mkdir(exist_ok=False);copy_report=w/'result.txt'
(d/'README.md').write_bytes(copy_report.read_bytes())
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
allowed={'.json','.log','.txt','.py','.cpp','.mm','.h','.inc','.metal','.cmake','.make','.csv','.f32','.jpg','.png','.safetensors','.gz','.tsv'};omitted=[]
for p in sorted(w.rglob('*')):
 if not p.is_file()or '__pycache__'in p.parts:continue
 rel=p.relative_to(w)
 if any(part.startswith('invalid-')and part.endswith('-config')for part in rel.parts):continue
 if p.suffix in allowed:copy(p,d/rel)
 elif p.suffix in ['.dylib','.plist']or p.stat().st_mode&0o111:omitted.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,reason='Local compiled/lifecycle artifact, reconstruct and revalidate; not bundled.'))
for mode in ['off','primary','on','fault','restore','direct64','dual66']:
 build=r/('out/build/garnet-metal-worldsense-cooperative-prefill-'+mode)
 for name in ['primary_projection_words.bin','sole_projection_words.bin','sole_dual_projection_words.bin']:copy(build/name,d/('native-'+mode+'-'+name))
 for rel in ['CMakeCache.txt','Testing/Temporary/LastTest.log','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
  if(build/rel).exists():copy(build/rel,d/'build-identities'/mode/rel)
def reconstruct(source,manifest,prefix,label):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(base+':'+rel for rel in manifest)+'\n').encode());offset=0;overlays=[]
 for rel,digest in manifest.items():
  end=raw.index(b'\n',offset);head=raw[offset:end].split();offset=end+1
  if head[-1]==b'missing':original=None
  else:size=int(head[2]);original=raw[offset:offset+size];offset+=size+1
  assert sha(source/rel)==digest
  if original!=(source/rel).read_bytes():copy(source/rel,d/prefix/rel);overlays.append(rel)
 assert offset==len(raw)
 (d/(label+'source-reconstruction.json')).write_text(json.dumps(dict(base_commit=base,overlay_root=prefix,source_manifest=('correctness-source-manifest.json'if not label else'production-speed/control-source-before-dispatch.json'),overlay_files=overlays,source_files=len(manifest),reconstruction='Empty directory, only manifest paths, overlay if listed otherwise git base:path. Rebuild from recorded commands/flags, revalidate; binary/time reproducibility not assumed.'),indent=2)+'\n');return len(overlays)
overlays=reconstruct(r/'out/research/worldsense-cooperative-prefill/source',load(w/'correctness-source-manifest.json'),'source','')
legacy_overlays=reconstruct(r/'out/research/worldsense-prefix-cache/source',load(w/'production-speed/control-source-before-dispatch.json'),'legacy-source','legacy-')
oldbuild=r/'out/build/garnet-metal-worldsense-prefix-cache'
for rel in ['CMakeCache.txt','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
 if(oldbuild/rel).exists():copy(oldbuild/rel,d/'legacy-build'/rel)
prior=repo/'results/20261018/worldsense-m5-validated-text-worker'
for name in ['frozen-input-identities.json','frozen-native-guard-identities.json']:copy(prior/name,d/name)
for folder in ['frozen-inputs','frozen-native-guards','oracles']:
 for p in(prior/folder).rglob('*'):
  if p.is_file():copy(p,d/p.relative_to(prior))
selection=w.parent/'worldsense-primary-blocks/matrix-selection.json';expected=load(w/'learned-identities-before.json')['external'][str(selection)];assert sha(selection)==expected
copy(selection,d/'frozen-native-guards/matrix-selection.json')
p=d/'frozen-native-guard-identities.json';frozen=load(p);frozen['files']['frozen-native-guards/matrix-selection.json']=dict(original_path=str(selection),sha256=expected);p.write_text(json.dumps(frozen,indent=2)+'\n')
calls=[];parents=[];routes=[];memory=[];attempts=[];counts={}
for stage,folder,labels in [('resource',d,['control-1','candidate-1']),('production-speed',d/'production-speed',['control-1','candidate-1']),('functional-correctness-only',d/'functional-smoke',['candidate-1']),('primary-speed',d/'primary-speed',['control-1','candidate-1'])]:
 for label in labels:
  path=folder/(label+'.json');data=load(path);cohort=next(row for row in load(folder/'cohort-manifest.json')if row['label']==label);pid=int(re.search(r'\n\s+pid = (\d+)',cohort['cantor_before']).group(1))
  attempts.append(dict(stage=stage,label=label,valid=data['valid'],client_exit=cohort['client_exit'],returned_items=len(data['items'])))
  for item in data['items']:
   common=dict(stage=stage,artifact=str(path.relative_to(d)),artifact_sha256=sha(path),worker=label,warmup=item['warmup'],repetition=item['repetition'],native_process_pid=pid)
   parents.append(dict(common,native_call_seconds=item['native_call_seconds'],observation_id=item['observation_id']))
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    counts[stage]=counts.get(stage,0)+1;t=c['timings'];identity=dict(common,phase=phase,model_id=c['model_id']);first=c['ttft_seconds'];total=c['total_seconds'];n=c['generated_tokens']
    calls.append(dict(identity,ttft_seconds=first,total_seconds=total,generated_tokens=n,derived_decode_tokens_per_second=(n-1)/(total-first),input_tokens=t['input_tokens'],input_prompt_sha256=c['input_prompt_sha256'],image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256'],eligible_for_speed=stage in ['production-speed','primary-speed']and not item['warmup']))
    row=dict(identity,backend=c['backend'],serving_source=t['serving_source'],memory_available='memory_after_decode'in t)
    for group in ['primary_storage_after','primary_encoded_delta','sole_storage_before','sole_storage_after','sole_encoded_delta','dual_projection_before','dual_projection_after','dual_projection_delta']:
     for k,v in t.get(group,{}).items():row[group+'_'+k]=v
    routes.append(row)
    if'memory_after_decode'in t:memory.append(dict(identity,**t['memory_after_decode']))
for name,rows in [('calls.csv',calls),('skill-requests.csv',parents),('execution-routes.csv',routes),('process-memory-samples.csv',memory),('worker-attempts.csv',attempts)]:
 with(d/name).open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
assert counts=={'resource':8,'production-speed':20,'functional-correctness-only':4,'primary-speed':20}and len(parents)==26 and len(memory)==42
(d/'export-scope.json').write_text(json.dumps(dict(stage_calls=counts,calls=len(calls),Skill_requests=len(parents),memory_rows=len(memory),worker_attempts=len(attempts),no_hidden_retries_or_post_primary_stage=True,first='All first/followups retained; first is not hardware-cold. No separate init/model-load measurement from old readiness.',decode='(generated-1)/(total-TTFT), not pureGPUthroughput.',memory='Postdecode native snapshots, not peaks/free/uniqueRAM. Legacy speed memory unavailable, not zero.',decision='Valid rejection at matchedPRIMARY20: original5%VLMgain and2%per-model/wall/memory preservation fail; no promotion.'),indent=2)+'\n')
for p in sorted(Path(__file__).parent.glob('*ws67*.py')):copy(p,d/'work'/p.name)
(d/'not-bundled-local-artifacts.json').write_text(json.dumps(omitted,indent=2)+'\n')
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(archive=str(d),manifest_files=len(load(d/'file-manifest.json')),candidate_source_files=1724,candidate_overlays=overlays,legacy_source_files=1706,legacy_overlays=legacy_overlays,stage_calls=counts,requests=len(parents),memory_rows=len(memory))))
