"""Package terminal WS61 evidence and both reconstructable source closures."""
from pathlib import Path
import csv, hashlib, json, shutil, subprocess, re
r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-validated-text-worker';d=repo/'results/20261018/worldsense-m5-validated-text-worker';base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
assert load(w/'decision.json')['decision']=='reject'and load(w/'restoration-verification.json')['benchmark_valid_request_http404']
assert(d/'README.md').read_text().startswith('# WS-M5-61 ')
allowed={'.json','.log','.py','.txt','.make','.csv','.cpp','.mm','.h','.metal','.ll','.cmake','.patch','.f32','.jpg','.png'};omitted=[]
for p in sorted(w.rglob('*')):
 if not p.is_file()or '__pycache__'in p.parts:continue
 rel=p.relative_to(w)
 if any(part.startswith('invalid-')and part.endswith('-config')for part in rel.parts):continue
 if p.suffix in allowed:copy(p,d/rel)
 elif p.suffix=='.dylib'or p.stat().st_mode&0o111:omitted.append(dict(local_path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Compiled artifact identity retained; source/commands included, binary not bundled. Rebuild requires revalidation, not assumed bit reproducibility.'))
for mode in ['off','primary','on','fault']:
 copy(r/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode)/'primary_projection_words.bin',d/('native-'+mode+'-projection-words.bin'))
copy(w/'legacy-projection-words.bin',d/'legacy-projection-words.bin')
def source_overlay(source,manifest,folder,record):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(base+':'+rel for rel in manifest)+'\n').encode());offset=0;overlay=[]
 for rel,digest in manifest.items():
  end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
  if header[-1]==b'missing':old=None
  else:
   assert header[1]==b'blob';size=int(header[2]);old=raw[offset:offset+size];offset+=size+1
  assert sha(source/rel)==digest,rel
  if old!=(source/rel).read_bytes():copy(source/rel,d/folder/rel);overlay.append(rel)
 assert offset==len(raw)
 (d/record).write_text(json.dumps(dict(base_commit=base,overlay_directory=folder,overlay_files=overlay,full_closure_files=len(manifest),reconstruction='Create an empty directory; write only manifest-listed files from the overlay when present, otherwise git base:path. Do not retain unlisted files from the base tree.',scope='Native frozen source, not a guarantee of byte-identical compiler output.'),indent=2)+'\n')
 return overlay
parent=load(w/'gateway-source-manifest.json');legacy=load(w/'validated-source-manifest.json');assert len(parent)==1726 and len(legacy)==1706
parent_overlay=source_overlay(r/'out/research/worldsense-validated-text-worker/source',parent,'source','source-reconstruction.json')
legacy_overlay=source_overlay(r/'out/research/worldsense-prefix-cache/source',legacy,'legacy-source','legacy-source-reconstruction.json')
oldbuild=r/'out/build/garnet-metal-worldsense-prefix-cache'
for rel in ['CMakeCache.txt','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
 if(oldbuild/rel).exists():copy(oldbuild/rel,d/'legacy-build'/rel)
# Exact already-recorded input bytes; no decoding/transformation or camera use.
previous=repo/'results/20261018/worldsense-m5-dense-prefetch'
for name in ['frozen-input-identities.json','frozen-native-guard-identities.json']:copy(previous/name,d/name)
for name in ['frozen-inputs','frozen-native-guards']:
 for p in (previous/name).rglob('*'):
  if p.is_file():copy(p,d/p.relative_to(previous))
oracles={'api-state-guard-off.json':w.parent/'worldsense-isolated-primary/api-state-guard-off.json',
         'worldsense-golden.json':repo/'results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json'}
for name,path in oracles.items():copy(path,d/'oracles'/name)
(d/'not-bundled-local-compiled-artifacts.json').write_text(json.dumps(omitted,indent=2)+'\n')
sets=[]
for stage,folder in [('resource',d),('production-speed',d/'production-speed'),('functional-correctness-only',d/'functional-smoke')]:
 for p in sorted(folder.glob('*.json')):
  if not p.name.startswith(('control-','candidate-')):continue
  data=load(p)
  if isinstance(data,dict)and 'items'in data and 'search_plans'in data:sets.append((stage,p,data))
calls=[];requests=[];routes=[];samples=[];attempts=[];counts={}
for stage,p,data in sets:
 cohort=next(x for x in load(p.parent/'cohort-manifest.json')if x['label']==data['label']);pid=int(re.search(r'\n\s+pid = (\d+)',cohort['cantor_before']).group(1))
 attempts.append(dict(stage=stage,worker=data['label'],valid=data['valid'],assignments_restored=data['assignments_restored'],returned_items=len(data['items']),client_exit=cohort['client_exit']))
 for index,item in enumerate(data['items']):
  common=dict(stage=stage,artifact=str(p.relative_to(d)),artifact_sha256=sha(p),worker=data['label'],sample_index=index,warmup=item['warmup'],repetition=item['repetition'],native_process_pid_from_cohort=pid)
  requests.append(dict(common,native_call_seconds=item['native_call_seconds'],observation_id=item['observation_id']))
  for phase,c in item['result']['model_response'].items():
   if not isinstance(c,dict)or'model_id'not in c:continue
   counts[stage]=counts.get(stage,0)+1;t=c['timings'];g=t.get('native_gateway',{});memory=t.get('memory_after_decode');child=g.get('child')or{};support=child.get('support',{});n=c['generated_tokens'];total=c['total_seconds'];first=c['ttft_seconds'];identity=dict(common,phase=phase,model_id=c['model_id'])
   calls.append(dict(identity,status=c['status'],input_tokens=t['input_tokens'],generated_tokens=n,ttft_seconds=first,total_seconds=total,derived_decode_tokens_per_second=(n-1)/(total-first),input_prompt_sha256=c['input_prompt_sha256'],image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256'],eligible_for_speed=stage=='production-speed'and not item['warmup'],route=g.get('route','legacy-local'),child_pid=support.get('pid',''),child_birth=child.get('birth_id',''),child_infer_calls=child.get('infer_calls',''),raw_model_ttft_seconds=g.get('raw_model_ttft_seconds',''),raw_model_total_seconds=g.get('raw_model_total_seconds','')))
   route=dict(identity,backend=c['backend'],serving_source=t['serving_source'],native_gateway_json=json.dumps(g,sort_keys=True),memory_available=memory is not None)
   for key in ['primary_storage_after','primary_encoded_delta','memory_after_decode']:
    for k,v in t.get(key,{}).items():route[key+'_'+k]=json.dumps(v,sort_keys=True)if isinstance(v,(dict,list))else v
   routes.append(route)
   if memory is not None:
    parts=memory.get('process_samples')or[dict(memory,pid=pid)]
    for sample in parts:samples.append(dict(identity,**sample,is_parent=sample['pid']==pid,origin='measured-native-process'if'metal_registry_id'in sample else'raw-control-metadata'))
for name,rows in [('calls.csv',calls),('skill-requests.csv',requests),('execution-routes.csv',routes),('process-memory-samples.csv',samples),('worker-attempts.csv',attempts)]:
 with(d/name).open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
assert counts=={'resource':8,'production-speed':20,'functional-correctness-only':4}and len(requests)==16
(d/'export-scope.json').write_text(json.dumps(dict(stage_calls=counts,returned_model_calls=len(calls),native_requests=len(requests),worker_attempts=len(attempts),memory_process_rows=len(samples),scope='Each first/warm/functional returned call once. Native correctness stays raw JSON, not speed samples.',decode='(generated-1)/(total-TTFT), not pure GPU throughput; includes host/stop/postloop/possible extra EOS/IPC/snapshots.',memory='Native process snapshots, not peaks/free/unique physical residency. Candidate process categories summed separately with device recommendation once; shared pages can double-count. RAW resource control PID from launchd cohort. Legacy speed memory unavailable, never zero.',first='First records retained; startup/model load not separately exposed by old readiness. No hardware-cold claim.',parent_wall='One row per Skill; do not sum parent wall across model phases.',confirmation='No confirmation or promotion; original valid text decode gate failed.'),indent=2)+'\n')
for p in sorted(Path(__file__).parent.glob('*ws61*.py')):copy(p,d/'work'/p.name)
for p in sorted(Path(__file__).parent.glob('ws61*.json')):
 if p.name!='ws61-archive-independent-audit.json':copy(p,d/'work'/p.name)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(archive=str(d),manifest_files=len(load(d/'file-manifest.json')),parent_overlay=len(parent_overlay),legacy_overlay=len(legacy_overlay),stage_calls=counts,requests=len(requests),process_memory_rows=len(samples))))
