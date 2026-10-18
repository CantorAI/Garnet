"""WS59 research packaging only AFTER terminal decision and normal restoration."""
from pathlib import Path
import csv,hashlib,json,shutil,subprocess,datetime,re
r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-thread-qos');s=r/'out/research/worldsense-thread-qos/source';d=repo/'results/20261018/worldsense-m5-thread-qos';base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
assert load(w/'protocol.json')['experiment']=='WS-M5-59';assert(w/'decision.json').is_file();assert load(w/'restoration-verification.json')['benchmark_valid_request_http404'];assert(d/'README.md').read_text().startswith('# WS-M5-59 ')
allowed={'.json','.log','.py','.txt','.make','.csv','.cpp','.mm','.h','.metal','.ll','.cmake','.patch','.f32','.jpg','.png'};omitted=[]
# Experimental setup/build directories are irrelevant generated configuration;
# their root logs/commands are retained separately. Never traverse private helper.
for p in sorted(w.rglob('*')):
 if not p.is_file()or '__pycache__'in p.parts:continue
 rel=p.relative_to(w)
 if any(part.startswith('invalid-')and part.endswith('-config')for part in rel.parts):continue
 if p.suffix in allowed:copy(p,d/rel)
 elif p.suffix=='.dylib'or p.stat().st_mode&0o111:
  omitted.append(dict(local_path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Compiled local artifact; rebuild from preserved source/commands. Identity retained, not bundled.'))
for mode in ['off','primary','on','fault']:copy(r/('out/build/garnet-metal-worldsense-thread-qos-'+mode)/'primary_projection_words.bin',d/('native-'+mode+'-projection-words.bin'))
manifest=load(w/'starting-source-manifest.json');assert manifest==load(w/'gateway-source-manifest.json');assert len(manifest)==1723
raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(base+':'+rel for rel in manifest)+'\n').encode());offset=0;overlay=[]
for rel,digest in manifest.items():
 end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
 if header[-1]==b'missing':old=None
 else:
  assert header[1]==b'blob';size=int(header[2]);old=raw[offset:offset+size];offset+=size+1
 assert sha(s/rel)==digest,rel
 if old!=(s/rel).read_bytes():copy(s/rel,d/'source'/rel);overlay.append(rel)
assert offset==len(raw)
(d/'source-reconstruction.json').write_text(json.dumps(dict(base_commit=base,overlay_files=overlay,source_manifest='starting-source-manifest.json',full_closure_files=len(manifest),earlier_revision_scope='Preserved named source-overlay/manifests under bootstrap/reviewed/failed subdirectories are historical and must be reconstructed against their declared seed, not silently treated as latest.'),indent=2)+'\n')
(d/'not-bundled-local-compiled-artifacts.json').write_text(json.dumps(omitted,indent=2)+'\n')
# Export each actual WorldSense dataset once. Failed integration pair remains a
# separate diagnostic cohort, with its returned control calls ineligible for speed.
sets=[]
for stage,folder in [('resource',d),('production-speed',d/'production-speed'),('primary-speed',d/'primary-speed'),('speed',d/'speed'),('confirmation',d/'confirmation'),('functional-correctness-only',d/'functional-smoke')]:
 for p in sorted(folder.glob('*.json')):
  if not p.name.startswith(('control-','candidate-')):continue
  data=load(p)
  if isinstance(data,dict)and 'items'in data and 'search_plans'in data:sets.append((stage,p,data))
calls=[];requests=[];routes=[];samples=[];attempts=[];counts={}
for stage,p,data in sets:
 attempts.append(dict(stage=stage,artifact=str(p.relative_to(d)),worker=data['label'],valid=data['valid'],error=data.get('error',''),returned_items=len(data['items']),assignments_restored=data['assignments_restored'],http_wall_seconds=data.get('http_wall_seconds','')))
 for category,items in [('frame',data['items']),('query',data['search_plans'])]:
  for index,item in enumerate(items):
   common=dict(stage=stage,artifact=str(p.relative_to(d)),artifact_sha256=sha(p),worker=data['label'],category=category,sample_index=index,warmup=item.get('warmup',''),repetition=item.get('repetition',''))
   requests.append(dict(common,native_call_seconds=item.get('native_call_seconds',''),observation_id=item.get('observation_id',''),query=item.get('query','')))
   responses=item['result'].get('model_response',{})if category=='frame'else{'query_plan':item['result'].get('benchmark_model_response',{})}
   for phase,c in responses.items():
    if not isinstance(c,dict)or'model_id'not in c:continue
    counts[stage]=counts.get(stage,0)+1;t=c.get('timings',{});g=t.get('native_gateway',{});memory=t.get('memory_after_decode');child=g.get('child')or{};support=child.get('support',{});total,first,n=(c.get(k)for k in ['total_seconds','ttft_seconds','generated_tokens']);rate=(n-1)/(total-first)if all(isinstance(x,(int,float))for x in [n,total,first])and total>first and n>1 else'';identity=dict(common,phase=phase,model_id=c['model_id'])
    calls.append(dict(identity,status=c.get('status',''),input_tokens=t.get('input_tokens',''),generated_tokens=n,ttft_seconds=first,total_seconds=total,derived_decode_tokens_per_second=rate,input_prompt_sha256=c.get('input_prompt_sha256',''),image_sha256=item.get('image_sha256',''),skill_sha256=item.get('skill_sha256',''),eligible_for_speed=bool(data['valid']and stage not in ['resource','functional-correctness-only']and not item.get('warmup',True)),route=g.get('route','legacy-local'),parent_pid=g.get('parent_pid',''),child_pid=support.get('pid',''),child_birth=child.get('birth_id',''),child_infer_calls=child.get('infer_calls',''),raw_model_ttft_seconds=g.get('raw_model_ttft_seconds',''),raw_model_total_seconds=g.get('raw_model_total_seconds','')))
    route=dict(identity,backend=c.get('backend',''),serving_source=t.get('serving_source',''),cohort_memory_available=memory is not None,native_gateway_json=json.dumps(g,sort_keys=True))
    for key in ['primary_storage_after','primary_encoded_delta','memory_after_decode']:
     route[key+'_available']=key in t
     for k,v in t.get(key,{}).items():route[key+'_'+k]=json.dumps(v,sort_keys=True)if isinstance(v,(dict,list))else v
    routes.append(route)
    if memory is not None:
     for process in memory.get('process_samples',[]):samples.append(dict(identity,**process,is_parent=process['pid']==g['parent_pid']))
for name,rows in [('calls.csv',calls),('execution-routes.csv',routes),('skill-requests.csv',requests),('process-memory-samples.csv',samples),('worker-attempts.csv',attempts)]:
 with(d/name).open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
(d/'export-scope.json').write_text(json.dumps(dict(stage_calls=counts,returned_model_calls=len(calls),native_requests=len(requests),worker_attempts=len(attempts),scope='Every returned WorldSense frame/query model call, first/warm/failure, once. The candidate-only functional cohort is retained separately and ineligible for speed. Native correctness calls remain raw JSON, not performance samples.',decode='(generated-1)/(total-TTFT), includes host/stop/possible extraEOS/IPC/snapshot costs, not pureGPU.',memory='Snapshots, not peaks/freeRAM/residency. Candidate controlled-process sums per category may double-count shared pages. Recommended device limit is not summed. LegacyWS10 missing memory stays unavailable, never zero. Per-process sample rows exist only for candidate gateway telemetry; RAW control process identity follows its cohort manifest.',parent_wall='One native Skill request row only; do not sum parent wall across model phases.',initialization='First request preserved separately; old ready API lacks separate worker/model-load decomposition. No hardware-cold claim.',confirmation='Resource/screen exports are not independent confirmation or proof of isolated mechanism causality.'),indent=2)+'\n')
for p in sorted(Path(__file__).parent.glob('*ws59*.py')):copy(p,d/'work'/p.name)
for name in ['archive_thread_qos.py','verify_thread_qos.py']:
 p=Path(__file__).with_name(name)
 if p.exists():copy(p,d/'work'/name)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(archive=str(d),manifest_files=len(load(d/'file-manifest.json')),source_overlay=len(overlay),stage_calls=counts,requests=len(requests))))
