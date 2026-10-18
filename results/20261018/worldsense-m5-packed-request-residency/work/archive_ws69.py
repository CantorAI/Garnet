"""Archive complete WS69 evidence and source reconstruction, excluding private auth/binaries."""
from pathlib import Path
import json,hashlib,shutil,subprocess,csv,re
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';d=repo/'results/20261018/worldsense-m5-packed-request-residency';load=lambda p:json.loads(p.read_text());base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
audit=load(w/'production-speed/independent-screen-audit.json');assert audit['decision']=='valid-reject'and not audit['promotion']
d.mkdir(exist_ok=False)
def copy(p,q):
 assert p.name!='worldsense-http.py','Protected auth helper is never inspected/exported'
 q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
report=(b/'outputs/Garnet-4B-WS69-Result.txt').read_text().replace('Canonical archival and publication pending.','Canonical source/data reconstruction and independent archival audit are required before publication; this report records the original performance decision.')
(d/'README.md').write_text(report);(d/'decision.json').write_text(json.dumps(dict(audit,experiment='WS-M5-69',later_stages_skipped=True,full_goal_active=True),indent=2)+'\n')
allowed={'.json','.log','.txt','.py','.cpp','.mm','.h','.inc','.metal','.cmake','.make','.csv','.f32','.jpg','.png','.gz','.tsv','.bin'};omitted=[]
for p in sorted(w.rglob('*')):
 if not p.is_file()or'__pycache__'in p.parts:continue
 assert p.name!='worldsense-http.py'
 rel=p.relative_to(w)
 if any(part.startswith('invalid-')and part.endswith('-config')for part in rel.parts):continue
 if p.suffix in allowed:copy(p,d/rel)
 elif p.suffix in ['.dylib','.plist']or p.stat().st_mode&0o111:omitted.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,reason='Local compiled artifact not bundled; recorded identity and rebuild commands require fresh validation, not binary/time reproducibility.'))
roles=['off','primary','on','fault','compat56','compat56-fault']
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role)
 for name in ['primary_projection_words.bin','packed_request_projection_words.bin','packed_request_projection_words.bin.leased.bin']:copy(build/name,d/('native-'+role+'-'+name))
 for rel in ['CMakeCache.txt','Testing/Temporary/LastTest.log','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
  if(build/rel).exists():copy(build/rel,d/'build-identities'/role/rel)
 for name in ['libgarnet.dylib','libgarnet_metal.a','primary_corpus_test','direct_float_corpus_test','packed_lease_corpus_test']:
  p=build/name
  if p.is_file():omitted.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,reason='Locally compiled library/proof executable; reconstruct from source/flags and revalidate.'))
def reconstruct(source,manifest,prefix,label):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(base+':'+rel for rel in manifest)+'\n').encode());position=0;overlays=[]
 for rel,digest in manifest.items():
  end=raw.index(b'\n',position);head=raw[position:end].split();position=end+1
  if head[-1]==b'missing':original=None
  else:
   assert head[1]==b'blob';size=int(head[2]);original=raw[position:position+size];position+=size+1
  assert sha(source/rel)==digest
  if original!=(source/rel).read_bytes():copy(source/rel,d/prefix/rel);overlays.append(rel)
 assert position==len(raw)
 target='correctness-source-manifest.json'if not label else'production-speed/control-source-before-dispatch.json'
 (d/(label+'source-reconstruction.json')).write_text(json.dumps(dict(base_commit=base,overlay_root=prefix,source_manifest=target,overlay_files=overlays,source_files=len(manifest),reconstruction='Only manifest paths in empty directory, overlay if listed otherwise immutable git base:path. Rebuild from recorded commands/flags and revalidate; not a binary/time reproducibility claim.'),indent=2)+'\n');return len(overlays)
overlays=reconstruct(r/'out/research/worldsense-packed-request-residency/source',load(w/'correctness-source-manifest.json'),'source','')
legacy_overlays=reconstruct(r/'out/research/worldsense-prefix-cache/source',load(w/'production-speed/control-source-before-dispatch.json'),'legacy-source','legacy-')
for rel in ['CMakeCache.txt','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
 p=r/'out/build/garnet-metal-worldsense-prefix-cache'/rel
 if p.exists():copy(p,d/'legacy-build'/rel)
# The immutable previous archive holds the common frame/Skill and original native oracles.
prior=repo/'results/20261018/worldsense-m5-primary-simd-metadata'
for name in ['frozen-input-identities.json','frozen-native-guard-identities.json']:copy(prior/name,d/name)
for folder in ['frozen-inputs','frozen-native-guards','oracles']:
 for p in(prior/folder).rglob('*'):
  if p.is_file():copy(p,d/p.relative_to(prior))
copy(b/'work/worldsense-isolated-primary/production-speed/control-1.json',d/'oracles/legacy-native-route-control.json')
expected=load(w/'learned-identities-before.json')['external']
for original,target in [(w.parent/'worldsense-primary-blocks/matrix-selection.json','matrix-selection.json'),(w.parent/'worldsense-prefix-cache/state-guard.json','state-guard.json'),(w.parent/'worldsense-prefix-cache/prefix-model-guard.json','prefix-model-guard.json'),(w.parent/'worldsense-request-residency/vision-grid-off.json','vision-grid-original.json')]:
 assert sha(original)==expected[str(original)];copy(original,d/'frozen-native-guards'/target)
for p in sorted(Path(__file__).parent.glob('*ws69*.py')):copy(p,d/'work'/p.name)
# Recheck every explicitly consumed external input locally once, after all jobs have
# terminated. Bundle source/driver/oracle material where available; model weights,
# compiler/SDK/runtime and compiled binaries retain identities, not invented copies.
external={}
def pin(path,digest):
 assert Path(path).name!='worldsense-http.py'
 if path in external:assert external[path]==digest,path
 external[path]=digest
for filename,key in [('native-input-identities-before-dispatch.json','files'),('learned-identities-before.json','external'),('driver-identities-before-dispatch.json','files'),('outer-method-verifier-identities-before-resource.json','files')]:
 for path,digest in load(w/filename)[key].items():pin(path,digest)
for folder in ['functional-smoke','production-speed']:
 key='drivers'if folder=='functional-smoke'else'files';name='before-functional-identities.json'if folder=='functional-smoke'else'driver-identities-before-dispatch.json'
 for path,digest in load(w/folder/name)[key].items():pin(path,digest)
for path,info in load(w/'build-host-dependencies-before-dispatch.json')['files'].items():pin(path,info['sha256'])
resolutions={}
for path,digest in external.items():
 p=Path(path);assert p.is_file()and sha(p)==digest,path
 bundled=None
 if p.is_relative_to(w):
  target=d/p.relative_to(w)
  if target.is_file():assert sha(target)==digest;bundled=str(target.relative_to(d))
 elif p.is_relative_to(Path(__file__).parent):
  target=d/'work'/p.name
  if target.is_file():assert sha(target)==digest;bundled=str(target.relative_to(d))
 elif p.is_relative_to(r/'out/research/worldsense-packed-request-residency/source'):
  rel=str(p.relative_to(r/'out/research/worldsense-packed-request-residency/source'));assert load(w/'correctness-source-manifest.json')[rel]==digest;bundled='reconstructed-candidate:'+rel
 resolutions[path]=dict(sha256=digest,bytes=p.stat().st_size,bundled=bundled,scope='Locally rehashed after terminal jobs; unbundled external dependencies must be supplied and revalidated for reproduction.')
(d/'external-input-resolution.json').write_text(json.dumps(resolutions,indent=2)+'\n')
calls=[];parents=[];routes=[];memory=[];cleanup_memory=[];leases=[];attempts=[];counts={}
for stage,folder,labels in [('resource',d,['control-1','candidate-1']),('production-speed',d/'production-speed',['control-1','candidate-1']),('functional-correctness-only',d/'functional-smoke',['candidate-1'])]:
 for label in labels:
  path=folder/(label+'.json');data=load(path);cohort=next(v for v in load(folder/'cohort-manifest.json')if v['label']==label);pid=int(re.search(r'\n\s+pid = (\d+)',cohort['cantor_before']).group(1));attempts.append(dict(stage=stage,label=label,valid=data['valid'],client_exit=cohort['client_exit'],returned_items=len(data['items'])))
  for item in data['items']:
   common=dict(stage=stage,artifact=str(path.relative_to(d)),artifact_sha256=sha(path),worker=label,warmup=item['warmup'],repetition=item['repetition'],native_process_pid=pid)
   parents.append(dict(common,native_call_seconds=item['native_call_seconds'],observation_id=item['observation_id']))
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    counts[stage]=counts.get(stage,0)+1;t=c['timings'];identity=dict(common,phase=phase,model_id=c['model_id']);first=c['ttft_seconds'];total=c['total_seconds'];n=c['generated_tokens']
    calls.append(dict(identity,ttft_seconds=first,total_seconds=total,generated_tokens=n,derived_decode_tokens_per_second=(n-1)/(total-first),input_tokens=t['input_tokens'],input_prompt_sha256=c['input_prompt_sha256'],image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256'],eligible_for_speed=stage=='production-speed'and not item['warmup']))
    row=dict(identity,backend=c['backend'],serving_source=t['serving_source'],memory_available='memory_after_decode'in t)
    for group in ['primary_storage_after','primary_encoded_delta','packed_request_weight_residency']:
     for key,value in t.get(group,{}).items():row[group+'_'+key]=value
    routes.append(row)
    if'memory_after_decode'in t:memory.append(dict(identity,**t['memory_after_decode']))
    if'memory_after_packed_request_cleanup'in t:cleanup_memory.append(dict(identity,**t['memory_after_packed_request_cleanup']))
    if'packed_request_weight_residency'in t:leases.append(dict(identity,packed_request_eligible=t['packed_request_eligible'],**t['packed_request_weight_residency']))
for name,rows in [('calls.csv',calls),('skill-requests.csv',parents),('execution-routes.csv',routes),('process-memory-samples.csv',memory),('post-cleanup-memory-samples.csv',cleanup_memory),('packed-request-lifecycle.csv',leases),('worker-attempts.csv',attempts)]:
 with(d/name).open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
assert counts=={'resource':8,'production-speed':20,'functional-correctness-only':4}and len(parents)==16 and len(memory)==22 and len(cleanup_memory)==len(leases)==18
(d/'export-scope.json').write_text(json.dumps(dict(stage_calls=counts,calls=len(calls),Skill_requests=len(parents),memory_rows=len(memory),post_cleanup_rows=len(cleanup_memory),packed_lifecycle_rows=len(leases),worker_attempts=len(attempts),no_hidden_performance_retries_or_later_stages=True,first='All first/followups retained; first not hardware-cold, no init split inferred from readiness.',decode='(generated-1)/(total-TTFT), includes host/stop/postloop/possibleEOS/snapshots/cleanup; not pureGPU.',memory='Original postdecode snapshot metric retained. Separate postcleanup snapshots show release, not a replacement metric. Not peaks/free/unique/physical-residency RAM. Legacy speed memory unavailable/null.',decision='Valid rejection: both model latency and4Bderiveddecode gates fail; wholeSkillwall within2%. No later stages/promotion.'),indent=2)+'\n')
(d/'not-bundled-local-artifacts.json').write_text(json.dumps(omitted,indent=2)+'\n')
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(archive=str(d),manifest_files=len(load(d/'file-manifest.json')),source_files=1721,source_overlays=overlays,legacy_files=1706,legacy_overlays=legacy_overlays,calls=len(calls),requests=len(parents),memory_rows=len(memory),post_cleanup_rows=len(cleanup_memory))))
