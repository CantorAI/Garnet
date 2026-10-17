from pathlib import Path
import csv,json,hashlib,subprocess,shutil
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI/Garnet');w=root/'work/worldsense-raw-runtime-diagnostic';d=r/'results/20261017/worldsense-m5-raw-runtime-diagnostic';base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(w/'terminal.json')['controller_session']==50693 and load(w/'terminal.json')['terminal_exit_code']==0 and not load(w/'terminal.json')['promote'];assert (d/'README.md').read_text().startswith('# WS-M5-55 ')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for p in w.iterdir():
 if p.is_file()and p.suffix in ['.json','.py','.log','.txt']:copy(p,d/p.name)
recon={}
env=load(w/'environment.json')
for role,source,manifest in [('old',Path(env['old_source']),load(w/'control-source-before-dispatch.json')),('newraw',Path(env['new_source']),load(w/'starting-source-manifest.json'))]:
 raw=subprocess.check_output(['git','-C',str(r),'cat-file','--batch'],input=('\n'.join(base+':'+k for k in manifest)+'\n').encode());at=0;overlay=[]
 for rel,h in manifest.items():
  end=raw.index(b'\n',at);header=raw[at:end].split();at=end+1
  if header[-1]==b'missing':old=None
  else:size=int(header[2]);old=raw[at:at+size];at+=size+1
  assert sha(source/rel)==h
  if old!=(source/rel).read_bytes():copy(source/rel,d/'source'/role/rel);overlay.append(rel)
 assert at==len(raw);recon[role]={'base_commit':base,'overlay_files':overlay,'source_files':len(manifest),'manifest':'control-source-before-dispatch.json'if role=='old'else'starting-source-manifest.json','scope':'Reconstruct ONLY keys in this manifest from base+overlays; files outside the manifest are not part of this recorded source closure.'}
(d/'source-reconstruction.json').write_text(json.dumps(recon,indent=2)+'\n')
for role,build in [('old',Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache')),('newraw',Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-isolated-primary-off'))]:
 for rel in ['CMakeCache.txt','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
  p=build/rel
  if p.exists():copy(p,d/'build-records'/role/rel)
for p in (root/'work/worldsense-isolated-primary/build-profile').glob('*.cmake'):copy(p,d/'build-profile'/p.name)
copy(root/'work/worldsense-isolated-primary/build-profile-identity.json',d/'build-profile-identity.json')
rows=[];parents=[];routes=[]
for label in load(w/'protocol.json')['schedule']:
 p=d/(label+'.json');data=load(p)
 for idx,i in enumerate(data['items']):
  common={'artifact':p.name,'artifact_sha256':sha(p),'worker':label,'sample_index':idx,'repetition':i['repetition'],'warmup':i['warmup']};parents.append(dict(common,native_call_seconds=i['native_call_seconds'],observation_id=i['observation_id']))
  for phase,c in i['result']['model_response'].items():
   if not isinstance(c,dict)or'model_id'not in c:continue
   t=c['timings'];identity=dict(common,phase=phase,model_id=c['model_id']);rows.append(dict(identity,status=c['status'],ttft_seconds=c['ttft_seconds'],total_seconds=c['total_seconds'],generated_tokens=c['generated_tokens'],derived_decode_tokens_per_second=(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds']),decode_phase_seconds=t['decode_seconds'],post_decode_residual_seconds=c['total_seconds']-c['ttft_seconds']-t['decode_seconds'],input_tokens=t['input_tokens'],input_prompt_sha256=c['input_prompt_sha256'],image_sha256=i['image_sha256'],skill_sha256=i['skill_sha256'],eligible_for_diagnostic_warm=not i['warmup'],**{'decode_profile_'+k:v for k,v in t['decode_profile'].items()}))
   row=dict(identity,backend=c['backend'],serving_source=t['serving_source'])
   for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta','memory_after_decode']:
    row[key+'_available']=key in t;row.update({key+'_'+k:v for k,v in t.get(key,{}).items()})
   routes.append(row)
assert len(rows)==len(routes)==40 and len(parents)==20
for name,data in [('calls.csv',rows),('execution-routes.csv',routes),('skill-requests.csv',parents)]:
 with(d/name).open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in data for k in row)));writer.writeheader();writer.writerows(data)
(d/'export-scope.json').write_text(json.dumps({'calls':40,'parent_requests':20,'schedule':load(w/'protocol.json')['schedule'],'scope':'Every first/warm/failure model call once; parent Skill wall once per request. No optimized candidate or promotion. Two pairs independent, not pooled. Derived rates include host/stop/possible extraEOS; spans overlap/include gaps; memory/VM snapshots not peaks/process-specific swap attribution.'},indent=2)+'\n')
for name in ['design_ws55.py','prepare_ws55.py','drivers_ws55.py','finalize_ws55.py','archive_raw_runtime_diagnostic.py','verify_raw_runtime_diagnostic.py']:
 p=Path(__file__).parent/name
 if p.exists():copy(p,d/'work'/name)
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n');print(json.dumps({'archive':str(d),'SHA_files':len(load(d/'file-manifest.json')),'source_reconstruction':{k:{'files':v['source_files'],'overlays':len(v['overlay_files'])}for k,v in recon.items()},'calls':40,'parents':20}))
