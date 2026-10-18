"""Archive complete diagnostic source and first/warm/raw counter evidence."""
from pathlib import Path
import json,hashlib,shutil,subprocess,csv
r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-gpu-counter-profile');d=repo/'results/20261018/worldsense-m5-gpu-pass-profile';s=r/'out/research/worldsense-gpu-counter-profile/source';base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
assert load(w/'decision.json')['decision']=='diagnostic-complete'and load(w/'capture/verification.json')['four_actualWorldSense_Galaxy_native_calls_exact']
allowed={'.json','.jsonl','.log','.txt','.py','.cpp','.mm','.h','.inc','.metal','.cmake','.make','.csv','.f32','.jpg','.png'};omitted=[]
for p in sorted(w.rglob('*')):
 if not p.is_file()or '__pycache__'in p.parts:continue
 rel=p.relative_to(w)
 if p.suffix in allowed:copy(p,d/rel)
 elif p.suffix in ['.dylib','.plist']or p.stat().st_mode&0o111:omitted.append(dict(local_path=str(p),sha256=sha(p),bytes=p.stat().st_size,reason='Local compiled/lifecycle artifact: identity retained, not bundled. Binaries require rebuild/revalidation; launchd plist reconstructed from local authorized service by driver.'))
for mode in ['off','primary','on','quiet','fault']:
 b=r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode);copy(b/'primary_projection_words.bin',d/('native-'+mode+'-projection-words.bin'))
 for rel in ['CMakeCache.txt','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt','Testing/Temporary/LastTest.log']:
  if(b/rel).is_file():copy(b/rel,d/'build-identities'/mode/rel)
manifest=load(w/'correctness-source-manifest.json');assert len(manifest)==1720
raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(base+':'+rel for rel in manifest)+'\n').encode());offset=0;overlays=[]
for rel,digest in manifest.items():
 end=raw.index(b'\n',offset);head=raw[offset:end].split();offset=end+1
 if head[-1]==b'missing':original=None
 else:
  assert head[1]==b'blob';size=int(head[2]);original=raw[offset:offset+size];offset+=size+1
 assert sha(s/rel)==digest
 if original!=(s/rel).read_bytes():copy(s/rel,d/'source'/rel);overlays.append(rel)
assert offset==len(raw)
(d/'source-reconstruction.json').write_text(json.dumps(dict(base_commit=base,source_manifest='correctness-source-manifest.json',overlay_files=overlays,full_closure_files=1720,reconstruction='Write only manifest-listed files into an empty directory, from source/ overlay if listed, otherwise git base:path. Build commands/flags are recorded; no portable binary identity or speed guarantee.'),indent=2)+'\n')
previous=repo/'results/20261018/worldsense-m5-validated-text-worker'
for name in ['frozen-input-identities.json','frozen-native-guard-identities.json']:copy(previous/name,d/name)
for name in ['frozen-inputs','frozen-native-guards','oracles']:
 for p in(previous/name).rglob('*'):
  if p.is_file():copy(p,d/p.relative_to(previous))
data=load(d/'capture/candidate-1.json');calls=[];parents=[]
for item in data['items']:
 parents.append(dict(repetition=item['repetition'],warmup=item['warmup'],native_call_seconds=item['native_call_seconds'],observation_id=item['observation_id'],image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256'],eligible_for_speed=False))
 for phase,c in item['result']['model_response'].items():
  if isinstance(c,dict)and c.get('status')=='ok':calls.append(dict(phase=phase,model=c['model_id'],request_id=c['timings']['counter_trace_request_id'],repetition=item['repetition'],warmup=item['warmup'],ttft_seconds=c['ttft_seconds'],total_seconds=c['total_seconds'],generated_tokens=c['generated_tokens'],input_prompt_sha256=c['input_prompt_sha256'],eligible_for_speed=False))
assert len(calls)==4 and len(parents)==2
for name,rows in [('calls.csv',calls),('skill-requests.csv',parents)]:
 with(d/name).open('w',newline='')as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(d/'export-scope.json').write_text(json.dumps(dict(returned_pipeline_model_calls=4,native_Skill_requests=2,explicit_decode_windows=12,custom_encoder_intervals=7716,all_timings_ineligible_for_speed=True,scope='First and followup raw records retained. No optimization/memory/speed/confirmation gate or statistical selection from these diagnostics. Native tests/learned correctness are separate, not timing samples.'),indent=2)+'\n')
for p in Path(__file__).parent.glob('*ws62*.py'):copy(p,d/'work'/p.name)
for p in Path(__file__).parent.glob('ws62*.json'):
 if 'archive-independent-audit'not in p.name:copy(p,d/'work'/p.name)
(d/'not-bundled-local-artifacts.json').write_text(json.dumps(omitted,indent=2)+'\n')
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(archive=str(d),manifest_files=len(load(d/'file-manifest.json')),source_files=1720,overlays=len(overlays),calls=4,Skill_requests=2,windows=12,samples=7716)))
