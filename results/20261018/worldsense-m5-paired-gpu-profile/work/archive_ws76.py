"""Archive both diagnostic source closures and all original/supplemental evidence."""
from pathlib import Path
import csv,datetime,hashlib,json,shutil,subprocess
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';w=b/'work/worldsense-paired-gpu-profile';cur=b/'work/worldsense-paired-gpu-profile-current';leg=b/'work/worldsense-paired-gpu-profile-legacy';d=repo/'results/20261018/worldsense-m5-paired-gpu-profile';opt=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert load(w/'capture/verification.json')['counter_commands']==48 and load(cur/'learned-independent-audit.json')['observed_exit_code']==0
assert load(w/'normal-after-capture.json')['valid_benchmark_post_http404'] and not load(w/'capture/controller-terminal.json')['promotion']
pre=load(w/'native-preparation.json')['preprocessing'];materialized={}
for family,info in pre.items():
 expanded=subprocess.check_output(info['command']);assert hashlib.sha256(expanded).hexdigest()==info['preprocessed_source_sha256']
 text=expanded.decode();start=text.index('void MetalContext::EncodeBF16MatVec(');end=text.index('void MetalContext::Update(',start);section=text[start:end]
 assert hashlib.sha256(section.encode()).hexdigest()==info['projection_sha256']and 'PassTraceEncoder'not in section and 'std::string("bf16:")'not in section
 target=w/'canonical-preprocessor'/(family+'-quiet-projection.ii');target.parent.mkdir(exist_ok=True);target.write_text(section)
 materialized[family]=dict(original_record=info,full_preprocessed_sha256_matches_original=True,section_sha256=sha(target),section=str(target.relative_to(w)),posthoc_deterministic_CPU_only_reproduction=True,no_native_GPU_or_model_run=True)
(w/'canonical-preprocessor-reproduction.json').write_text(json.dumps(materialized,indent=2)+'\n')
assert not d.exists();d.mkdir()
allowed={'.json','.jsonl','.log','.txt','.py','.cpp','.mm','.h','.inc','.metal','.cmake','.make','.csv','.f32','.jpg','.png','.bin','.gz','.tsv','.ii'}
omitted=[]
def copy(p,q):
 assert p.name!='worldsense-http.py';q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def omit(p,reason):
 assert p.name!='worldsense-http.py';omitted.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,reason=reason))
for root,prefix in [(w,'main'),(cur,'current-proof'),(leg,'legacy-proof')]:
 for p in sorted(root.rglob('*')):
  assert p.name!='worldsense-http.py'
  if not p.is_file()or'__pycache__'in p.parts:continue
  if root==w and p.name.endswith('-projection-words.bin'):continue
  if p.suffix in allowed:copy(p,d/prefix/p.relative_to(root))
  elif p.suffix in ['.dylib','.plist']or p.stat().st_mode&0o111:omit(p,'Local compiled/lifecycle artifact retained by identity only; rebuild/revalidate or reconstruct authorized plist using recorded driver. No binary/time reproduction guarantee.')
report=(b/'outputs/Garnet-4B-WS76-Result.txt').read_text()
report=report.replace('Canonical archive, independent canonical reanalysis and publication are next.','Canonical evidence is prepared for independent source/raw-data reanalysis before publication.')
(d/'README.md').write_text('# WS-M5-76 — Paired legacy/current GPU-pass diagnostic\n\n'+report)
(d/'result.txt').write_text(report)
roles=load(w/'protocol.json')['roles']+list(load(w/'compatibility-proof-protocol.json')['roles'])
role_info={role:dict(family=role.split('-')[0],original74_role=({'current-quiet':'on','current-trace':'on','current-fault':'fault'}.get(role)or load(w/'compatibility-proof-protocol.json')['roles'].get(role,{}).get('original_role')))for role in roles}
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
 projection=w/(role+'-projection-words.bin')if role.startswith('legacy-')else build/'primary_projection_words.bin'
 copy(projection,d/'native'/role/'projection-words.bin')
 if role.startswith('current-'):
  for name in ['primary_core_projection_words.bin','primary_core_projection_words.bin.summary.json','main_shader_program.metal','prefill_shader_program.metal','primary_core_shader_program.metal']:copy(build/name,d/'native'/role/name)
 for rel in ['CMakeCache.txt','Testing/Temporary/LastTest.log','CMakeFiles/garnet_metal.dir/flags.make','CMakeFiles/garnet.dir/flags.make','CMakeFiles/garnet.dir/link.txt']:
  if(build/rel).is_file():copy(build/rel,d/'build-identities'/role/rel)
 for p in build.iterdir():
  if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','primary_corpus_test','direct_float_corpus_test'])and(p.suffix in ['.dylib','.a']or p.stat().st_mode&0o111):omit(p,'Recorded native/proof binary; source/commands/flags and identity retained, fresh rebuild/revalidation required.')
base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2'
def overlay(family):
 source=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source';manifest=load(w/(family+'-source-manifest.json'))
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(base+':'+rel for rel in manifest)+'\n').encode());position=0;changed=[]
 for rel,digest in manifest.items():
  end=raw.index(b'\n',position);header=raw[position:end].split();position=end+1
  if header[-1]==b'missing':original=None
  else:
   assert header[1]==b'blob';size=int(header[2]);original=raw[position:position+size];position+=size+1
  assert sha(source/rel)==digest
  if original!=(source/rel).read_bytes():copy(source/rel,d/(family+'-source')/rel);changed.append(rel)
 assert position==len(raw)
 record=dict(base_commit=base,overlay_root=family+'-source',source_manifest='main/'+family+'-source-manifest.json',overlay_files=changed,source_files=len(manifest),reconstruction='Write only manifest-listed paths into an empty directory from overlay if listed, otherwise immutable git base:path. Rebuild from recorded role commands/flags and revalidate; no binary/time reproducibility claim.')
 (d/(family+'-source-reconstruction.json')).write_text(json.dumps(record,indent=2)+'\n');return record
recon={family:overlay(family)for family in ['current','legacy']}
# Original complete numerical oracles, frame/Skill golden and guard ancestors.
for source,name in [(b/'work/worldsense-primary-blocks/matrix-selection.json','matrix-selection.json'),(b/'work/worldsense-prefix-cache/state-guard.json','state-guard.json'),(b/'work/worldsense-prefix-cache/prefix-model-guard.json','prefix-model-guard.json'),(b/'work/worldsense-request-residency/vision-grid-off.json','vision-grid-original.json'),(repo/'results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json','worldsense-golden.json')]:copy(source,d/'oracles'/name)
for name in ['car.jpg','traffic.jpg','bunny.jpg']:copy(b/'work/skill-expansion'/name,d/'oracles'/name)
prior=repo/'results/20261018/worldsense-m5-raw-wide-vocab'
copy(prior/'frozen-input-identities.json',d/'frozen-input-identities.json')
for p in (prior/'frozen-inputs').rglob('*'):
 if p.is_file():copy(p,d/'frozen-inputs'/p.relative_to(prior/'frozen-inputs'))
for p in sorted(opt.glob('*ws76*.py')):copy(p,d/'work'/p.name)
# Resolve EVERY consumed identity after all jobs have terminated. Protected auth
# is excluded before any stat/read; SDK/models/binaries are not silently copied.
external={}
def pin(name,digest):
 assert Path(name).name!='worldsense-http.py'
 if name in external:assert external[name]==digest,name
 external[name]=digest
for path,key in [(w/'native-identities-before-dispatch.json','files'),(w/'compatibility-native-identities-before-dispatch.json','files'),(cur/'learned-identities-before.json','external'),(w/'capture/identities-before-dispatch.json','external')]:
 for name,value in load(path)[key].items():pin(name,value)
for name,info in load(w/'capture/build-host-dependencies-before-dispatch.json')['files'].items():pin(name,info['sha256'])
for path,key in [(w/'build-plan.json','frozen_inputs'),(w/'compatibility-build-plan.json','inputs')]:
 for name,value in load(path)[key].items():pin(name,value)
for info in load(w/'native-preparation.json')['commands'].values():
 for recipe in info.values():
  cmd=recipe['command'];pin(cmd[cmd.index('-o')+1],recipe['executable_sha256'])
all_bundled={sha(p):str(p.relative_to(d))for p in sorted(d.rglob('*'))if p.is_file()}
resolutions={}
for name,digest in external.items():
 p=Path(name);assert p.name!='worldsense-http.py'and p.is_file()and sha(p)==digest,name
 bundled=all_bundled.get(digest);kind='archive-file'if bundled else'unbundled-external'
 for family in ['current','legacy']:
  root=r/('out/research/worldsense-paired-gpu-profile-'+family)/'source'
  if p.is_relative_to(root):
   rel=str(p.relative_to(root));assert load(w/(family+'-source-manifest.json'))[rel]==digest
   bundled='reconstructed-'+family+':'+rel;kind='reconstructed-source';break
 if bundled is None and p.is_relative_to(b/'work')and p.suffix in allowed:
  target=d/'frozen-external/work'/p.relative_to(b/'work');copy(p,target);bundled=str(target.relative_to(d));kind='archive-file';all_bundled[digest]=bundled
 if bundled is None and p.is_relative_to(repo)and p.suffix in {'.py','.cpp','.h','.json','.md','.txt','.metal','.inc','.cmake'}:
  target=d/'frozen-external/repository'/p.relative_to(repo)
  if target.name=='README.md':target=target.with_name('README.md.snapshot.txt')
  copy(p,target);bundled=str(target.relative_to(d));kind='archive-file';all_bundled[digest]=bundled
 resolutions[name]=dict(sha256=digest,bytes=p.stat().st_size,canonical_local_path=str(p.resolve()),bundled=bundled,kind=kind,scope='Rehashed locally after all jobs terminal. Unbundled SDK/compiler/runtime/weights/binaries must be supplied or rebuilt and revalidated; not a portable binary/time guarantee.')
(d/'external-input-resolution.json').write_text(json.dumps(resolutions,indent=2)+'\n')
calls=[];parents=[]
for worker in load(w/'capture/identities-before-dispatch.json')['workers']:
 label=worker['label'];data=load(w/'capture'/(label+'.json'))
 for item in data['items']:
  common=dict(worker=label,family=worker['family'],artifact='main/capture/'+label+'.json',artifact_sha256=sha(w/'capture'/(label+'.json')),repetition=item['repetition'],warmup=item['warmup'],observation_id=item['observation_id'])
  parents.append(dict(common,native_call_seconds=item['native_call_seconds'],image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256'],eligible_for_speed=False))
  for phase,call in item['result']['model_response'].items():
   if not isinstance(call,dict)or call.get('status')!='ok':continue
   calls.append(dict(common,phase=phase,model=call['model_id'],request_id=call['timings']['counter_trace_request_id'],ttft_seconds=call['ttft_seconds'],total_seconds=call['total_seconds'],generated_tokens=call['generated_tokens'],input_prompt_sha256=call['input_prompt_sha256'],eligible_for_speed=False))
assert len(calls)==16 and len(parents)==8
for name,rows in [('calls.csv',calls),('skill-requests.csv',parents)]:
 with(d/name).open('w',newline='')as stream:writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
scope=dict(experiment='WS-M5-76',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='diagnostic-complete',promotion=False,full_goal_active=True,model_calls=16,Skill_requests=8,selected_commands=48,custom_compute_intervals=30864,native_CTest_pass=79,full_learned_checks=25,role_info=role_info,source_reconstructions={key:dict(files=value['source_files'],overlays=len(value['overlay_files']))for key,value in recon.items()},external_input_resolutions=len(resolutions),actual_dependency_files=1920,all_timings_ineligible_for_speed=True,scope='Complete first/followup and both worker orders retained. Reconstructed traced legacy differs from deployedWS10. Recorded custom-pass labels/counts are not complete GPU operation/dispatch/ISA identity. Overlap/rawunits/MPS omissions/posthoc residual analysis explicit. No acceptance, isolated causality or global optimum claim.')
(d/'export-scope.json').write_text(json.dumps(scope,indent=2)+'\n');(d/'not-bundled-local-artifacts.json').write_text(json.dumps(omitted,indent=2)+'\n')
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(archive=str(d),manifest_files=len(load(d/'file-manifest.json')),source_reconstructions=scope['source_reconstructions'],external_input_resolutions=len(resolutions),model_calls=16,Skill_requests=8,windows=48,intervals=30864)))
