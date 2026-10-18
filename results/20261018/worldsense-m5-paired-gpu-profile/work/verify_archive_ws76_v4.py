"""Canonical dual-source/native/learned/raw-data audit; no model execution."""
from pathlib import Path
import ast,collections,csv,datetime,hashlib,json,math,re,struct,subprocess
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-paired-gpu-profile'
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
digest=lambda value:hashlib.sha256(value).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}
assert all(Path(k).name!='worldsense-http.py'and sha(d/k)==v for k,v in manifest.items())
def batch(refs):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode());position=0;values=[]
 for _ in refs:
  end=raw.index(b'\n',position);head=raw[position:end].split();position=end+1
  if head[-1]==b'missing':value=None
  else:assert head[1]==b'blob';size=int(head[2]);value=raw[position:position+size];position+=size+1
  values.append(value)
 assert position==len(raw);return values

export_repair=load(d/'export-repair/repair.json');initial=load(d/'export-repair/initial-file-manifest.json')
assert export_repair['initial_audit_terminal_exit_code']==1 and export_repair['no_inference_or_timing_rerun']and export_repair['only_export_paths_and_missing_recorded_outputs_changed']
assert 'actual-current-quiet.core.metal'in(d/'export-repair/first-canonical-audit-failure.log').read_text()
for path,value in initial.items():
 if path in ['work/archive_ws76.py','external-input-resolution.json','export-scope.json']:continue
 assert manifest[export_repair['renamed_reference_READMEs'].get(path,path)]==value
old_resolution=load(d/'export-repair/initial-external-input-resolution.json');new_resolution=load(d/'external-input-resolution.json')
for key,value in old_resolution.items():
 expected=dict(value);expected['bundled']=export_repair['renamed_reference_READMEs'].get(value['bundled'],value['bundled']);assert new_resolution[key]==expected
assert set(old_resolution)==set(new_resolution)
for row in export_repair['added_recorded_native_outputs']:assert sha(d/row['archive'])==row['sha256']and(d/row['archive']).stat().st_size==row['bytes']
first=(d/'work/verify_archive_ws76.py').read_text();second=(d/'work/verify_archive_ws76_v2.py').read_text()
for operation in reversed(export_repair['verifier_operations']):assert second.count(operation['after'])==1;second=second.replace(operation['after'],operation['before'])
assert second==first
initial_archive=(d/'export-repair/initial-archive_ws76.py').read_text();current_archive=(d/'work/archive_ws76.py').read_text()
for operation in reversed(export_repair['archiver_operations']):assert current_archive.count(operation['after'])==1;current_archive=current_archive.replace(operation['after'],operation['before'])
assert current_archive==initial_archive


identity_repair=load(d/'identity-repair/repair.json');initial=load(d/'identity-repair/initial-file-manifest.json')
assert identity_repair['initial_audit_terminal_exit_code']==1 and identity_repair['only_alias_resolution_to_existing_frozen_targets']and identity_repair['no_inference_or_timing_rerun']
assert all(manifest[k]==v for k,v in initial.items())
assert 'current-off/libgarnet.dylib'in(d/'identity-repair/second-canonical-audit-failure.log').read_text()
old=(d/'work/verify_archive_ws76_v2.py').read_text();new=(d/'work/verify_archive_ws76_v3.py').read_text()
for operation in reversed(identity_repair['verifier_operations']):assert new.count(operation['after'])==1;new=new.replace(operation['after'],operation['before'])
assert new==old


source_identity_repair=load(d/'source-identity-repair/repair.json');initial=load(d/'source-identity-repair/initial-file-manifest.json')
assert source_identity_repair['only_source_closure_identity_mapping']and source_identity_repair['no_original_archive_entry_changed']and source_identity_repair['no_inference_or_timing_rerun']
assert all(manifest[k]==v for k,v in initial.items())
assert 'legacy/source/runtime/metal_serving.py'in(d/'source-identity-repair/third-canonical-audit-failure.log').read_text()
old=(d/'work/verify_archive_ws76_v3.py').read_text();new=(d/'work/verify_archive_ws76_v4.py').read_text()
for operation in reversed(source_identity_repair['verifier_operations']):assert new.count(operation['after'])==1;new=new.replace(operation['after'],operation['before'])
assert new==old

sources={};reconstructions={}
for family,count in [('current',1723),('legacy',1708)]:
 rec=load(d/(family+'-source-reconstruction.json'));m=load(d/rec['source_manifest']);values=batch([rec['base_commit']+':'+rel for rel in m]);source={}
 for (rel,expected),original in zip(m.items(),values):
  assert not Path(rel).is_absolute()and '..'not in Path(rel).parts
  value=(d/rec['overlay_root']/rel).read_bytes()if rel in rec['overlay_files']else original
  assert value is not None and digest(value)==expected;source[rel]=value
 assert len(source)==rec['source_files']==count;sources[family]=source;reconstructions[family]=rec
def seed(ref,prefix,recipe):
 rec=json.loads(batch([ref+':'+prefix+recipe])[0]);m=json.loads(batch([ref+':'+prefix+rec['source_manifest']])[0]);overlays=set(rec['overlay_files'])
 values=batch([(ref+':'+prefix+rec['overlay_root']+'/'+rel)if rel in overlays else rec['base_commit']+':'+rel for rel in m]);result=dict(zip(m,values))
 assert all(value is not None and digest(value)==m[rel]for rel,value in result.items());return m,result
current_seed,current_seed_bytes=seed('e15c3f4325682098bd4c7b4a17da62254e1e36c5','results/20261018/worldsense-m5-minimal-selective-defaults/','source-reconstruction.json')
legacy_seed,legacy_seed_bytes=seed('530a4ff8cb696fb48f54e79c05b0666f4e3aff0e','results/20261018/worldsense-m5-raw-wide-vocab/','legacy-source-reconstruction.json')
expected_paths=['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/gpu_pass_counter_trace.inc','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/gpu_pass_counter_test.cpp']
for family,sm,sb in [('current',current_seed,current_seed_bytes),('legacy',legacy_seed,legacy_seed_bytes)]:
 source=sources[family];assert sm==load(d/'main'/(family+'-seed-source-manifest.json'))
 assert sorted(k for k,v in source.items()if v!=sb.get(k))==expected_paths
 operations=load(d/'main'/(family+'-source-transformations.json'))
 for rel,ops in operations.items():
  text=source[rel].decode()
  for op in reversed(ops):
   if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
   else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
  assert text.encode()==sb[rel]
 assert all(source[k]==v for k,v in sb.items()if k not in operations)
 literals=lambda text:re.findall(rb'R"([A-Za-z_]+)\((.*?)\)\1"',text,re.S)
 assert literals(source['src/backends/metal/metal_context.mm'])==literals(sb['src/backends/metal/metal_context.mm'])
raw_resolution=load(d/'external-input-resolution.json')
declared_aliases=load(d/'main/learned-build-aliases.json')
alias_prefix='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-paired-gpu-profile-current-'
expected_aliases={alias_prefix+left:alias_prefix+right for left,right in [('off','raw'),('on','trace'),('compat56','compat'),('compat56-fault','compatfault')]}
assert set(declared_aliases)==set(expected_aliases)
for name,info in declared_aliases.items():assert info['target']==info['resolved']==expected_aliases[name]
class CanonicalResolution(dict):
 def __missing__(self,key):
  # The original snapshot pins actual target libraries plus an explicit alias
  # table. An alias lookup returns that EXISTING target record without adding
  # an alias key or claiming it was separately frozen as an external input.
  for alias,info in declared_aliases.items():
   if key==str(Path(alias)/'libgarnet.dylib'):
    target=str(Path(info['target'])/'libgarnet.dylib');record=dict.__getitem__(self,target)
    assert record['sha256']==info['library_sha256'];return record
  raise KeyError(key)
resolution=CanonicalResolution(raw_resolution)
def resolved_bytes(name):
 rec=resolution[str(name)];bundled=rec['bundled'];assert bundled is not None,str(name)
 if rec['kind']=='reconstructed-source':
  kind,rel=bundled.split(':',1);value=sources[kind.removeprefix('reconstructed-')][rel]
 else:value=(d/bundled).read_bytes()
 assert digest(value)==rec['sha256']and len(value)==rec['bytes'];return value
def resolved_text(name):return resolved_bytes(name).decode()
for family in ['current','legacy']:
 helper=sources[family]['src/backends/metal/gpu_pass_counter_trace.inc'];test=sources[family]['test2026/metal/gpu_pass_counter_test.cpp']
 assert helper==resolved_bytes(b/'work/worldsense-gpu-counter-profile/gpu_pass_counter_trace.inc').replace(b'GARNET_METAL_PRIMARY_TEST_FAULTS',b'GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS')
 assert test==resolved_bytes(b/'work/worldsense-gpu-counter-profile/gpu_pass_counter_test.cpp').replace(b'GARNET_METAL_PRIMARY_TEST_FAULTS',b'GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS')
 assert b'#define PassTraceEncoder(state,name) [(state).command computeCommandEncoder]'in helper
for filename,key in [('main/native-identities-before-dispatch.json','files'),('main/compatibility-native-identities-before-dispatch.json','files'),('current-proof/learned-identities-before.json','external'),('main/capture/identities-before-dispatch.json','external')]:
 for name,value in load(d/filename)[key].items():assert resolution[name]['sha256']==value
for name,info in resolution.items():
 assert Path(name).name!='worldsense-http.py'
 if info['bundled']is not None:assert digest(resolved_bytes(name))==info['sha256']
 # Independent post-terminal audit also verifies all supplied external files.
 p=Path(name);assert p.is_file()and p.stat().st_size==info['bytes']and sha(p)==info['sha256'],name
repair=load(d/'main/native-auditor-log-format-repair.json');first=resolved_text(repair['original_frozen_verifier']);second=resolved_text(repair['revised_auditor'])
assert digest(first.encode())==repair['original_sha256']and digest(second.encode())==repair['revised_sha256']and second.replace(repair['after'],repair['before'])==first
assert repair['native_exit_code']==0 and repair['original_frozen_verifier_unchanged']and 'AssertionError'in(d/'main/native-independent-audit.log').read_text()
pre=load(d/'main/native-preparation.json')['preprocessing'];reproduced=load(d/'main/canonical-preprocessor-reproduction.json')
for family,info in pre.items():
 section=(d/'main'/reproduced[family]['section']).read_bytes();assert digest(section)==info['projection_sha256']==reproduced[family]['section_sha256']
 assert reproduced[family]['full_preprocessed_sha256_matches_original']and info['diagnostic_label_arguments_and_wrapper_absent']
 assert b'PassTraceEncoder'not in section and b'std::string("bf16:")'not in section and b'[(*impl_).command computeCommandEncoder]'in section
 assert not any('GARNET_METAL_GPU_PASS_COUNTER_PROFILE=1'in arg for arg in info['command'])
scope=load(d/'export-scope.json');roles=scope['role_info'];initial=load(d/'main/native-identities-before-dispatch.json');extra=load(d/'main/compatibility-native-identities-before-dispatch.json')
original_projection=batch(['e15c3f4325682098bd4c7b4a17da62254e1e36c5:results/20261018/worldsense-m5-minimal-selective-defaults/native-on-primary_projection_words.bin'])[0]
native_total=0;native_counter_commands=0
for role,info in roles.items():
 family=info['family'];tests=(initial if role in initial['test_inventories']else extra)['test_inventories'][role];log=(d/'main'/('native-'+role+'.log')).read_text()
 passed=re.findall(r'^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+)\s+\.{2,}\s+Passed\s+',log,re.M);assert passed==[t['name']for t in tests]and len(passed)==(3 if family=='legacy'else 10);native_total+=len(passed)
 assert(d/'native'/role/'projection-words.bin').read_bytes()==original_projection
 cache=(d/'build-identities'/role/'CMakeCache.txt').read_text();flags=dict(re.findall(r'^(GARNET_METAL(?:_[A-Z0-9_]+)?):BOOL=(ON|OFF)$',cache,re.M))
 planned=load(d/'main/build-plan.json')['commands'].get(role)
 wanted=planned['flags']if planned else load(d/'main/compatibility-proof-protocol.json')['roles'][role]['flags'];assert all(flags[k]==v for k,v in wanted.items())
 if family=='current':
  data=(d/'native'/role/'primary_core_projection_words.bin').read_bytes();position=0;words=0
  cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
  for rows,inner,columns,raw_mode in cases:
   count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
   expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+4*count]==expected;position+=4*count;words+=count
  assert position==len(data)and words==432804
  summary=load(d/'native'/role/'primary_core_projection_words.bin.summary.json');assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and not summary['core_enabled']and all(summary[k]==0 for k in ['core_decode_delta','core_prefill_delta','core_head_delta'])
  for kind in ['main','prefill','core']:
   actual=(d/'native'/role/({'main':'main_shader_program.metal','prefill':'prefill_shader_program.metal','core':'primary_core_shader_program.metal'}[kind])).read_bytes();old=batch(['e15c3f4325682098bd4c7b4a17da62254e1e36c5:results/20261018/worldsense-m5-minimal-selective-defaults/actual-'+info['original74_role']+'-'+kind+'-program.metal'])[0];assert actual==old
 else:
  assert(d/'main'/('actual-'+role+'.main.metal')).read_bytes()==batch(['e15c3f4325682098bd4c7b4a17da62254e1e36c5:results/20261018/worldsense-m5-minimal-selective-defaults/actual-off-main-program.metal'])[0]
  assert not(d/'main'/('actual-'+role+'.prefill.metal')).read_bytes()
 if role.endswith('quiet'):assert not(d/'main'/('native-counter-'+role+'.jsonl')).exists();continue
 pending=[];commands=0
 for line in(d/'main'/('native-counter-'+role+'.jsonl')).read_text().splitlines():
  kind,payload=line.split(' ',1);v=json.loads(payload);assert v['diagnostic']is True and v['serialized']is False
  if kind=='metal_counter_profile':assert v['index']==len(pending)and 0<v['start_raw']<=v['end_raw']<2**64-1;pending.append(v)
  else:
   assert kind=='metal_counter_command'and v['samples']==len(pending)==2 and v['partial_submissions']==0 and v['decode_step']in[4,16,28]
   assert all(all(row[k]==v[k]for k in ['pid','context','command','request','model','decode_step'])for row in pending);pending=[];commands+=1
 assert not pending and commands==(7 if role in ['legacy-fault','current-fault','current-compatfault']else 4);native_counter_commands+=commands
assert native_total==79 and native_counter_commands==41
# Retain every original mathematical/helper comparison through76->74 and older
# published guard transformations. Only filesystem/identity accesses are mapped.
for name,info in load(d/'current-proof/learned-guard-adaptation-proof.json').items():
 original=resolved_text(info['seed']);text=(d/'current-proof'/name).read_text();assert digest(original.encode())==info['seed_sha256']and digest(text.encode())==info['current_sha256']
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-paired-gpu-profile-current','worldsense-minimal-selective-defaults')==original
for child_ref,child_prefix,child_slug,parent_ref,parent_prefix,parent_slug in [
 ('e15c3f4325682098bd4c7b4a17da62254e1e36c5','results/20261018/worldsense-m5-minimal-selective-defaults/','worldsense-minimal-selective-defaults','4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill'),
 ('4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation'),
 ('80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation','290f5742ba51c41197f24179cfee4876e96e789a','results/20261017/worldsense-m5-request-weight-residency/','worldsense-request-weight-residency')]:
 proof=json.loads(batch([child_ref+':'+child_prefix+'learned-guard-adaptation-proof.json'])[0])
 for name,info in proof.items():
  text=batch([child_ref+':'+child_prefix+name])[0].decode();original=batch([parent_ref+':'+parent_prefix+name])[0].decode()
  assert digest(text.encode())==info['current_sha256']and digest(original.encode())==info['seed_sha256']
  for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
  assert text.replace(child_slug,parent_slug)==original
frozen=load(d/'current-proof/learned-identities-before.json')
for name,recipe in frozen['exact_corpus_build_recipes'].items():
 assert resolution[recipe['source']]['sha256']==recipe['source_sha256']and resolution[recipe['library']]['sha256']==recipe['library_sha256']and resolution[recipe['executable']]['sha256']==recipe['executable_sha256']
 assert recipe['command'][-2:]==['-o',recipe['executable']]and recipe['source']in recipe['command']and recipe['library']in recipe['command']
alias_data=load(d/'main/learned-build-aliases.json');assert alias_data==frozen['build_aliases']
for name,info in alias_data.items():assert resolution[str(Path(name)/'libgarnet.dylib')]['sha256']==info['library_sha256']and info['target']==info['resolved']
def canonical_learned_checks():
 # Materialize the already-frozen independent numerical auditor against archive
 # paths. Exact replacements do not alter any numeric/state/token comparison.
 text=(d/'work/verify_learned_ws76.py').read_text();start=text.index("terminal=load(w/'learned-controller-terminal.json')");end=text.index('result=dict(utc=',start);body=text[start:end]
 replacements=[
  ("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'oracles/matrix-selection.json'"),
  ("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'oracles/state-guard.json'"),
  ("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'oracles/prefix-model-guard.json'"),
  ("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'oracles/vision-grid-original.json'"),
  ("w.parent/'worldsense-paired-gpu-profile/compatibility-native-independent-audit.json'","d/'main/compatibility-native-independent-audit.json'"),
  ("main=w.parent/'worldsense-paired-gpu-profile';legacy=w.parent/'worldsense-paired-gpu-profile-legacy'","main=d/'main';legacy=d/'legacy-proof'"),
  ("assert len(frozen['legacy_source'])==1708 and all(sha(Path(frozen['legacy_root'])/k)==v for k,v in frozen['legacy_source'].items())","assert len(frozen['legacy_source'])==1708 and all(digest(sources['legacy'][k])==v for k,v in frozen['legacy_source'].items())"),
  ("assert all(str(Path(k).resolve())==v['resolved']and str(Path(k).readlink())==v['target']and sha(Path(k)/'libgarnet.dylib')==v['library_sha256']for k,v in frozen['build_aliases'].items())","assert frozen['build_aliases']==alias_data and all(resolution[str(Path(k)/'libgarnet.dylib')]['sha256']==v['library_sha256']for k,v in alias_data.items())"),
  ("original=Path(info['seed']).read_text();path=Path(info['current_path'])if 'current_path'in info else legacy/name;text=path.read_text()","original=resolved_text(info['seed']);text=resolved_text(info['current_path'])if 'current_path'in info else(legacy/name).read_text()"),
  ("(main/'learned-counter-command-proof.json').write_text(json.dumps(dict(proofs=trace_proofs,all_expected_selected_commands_and_complete_intervals_joined=True,all_aligned_ordered_operations_equal_after_declared_raw_primary_label_normalization=True,no_performance_claim=True),indent=2)+'\\n')","assert load(main/'learned-counter-command-proof.json')==dict(proofs=trace_proofs,all_expected_selected_commands_and_complete_intervals_joined=True,all_aligned_ordered_operations_equal_after_declared_raw_primary_label_normalization=True,no_performance_claim=True)")]
 for before,after in replacements:assert before in body,before;body=body.replace(before,after)
 namespace=dict(globals(),w=d/'current-proof');exec(compile(body,'frozen full learned auditor with explicit archive path/identity mapping','exec'),namespace)
canonical_learned_checks()

# Independently rebuild every actual request/clock/trace/CSV row from raw bytes.
cap=d/'main/capture';capture_frozen=load(cap/'identities-before-dispatch.json');cohort=load(cap/'cohort-manifest.json');terminal=load(cap/'controller-terminal.json')
native_source_groups=load(d/'main/native-identities-before-dispatch.json')['sources']
for family,key in [('current','source'),('legacy','legacy_source')]:
 expected={name:digest(value)for name,value in sources[family].items()}
 assert capture_frozen[key]==frozen[key]==native_source_groups[family]['files']==expected

assert terminal['success']and terminal['normal_restored']and not terminal['promotion']and not terminal['speed_eligibility']
assert [x['label']for x in cohort]==['legacy-1','current-1','current-2','legacy-2']and len({x['pid']for x in cohort})==4
assert all(x['client_exit']==0 and x['owned_worker_stopped']for x in cohort)
gold=load(d/'oracles/worldsense-golden.json');references={(item['observation_id'],phase):(item,call)for item in gold['items']if item['policy']=='candidate'for phase,call in item['result']['model_response'].items()if isinstance(call,dict)and call.get('status')=='ok'}
native_source=(cap/'native_route_admission.py').read_text();original=resolved_text(b/'work/worldsense-minimal-selective-defaults/production-speed/analyze.py');native_fn=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)and n.name=='native_route')
assert native_source==ast.get_source_segment(original,native_fn)+'\n';ns={};exec(compile(ast.parse(native_source),'unchanged original74 native route','exec'),ns)
inputs=load(d/'frozen-input-identities.json');assert sha(d/inputs['frame_relative'])==inputs['image_sha256']and sha(d/inputs['serialized_skill_relative'])==inputs['recorded_skill_sha256']
assert sha(d/inputs['source_yaml_relative'])==inputs['source_yaml_sha256']
skill=load(d/inputs['serialized_skill_relative']);assert skill==load(d/inputs['parsed_skill_relative'])
calls=[];parents=[];residuals=[];all_windows=[];all_families=[];records_by_worker={};sequence={};raw_files={};total_intervals=0;actual_commands=[]
def covered_duration(intervals):
 events=collections.Counter()
 for start,end in intervals:events[start]+=1;events[end]-=1
 active=0;last=None;covered=0
 for stamp,delta in sorted(events.items()):
  if last is not None and active>0:covered+=stamp-last
  active+=delta;assert active>=0;last=stamp
 assert active==0;return covered
for worker,state in zip(capture_frozen['workers'],cohort):
 label=worker['label'];family=worker['family'];pid=state['pid'];assert all(state[k]==worker[k]for k in worker)
 assert resolution[worker['library']]['sha256']==worker['library_sha256']
 assert worker['source_root']==native_source_groups[family]['root']and Path(worker['serving_source'])==Path(worker['source_root'])/'runtime/metal_serving.py'
 assert digest(sources[family]['runtime/metal_serving.py'])==worker['serving_sha256']
 data=load(cap/(label+'.json'));assert data['valid']and data['assignments_restored']and not data['search_plans']and len(data['items'])==2 and data['request']==load(cap/'cases.json')
 request_map={};expected=set();trace=cap/(label+'.counters.jsonl');raw_files[label]=sha(trace)
 for j,item in enumerate(data['items']):
  assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')and item['observation_id']==inputs['observation_id']and item['image_sha256']==inputs['image_sha256']and item['skill_sha256']==inputs['recorded_skill_sha256']and item['skill']==skill
  common=dict(worker=label,family=family,artifact='main/capture/'+label+'.json',artifact_sha256=sha(cap/(label+'.json')),repetition=j,warmup=item['warmup'],observation_id=item['observation_id'])
  parents.append(dict(common,native_call_seconds=item['native_call_seconds'],image_sha256=item['image_sha256'],skill_sha256=item['skill_sha256'],eligible_for_speed=False));models=set()
  for phase,call in item['result']['model_response'].items():
   if not isinstance(call,dict)or call.get('status')!='ok':continue
   ref_item,ref=references[(item['observation_id'],phase)];stable=lambda obj:{k:v for k,v in obj.items()if k not in ['model_response','model_response_json']}
   assert stable(item['result'])==stable(ref_item['result'])and all(call[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
   assert call['timings']['input_tokens']==ref['timings']['input_tokens']and ns['native_route'](call,family=='current')
   assert call['timings']['serving_source']==worker['serving_source']and 'gpu_pass_counter_profile'in call['backend']and 'native_gateway'not in call['timings']
   if family=='current':assert call['timings']['memory_after_decode']['task_info_status']==0
   else:assert not any(k in call['timings']for k in ['primary_storage_before','primary_storage_after','primary_encoded_delta','memory_after_decode'])
   if call['model_id']=='Qwen3.5-4B':assert call['image_preprocess_backend']=='native-cpp'
   request=call['timings']['counter_trace_request_id'];assert type(request)is int and request>0 and request not in request_map
   count=call['generated_tokens'];assert count==len(call['generated_token_ids']);limit=call['serving_configuration']['requested_output_limit'];steps=count-int(count==limit)
   assert[step for step in [4,16,28]if step<=steps]==[4,16,28]
   request_map[request]=dict(call=call,phase='first'if item['warmup']else'followup');expected.update((request,step)for step in [4,16,28]);models.add(call['model_id'])
   calls.append(dict(common,phase=phase,model=call['model_id'],request_id=request,ttft_seconds=call['ttft_seconds'],total_seconds=call['total_seconds'],generated_tokens=count,input_prompt_sha256=call['input_prompt_sha256'],eligible_for_speed=False))
   t=call['timings'];residuals.append(dict(label=label,phase='first'if item['warmup']else'followup',model=call['model_id'],generated_tokens=count,ttft_seconds=call['ttft_seconds'],total_seconds=call['total_seconds'],decode_seconds=t['decode_seconds'],postdecode_residual_seconds=(call['total_seconds']-call['ttft_seconds'])-t['decode_seconds'],decode_GPU_span_ms=t['decode_profile']['gpu_ms'],scope='Retained actual diagnostic call; arithmetic residual, not isolated CPU/telemetry cost or speed acceptance'))
  assert models=={'Qwen3.5-4B','Qwen3-1.7B'}
 assert len(request_map)==4;seen=set();pending=[];last={}
 for line in trace.read_text().splitlines():
  kind,payload=line.split(' ',1);v=json.loads(payload);assert v['pid']==pid and v['request']in request_map and v['model']==request_map[v['request']]['call']['model_id']and(v['request'],v['decode_step'])in expected
  assert v['diagnostic']is True and v['serialized']is False
  if kind=='metal_counter_profile':assert v['index']==len(pending)and 0<v['start_raw']<=v['end_raw']<2**64-1;pending.append(v);total_intervals+=1;continue
  assert kind=='metal_counter_command'and v['samples']==len(pending)=={'Qwen3.5-4B':749,'Qwen3-1.7B':537}[v['model']]and v['partial_submissions']==0
  assert math.isfinite(v['gpu_span_ms'])and v['gpu_span_ms']>=0 and all(all(row[k]==v[k]for k in ['pid','context','command','request','model','decode_step'])for row in pending)
  key=(v['request'],v['decode_step']);assert key not in seen and v['command']>last.get(v['context'],0);last[v['context']]=v['command'];seen.add(key)
  normalized=[row['op'][:-8]+':raw'if row['op'].endswith(':primary')else row['op']for row in pending];skey=(v['model'],v['decode_step'])
  if skey in sequence:assert sequence[skey]==normalized
  else:sequence[skey]=normalized
  actual_commands.append(dict(label=label,**v,raw_trace_sha256=raw_files[label],ordered_normalized_ops_sha256=digest(json.dumps(normalized,separators=(',',':')).encode())))
  base=dict(label=label,phase=request_map[v['request']]['phase'],model=v['model'],step=v['decode_step'],pid=pid,request=v['request'],context=v['context'],command=v['command'],raw_trace_sha256=raw_files[label])
  intervals=[(row['start_raw'],row['end_raw'])for row in pending];summed=sum(z-a for a,z in intervals);covered=covered_duration(intervals);span=max(z for a,z in intervals)-min(a for a,z in intervals)
  assert summed>covered and span>=covered
  all_windows.append(dict(base,passes=len(pending),duration_sum_raw=summed,merged_union_raw=covered,overlap_raw=summed-covered,sampled_span_raw=span,gaps_raw=span-covered,original_command_gpu_span_ms=v['gpu_span_ms']))
  grouped=collections.defaultdict(list)
  for row,name in zip(pending,normalized):grouped[name].append(row)
  for name,rows in sorted(grouped.items()):
   pieces=[(row['start_raw'],row['end_raw'])for row in rows];value=sum(z-a for a,z in pieces)
   all_families.append(dict(base,normalized_family=name,actual_family=rows[0]['op'],passes=len(rows),duration_sum_raw=value,merged_union_raw=covered_duration(pieces),fraction_of_sampled_interval_duration_sum=value/summed if summed else None))
  pending=[]
 assert not pending and seen==expected and len(seen)==12;records_by_worker[label]=request_map
assert len(calls)==16 and len(parents)==8 and len(all_windows)==48 and total_intervals==30864
def compare_csv(path,rows):
 actual=list(csv.DictReader(path.open()));expected=[{k:(''if value is None else str(value))for k,value in row.items()}for row in rows];assert actual==expected,path
compare_csv(d/'calls.csv',calls);compare_csv(d/'skill-requests.csv',parents)
compare_csv(cap/'command-windows.csv',all_windows);compare_csv(cap/'operation-family-windows.csv',all_families)
comparisons=[]
for legacy,current,pair in [('legacy-1','current-1','first-order'),('legacy-2','current-2','mirrored-order')]:
 for phase in ['first','followup']:
  for model in ['Qwen3.5-4B','Qwen3-1.7B']:
   for step in [4,16,28]:
    left=next(row for row in all_windows if(row['label'],row['phase'],row['model'],row['step'])==(legacy,phase,model,step));right=next(row for row in all_windows if(row['label'],row['phase'],row['model'],row['step'])==(current,phase,model,step))
    comparisons.append(dict(pair=pair,phase=phase,model=model,step=step,legacy=legacy,current=current,legacy_original_command_gpu_span_ms=left['original_command_gpu_span_ms'],current_original_command_gpu_span_ms=right['original_command_gpu_span_ms'],original_command_span_ratio=right['original_command_gpu_span_ms']/left['original_command_gpu_span_ms']if left['original_command_gpu_span_ms']else None,legacy_duration_sum_raw=left['duration_sum_raw'],current_duration_sum_raw=right['duration_sum_raw'],raw_duration_sum_ratio=right['duration_sum_raw']/left['duration_sum_raw']if left['duration_sum_raw']else None,legacy_union_raw=left['merged_union_raw'],current_union_raw=right['merged_union_raw']))
compare_csv(cap/'paired-command-comparisons.csv',comparisons)
ranks=[]
for label in ['legacy-1','current-1','current-2','legacy-2']:
 for phase in ['first','followup']:
  for model in ['Qwen3.5-4B','Qwen3-1.7B']:
   rows=[row for row in all_families if(row['label'],row['phase'],row['model'])==(label,phase,model)];values=collections.Counter();counts=collections.Counter()
   for row in rows:values[row['normalized_family']]+=row['duration_sum_raw'];counts[row['normalized_family']]+=row['passes']
   total=sum(values.values())
   for rank,(name,value)in enumerate(sorted(values.items(),key=lambda kv:kv[1],reverse=True),1):ranks.append(dict(label=label,phase=phase,model=model,rank=rank,normalized_family=name,passes=counts[name],duration_sum_raw=value,fraction_of_three_selected_steps_interval_duration_sum=value/total if total else None))
compare_csv(cap/'ranked-families.csv',ranks)
analysis=load(cap/'analysis.json');assert analysis['top_families']==[row for row in ranks if row['rank']<=5]and analysis['raw_files']==raw_files and analysis['windows_with_overlap']==48
assert analysis['selected_commands']==48 and analysis['compute_intervals']==30864 and analysis['all_first_followup_windows_and_both_worker_orders_retained']and analysis['no_pooled_acceptance_or_performance_promotion']
assert load(cap/'posthoc-request-phase-diagnostics.json')['rows']==residuals
proof=load(cap/'verification.json');assert proof['commands']==actual_commands and proof['counter_commands']==48 and proof['counter_intervals']==30864 and not proof['speed_eligibility']and not proof['promotion']
assert load(d/'main/normal-after-capture.json')['valid_benchmark_post_http404']and load(d/'main/normal-after-capture.json')['four_file_sources_online']and load(d/'main/normal-after-capture.json')['no_camera']
assert scope['model_calls']==16 and scope['Skill_requests']==8 and scope['selected_commands']==48 and scope['custom_compute_intervals']==30864 and scope['all_timings_ineligible_for_speed']and not scope['promotion']and scope['full_goal_active']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_reconstructions={family:dict(files=len(sources[family]),overlays=len(reconstructions[family]['overlay_files']))for family in sources},external_input_resolutions=len(resolution),all_supplied_external_files_independently_rehashed=True,native_CTest_pass=79,ten_roles_complete_original186_FP32_streams_exact=True,seven_current_roles_original20records432804words_recomputed=True,actual_programs_and_both_off_preprocessor_materializations_verified=True,source7path_and_guard76_to74_to71_to70_to56_inverse_exact=True,original_five_full_learned_GPU_readers_EACH4200857600words_and_all_states_tokens_vision_lifecycle_pass=True,legacy_three_API_modes_and_both_real_counter_fault_recovery_pass=True,full_learned_checks=25,actual_model_calls=16,Skill_requests=8,selected_commands=48,compute_intervals=30864,original_frame_serializedSkill_and_complete_goldens_native_routes_scope_PIDs_recomputed=True,all_raw_CSV_unions_via_independent_event_sweep_overlap_shapes_ranks_bothorders_and_posthoc_residuals_recomputed=True,all48windows_overlap=True,earlier_native_auditor_parser_only_repair_preserved=True,decision='diagnostic-complete',promotion=False,full_goal_active=True)
(Path(__file__).parent/'ws76-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
