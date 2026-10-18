"""Retain WS74's complete learned proofs and add both diagnostic source routes."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');main=b/'work/worldsense-paired-gpu-profile';r=Path('/Users/shawn/CantorAI');old=b/'work/worldsense-minimal-selective-defaults'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(main/'compatibility-native-independent-audit.json')['combined_native_CTest_count']==79
current=b/'work/worldsense-paired-gpu-profile-current';legacy=b/'work/worldsense-paired-gpu-profile-legacy'
assert not current.exists() and not legacy.exists();current.mkdir();legacy.mkdir()
addendum='''WS76 learned-proof execution plan, before helper adaptation/build/learned dispatch
Preserve all16original74subprocess checks, add currentquiet API (five complete
state/token calls), three legacy API roles quiet/trace/fault (five calls each),
three legacy vision-grid/shared-close/repeat roles, and actual learned counter
resolution-failure/reset recovery in both sources. Total25subprocess checks.
Current original five201matrix GPUreaders,245rawHOST buffers,546direct buffers,
64resident private steps, trainedlinear, shared/late-session/close/rebuild,
vision, all original32VLM/28text state/token/capacity/prefix/fault/4GiB controls
remain. Legacy's states()/prompts/inputprofiles/full comparison and error
recovery are unchanged from original references; new storage/counter/lazy
admission assertions unavailable in the older API are structurally absent,
not fabricated zeros. Strict explicit trace presence/identity/scope admission
is added for both APIs. Legacy vision helper changes paths only.
Four recorded current build aliases map original proof names off/on/compat56/
compat56-fault to actual raw/trace/compat/compatfault builds. All alias targets,
canonical paths and binary hashes are explicit and checked before/after;
they do not claim a different loaded binary. Original mode names remain in
numeric guards. Source overlays unchanged. All helper changes inverse-exact
to original74or62 and every original mathematical comparison retained.
The entire source/model/runtime/fixture/oracle/helper/outer-controller and
independent-verifier closure and exact corpus buildrecipes are frozen BEFORE
learned execution. No actual WorldSense diagnostic capture or speed claim.
SelectedWS10 is restored by finally. All original optimization gates remain.
'''
(main/'learned-plan-before-helper-adaptation.txt').write_text(addendum)
(b/'outputs/Garnet-4B-WS76-Learned-Plan.txt').write_text(addendum)
aliases={}
for left,right in [('off','raw'),('on','trace'),('compat56','compat'),('compat56-fault','compatfault')]:
    alias=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-current-'+left);target=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-current-'+right)
    assert not alias.exists() and not alias.is_symlink() and target.is_dir();alias.symlink_to(target,target_is_directory=True)
    aliases[str(alias)]=dict(target=str(target),resolved=str(alias.resolve()),library_sha256=sha(target/'libgarnet.dylib'))
(main/'learned-build-aliases.json').write_text(json.dumps(aliases,indent=2)+'\n')
names=['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py','full_correctness.py']
proof={}
for name in names:
    original=(old/name).read_text();t=original.replace('worldsense-minimal-selective-defaults','worldsense-paired-gpu-profile-current');ops=[]
    def change(a,z,count=1):
        global t
        assert t.count(a)==count,(name,a,t.count(a));t=t.replace(a,z);ops.append(dict(before=a,after=z,count=count))
    if name=='api_state_guard.py':
        change("['off','primary','on','compat56'];expected=", "['off','primary','on','quiet','compat56'];expected=")
        change("vector_expected=mode in ['on','compat56']", "vector_expected=mode in ['on','quiet','compat56']")
        change("selective_expected=mode in ['primary','on']", "selective_expected=mode in ['primary','on','quiet']")
        change('import garnet,metal_serving','import garnet,metal_serving\nassert hasattr(garnet,"counter_trace_step")==hasattr(garnet,"counter_trace_request_id")== (mode!="quiet")')
        change("assert call['status']=='ok'\n", "assert call['status']=='ok'\n if mode!='quiet':\n  assert type(call['timings']['counter_trace_request_id'])is int and call['timings']['counter_trace_request_id']>0 and garnet.counter_trace_request_id()==0\n else:assert 'counter_trace_request_id'not in call['timings']\n")
    if name=='full_correctness.py':
        change('len(manifest)==1721','len(manifest)==1723')
        change("assert load(w/'native-independent-audit.json')['native_tests_pass']==54", "assert load(w.parent/'worldsense-paired-gpu-profile/compatibility-native-independent-audit.json')['combined_native_CTest_count']==79")
        change("for mode in ['off','primary','on','compat56']:run", "for mode in ['off','primary','on','quiet','compat56']:run")
        change("with(w/name).open('w')as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)", "with(w/name).open('w')as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(__import__('os').environ,GARNET_METAL_COUNTER_TRACE='1',GARNET_METAL_COUNTER_TRACE_FILE=str(w/(name+'.counters.jsonl'))))")
        change("def verify():assert len(manifest)==1723 and all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in inputs.items())", "def verify():\n assert len(manifest)==1723 and all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in inputs.items())\n assert all(str(Path(k).resolve())==v['resolved']and str(Path(k).readlink())==v['target']and sha(Path(k)/'libgarnet.dylib')==v['library_sha256']for k,v in frozen['build_aliases'].items())\n assert len(frozen['legacy_source'])==1708 and all(sha(Path(frozen['legacy_root'])/k)==v for k,v in frozen['legacy_source'].items())")
        anchor=" verify();(w/'api-correctness-result.json')"
        extra=""" for mode in ['quiet','trace','fault']:
  run([str(n/'xlang3'),'-B',str(w.parent/'worldsense-paired-gpu-profile-legacy/api_state_guard.py'),mode],'legacy-api-state-guard-'+mode+'.log')
  run([str(n/'xlang3'),'-B',str(w.parent/'worldsense-paired-gpu-profile-legacy/vision_grid_guard.py'),mode],'legacy-vision-grid-'+mode+'.log')
 for family in ['current','legacy']:run([str(n/'xlang3'),'-B',str(w.parent/('worldsense-paired-gpu-profile-'+family)/'counter_fault_model_guard.py')],family+'-counter-fault-model-recovery.log')
"""
        change(anchor,extra+anchor)
        change('source1721_and_explicit_inputs_unchanged','source1723_and_explicit_inputs_unchanged')
        change('WS74 full learned controller PASS','WS76 full learned controller PASS')
    if name.endswith('.py'):ast.parse(t)
    inverse=t
    for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
    assert inverse.replace('worldsense-paired-gpu-profile-current','worldsense-minimal-selective-defaults')==original
    (current/name).write_text(t);proof[name]=dict(seed=str(old/name),seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(t.encode()).hexdigest(),operations=ops,inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,current/name)
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib']:shutil.copy2(main/name,current/name)
shutil.copy2(main/'current-source-manifest.json',current/'correctness-source-manifest.json')
(current/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
legacy_proof={}
for name in ['api_state_guard.py','vision_grid_guard.py']:
    original=(old/name).read_text();t=original.replace('worldsense-minimal-selective-defaults','worldsense-paired-gpu-profile-legacy');ops=[]
    def edit(a,z):
        global t
        assert t.count(a)==1,(name,a[:70]);t=t.replace(a,z);ops.append(dict(before=a,after=z,count=1))
    if name=='api_state_guard.py':
        original_line=next(line for line in t.splitlines() if line.startswith('mode=sys.argv[1];'))
        edit(original_line,"mode=sys.argv[1];assert mode in ['quiet','trace','fault']")
        edit('import garnet,metal_serving','import garnet,metal_serving\nassert hasattr(garnet,"counter_trace_step")==hasattr(garnet,"counter_trace_request_id")== (mode!="quiet")')
        start=t.index('def infer(');end=t.index("load('Qwen3.5-4B'",start);old_infer=t[start:end]
        new_infer='''def infer(name,prompt,image,limit,layers):
 call=json.loads(garnet.infer_json(prompt,image,limit,name));(w/('last-native-call-'+mode+'.json')).write_text(json.dumps(call,indent=2));assert call['status']=='ok'
 assert 'primary_bf16_blocks'not in call['backend']and 'lazy_resident_defaults'not in call['backend']
 if mode!='quiet':
  assert type(call['timings']['counter_trace_request_id'])is int and call['timings']['counter_trace_request_id']>0 and garnet.counter_trace_request_id()==0
 else:assert 'counter_trace_request_id'not in call['timings']
 row=dict(model_id=name,call=call,states=states(name,layers));records.append(row);save();return row
'''
        edit(old_infer,new_infer)
        metadata='primary_expected=expected,vector_prefill_expected=vector_expected,lazy_defaults_expected=bool(expected),selective_defaults_expected=selective_expected,raw_text_retirement_expected=compat_expected,cached_metadata_expected=compat_expected,'
        edit(metadata,'legacy_original_storage_domain=True,')
        def state_ast(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='states'),include_attributes=False)
        assert state_ast(t)==state_ast(original)
    ast.parse(t);inverse=t
    for op in reversed(ops):assert inverse.count(op['after'])==1;inverse=inverse.replace(op['after'],op['before'])
    assert inverse.replace('worldsense-paired-gpu-profile-legacy','worldsense-minimal-selective-defaults')==original
    (legacy/name).write_text(t);legacy_proof[name]=dict(seed=str(old/name),seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(t.encode()).hexdigest(),operations=ops,inverse_exact=True,all_original_state_hashes_prompts_profiles_and_comparisons_retained=True)
for family,folder in [('current',current),('legacy',legacy)]:
    original=(b/'work/worldsense-gpu-counter-profile/counter_fault_model_guard.py').read_text()
    t=original.replace('worldsense-gpu-counter-profile','worldsense-paired-gpu-profile-'+family)
    ast.parse(t);assert t.replace('worldsense-paired-gpu-profile-'+family,'worldsense-gpu-counter-profile')==original
    (folder/'counter_fault_model_guard.py').write_text(t)
    legacy_proof[family+'-counter_fault_model_guard.py']=dict(seed=str(b/'work/worldsense-gpu-counter-profile/counter_fault_model_guard.py'),seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_path=str(folder/'counter_fault_model_guard.py'),current_sha256=hashlib.sha256(t.encode()).hexdigest(),path_replacement_only=True)
(main/'legacy-and-counter-guard-adaptation-proof.json').write_text(json.dumps(legacy_proof,indent=2)+'\n')
(main/'learned-preparation-stage.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_proof_root=str(current),legacy_proof_root=str(legacy),original16_current_checks_plus9_supplemental_checks=25,original_all_numeric_state_token_vision_lifecycle_guards_retained=True,helper_adaptations_inverse_exact=True,build_aliases=aliases,no_learned_or_actual_capture_run=True),indent=2)+'\n')
print('WS76 inverse-exact learned helpers prepared: original16+9supplemental checks; no learned/model dispatch')
