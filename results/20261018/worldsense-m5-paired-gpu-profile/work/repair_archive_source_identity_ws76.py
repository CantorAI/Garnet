"""Use the separately frozen source closures for runtime source identities."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');opt=b/'work/worldsense-opt';w=b/'work/worldsense-paired-gpu-profile';d=Path('/Users/shawn/CantorAI/Garnet/results/20261018/worldsense-m5-paired-gpu-profile');load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not(opt/'ws76-archive-independent-audit.json').exists()
failure=(w/'canonical-independent-audit-v3.log').read_text();assert 'KeyError:'in failure and 'legacy/source/runtime/metal_serving.py'in failure
folder=d/'source-identity-repair';folder.mkdir(exist_ok=False);shutil.copy2(d/'file-manifest.json',folder/'initial-file-manifest.json');shutil.copy2(w/'canonical-independent-audit-v3.log',folder/'third-canonical-audit-failure.log')
first=(opt/'verify_archive_ws76_v3.py').read_text();second=first;ops=[]
before="capture_frozen=load(cap/'identities-before-dispatch.json');cohort=load(cap/'cohort-manifest.json');terminal=load(cap/'controller-terminal.json')"
after=before+'''
native_source_groups=load(d/'main/native-identities-before-dispatch.json')['sources']
for family,key in [('current','source'),('legacy','legacy_source')]:
 expected={name:digest(value)for name,value in sources[family].items()}
 assert capture_frozen[key]==frozen[key]==native_source_groups[family]['files']==expected
'''
assert second.count(before)==1;second=second.replace(before,after);ops.append(dict(before=before,after=after))
before="assert resolution[worker['library']]['sha256']==worker['library_sha256']and resolution[worker['serving_source']]['sha256']==worker['serving_sha256']"
after='''assert resolution[worker['library']]['sha256']==worker['library_sha256']
 assert worker['source_root']==native_source_groups[family]['root']and Path(worker['serving_source'])==Path(worker['source_root'])/'runtime/metal_serving.py'
 assert digest(sources[family]['runtime/metal_serving.py'])==worker['serving_sha256']'''
assert second.count(before)==1;second=second.replace(before,after);ops.append(dict(before=before,after=after))
extra='''
source_identity_repair=load(d/'source-identity-repair/repair.json');initial=load(d/'source-identity-repair/initial-file-manifest.json')
assert source_identity_repair['only_source_closure_identity_mapping']and source_identity_repair['no_original_archive_entry_changed']and source_identity_repair['no_inference_or_timing_rerun']
assert all(manifest[k]==v for k,v in initial.items())
assert 'legacy/source/runtime/metal_serving.py'in(d/'source-identity-repair/third-canonical-audit-failure.log').read_text()
old=(d/'work/verify_archive_ws76_v3.py').read_text();new=(d/'work/verify_archive_ws76_v4.py').read_text()
for operation in reversed(source_identity_repair['verifier_operations']):assert new.count(operation['after'])==1;new=new.replace(operation['after'],operation['before'])
assert new==old
'''
anchor='sources={};reconstructions={}';assert second.count(anchor)==1;second=second.replace(anchor,extra+'\n'+anchor);ops.append(dict(before=anchor,after=extra+'\n'+anchor))
ast.parse(second);target=opt/'verify_archive_ws76_v4.py';assert not target.exists();target.write_text(second);shutil.copy2(target,d/'work'/target.name);shutil.copy2(Path(__file__),d/'work'/Path(__file__).name)
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_audit_terminal_session=93161,initial_audit_terminal_exit_code=1,reason='Runtime source paths are frozen in the complete separate current/legacy source maps, not all as external-file keys. Auditor now checks equality of ALL native/learned/capture source maps and hashes serving source from the reconstructed closure; actual library identities remain strict existing external records.',verifier_operations=ops,initial_verifier_sha256=hashlib.sha256(first.encode()).hexdigest(),revised_verifier_sha256=hashlib.sha256(second.encode()).hexdigest(),only_source_closure_identity_mapping=True,no_original_archive_entry_changed=True,no_new_external_identity_added=True,all_original_numeric_word_state_token_golden_clock_CSV_metric_checks_retained=True,no_inference_or_timing_rerun=True)
(folder/'repair.json').write_text(json.dumps(record,indent=2)+'\n');(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(manifest_files=len(load(d/'file-manifest.json')),source_maps_separately_verified=True,new_external_input_keys=0,no_model_or_timing_rerun=True)))
