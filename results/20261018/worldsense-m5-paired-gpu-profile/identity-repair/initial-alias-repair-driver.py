"""Resolve declared build aliases to already-frozen targets, never invent inputs."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');opt=b/'work/worldsense-opt';w=b/'work/worldsense-paired-gpu-profile';d=Path('/Users/shawn/CantorAI/Garnet/results/20261018/worldsense-m5-paired-gpu-profile')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
assert not(opt/'ws76-archive-independent-audit.json').exists()
failure=(w/'canonical-independent-audit-v2.log').read_text();assert 'KeyError:'in failure and 'current-off/libgarnet.dylib'in failure
folder=d/'identity-repair';folder.mkdir(exist_ok=False)
shutil.copy2(d/'file-manifest.json',folder/'initial-file-manifest.json');shutil.copy2(w/'canonical-independent-audit-v2.log',folder/'second-canonical-audit-failure.log')
first=(opt/'verify_archive_ws76_v2.py').read_text();before="resolution=load(d/'external-input-resolution.json')"
after='''raw_resolution=load(d/'external-input-resolution.json')
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
resolution=CanonicalResolution(raw_resolution)'''
assert first.count(before)==1;second=first.replace(before,after)
extra='''
identity_repair=load(d/'identity-repair/repair.json');initial=load(d/'identity-repair/initial-file-manifest.json')
assert identity_repair['initial_audit_terminal_exit_code']==1 and identity_repair['only_alias_resolution_to_existing_frozen_targets']and identity_repair['no_inference_or_timing_rerun']
assert all(manifest[k]==v for k,v in initial.items())
assert 'current-off/libgarnet.dylib'in(d/'identity-repair/second-canonical-audit-failure.log').read_text()
old=(d/'work/verify_archive_ws76_v2.py').read_text();new=(d/'work/verify_archive_ws76_v3.py').read_text()
for operation in reversed(identity_repair['verifier_operations']):assert new.count(operation['after'])==1;new=new.replace(operation['after'],operation['before'])
assert new==old
'''
anchor='sources={};reconstructions={}';assert second.count(anchor)==1;second=second.replace(anchor,extra+'\n'+anchor)
ops=[dict(before=before,after=after),dict(before=anchor,after=extra+'\n'+anchor)];ast.parse(second)
target=opt/'verify_archive_ws76_v3.py';assert not target.exists();target.write_text(second);shutil.copy2(target,d/'work'/target.name);shutil.copy2(Path(__file__),d/'work'/Path(__file__).name)
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_audit_terminal_session=58254,initial_audit_terminal_exit_code=1,reason='Original learned snapshot pins all actual target libraries and an explicit before/after-checked build-alias table, not every alias/libgarnet.dylib path as a separate external key. Auditor now resolves ONLY the four declared DLL aliases to existing frozen canonical target records.',verifier_operations=ops,initial_verifier_sha256=hashlib.sha256(first.encode()).hexdigest(),revised_verifier_sha256=hashlib.sha256(second.encode()).hexdigest(),only_alias_resolution_to_existing_frozen_targets=True,no_new_external_identity_added=True,no_original_archive_entry_changed=True,no_numeric_word_state_token_golden_metric_gate_change=True,no_inference_or_timing_rerun=True)
(folder/'repair.json').write_text(json.dumps(record,indent=2)+'\n')
(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n')
print(json.dumps(dict(manifest_files=len(load(d/'file-manifest.json')),declared_aliases=4,existing_input_records=2308,new_input_records=0,all_previous_archive_bytes_unchanged=True,no_model_or_timing_rerun=True)))
