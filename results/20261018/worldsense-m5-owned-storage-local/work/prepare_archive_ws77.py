"""Prepare canonical archive/auditor after a valid WS10 speed rejection only.

This script is prospective tooling, not a prediction of the measured outcome.
If another stage fails or all speed gates pass, a different declared archive
scope must preserve those actual results rather than force this scenario.
"""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt'
def adapt(t):return t.replace('worldsense-minimal-selective-defaults','worldsense-owned-storage-local').replace('worldsense-m5-minimal-selective-defaults','worldsense-m5-owned-storage-local').replace('WS-M5-74','WS-M5-77').replace('WS74','WS77').replace('ws74','ws77')
t=adapt((f/'archive_ws74.py').read_text())
t=t.replace("w.parent/'worldsense-minimal-full-prefill'","w.parent/'worldsense-minimal-selective-defaults'")
t=t.replace("roles=['off','primary','on','fault','compat56','compat56-fault']","roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']")
t=t.replace("'primary_core_projection_words.bin.summary.json']","'primary_core_projection_words.bin.summary.json','owned_storage_projection_words.bin','owned_storage_projection_words.bin.summary.json']")
t=t.replace("source_files=1721","source_files=1722")
t=t.replace('Valid rejection: BOTHmodels derived generation rates fail unchanged2%gates despite favorable observedlatencies/wholeSkill. No later stages/promotion.','Valid WS10 speed rejection under unchanged ALL-model/whole-Skill2% preservation gates. Consult raw comparisons for exact failed metrics. No later stages/promotion.')
anchor="for path,info in load(w/'build-host-dependencies-before-dispatch.json')['files'].items():pin(path,info['sha256'])"
assert t.count(anchor)==1;t=t.replace(anchor,anchor+"\nfor path,digest in load(w/'host-codegen-review.json')['inputs'].items():pin(path,digest)")
ast.parse(t);p=f/'archive_ws77.py';assert not p.exists();p.write_text(t)
v=adapt((f/'verify_archive_ws74.py').read_text())
v=v.replace("len(current)==1721","len(current)==1722")
v=v.replace("seed_ref='4c811ec4e305317efac622177d4623b1cf8c83a4';seed_prefix='results/20261018/worldsense-m5-minimal-full-prefill/'","seed_ref='e15c3f4325682098bd4c7b4a17da62254e1e36c5';seed_prefix='results/20261018/worldsense-m5-minimal-selective-defaults/'")
a=v.index("changed=['cmake/GarnetMetal.cmake'");z=v.index("assert len(recon['overlay_files'])",a)
v=v[:a]+'''changed=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/owned_storage_local_test.cpp']
assert sorted(k for k in current if current[k]!=seed_source.get(k))==changed
assert set(current)-set(seed_source)=={changed[-1]} and not set(seed_source)-set(current)
operations=load(d/'source-transformations.json')
for rel in {op['path']for op in operations}:
 text=current[rel].decode()
 for op in reversed([op for op in operations if op['path']==rel]):
  if op['append']:assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.encode()==seed_source[rel]
assert load(d/'source-inverse-proof.json')['changed_or_added']==changed

'''+v[z:]
v=v.replace("len(recon['overlay_files'])==16","len(recon['overlay_files'])==17")
v=v.replace("roles=['off','primary','on','fault','compat56','compat56-fault']","roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']")
v=v.replace(r"r'\d+/9\s+Test\s+#\s*(\d+):[^\n]*Passed'",r"r'\d+/10\s+Test\s+#\s*(\d+):[^\n]*Passed'").replace('list(range(1,10))','list(range(1,11))').replace('out of 9','out of 10')
v=v.replace("selective=role in ['primary','on','fault'];compat=role.startswith('compat56');vector=role in ['on','fault']or compat","profile={'control':'on','raw-borrow':'off'}.get(role,role);selective=profile in ['primary','on','fault'];compat=profile.startswith('compat56');vector=profile in ['on','fault']or compat\n assert flags['OWNED_STORAGE_LOCAL']==('ON'if role in ['primary','on','fault','raw-borrow']else'OFF')")
v=v.replace("('OFF'if role=='off'else'ON')","('OFF'if profile=='off'else'ON')")
# Verify each program against immutable seed74, including new isolated modes.
a=v.index("for role,previous in [('off','off')");z=v.index("full_program=",a)
v=v[:a]+'''for role in roles:
 previous={'control':'on','raw-borrow':'off'}.get(role,role)
 for kind in ['main','prefill','core']:assert programs[role][kind]==batch([seed_ref+':'+seed_prefix+'actual-'+previous+'-'+kind+'-program.metal'])[0]
'''+v[z:]
v=v.replace("for role in ['on','fault']:assert programs","for role in ['on','fault','control']:assert programs")
v=v.replace("value.replace('worldsense-owned-storage-local','worldsense-minimal-full-prefill')==original","value.replace('worldsense-owned-storage-local','worldsense-minimal-selective-defaults')==original")
a=v.index('# Recursively prove');z=v.index('def canonical_learned_checks():',a)
chain='''# Recheck the complete immutable parent numerical-guard chain.
for child_ref,child_prefix,child_slug,parent_ref,parent_prefix,parent_slug in [
 (seed_ref,seed_prefix,'worldsense-minimal-selective-defaults','4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill'),
 ('4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation'),
 ('80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation','290f5742ba51c41197f24179cfee4876e96e789a','results/20261017/worldsense-m5-request-weight-residency/','worldsense-request-weight-residency')]:
 parent_proof=json.loads(batch([child_ref+':'+child_prefix+'learned-guard-adaptation-proof.json'])[0])
 for name,info in parent_proof.items():
  value=batch([child_ref+':'+child_prefix+name])[0].decode();original=batch([parent_ref+':'+parent_prefix+name])[0].decode()
  assert hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']and hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']
  for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
  assert value.replace(child_slug,parent_slug)==original
'''
v=v[:a]+chain+v[z:]
v=v.replace('len(journal)==16','len(journal)==18').replace("['native_tests_pass']==54","['native_tests_pass']==80")
v=v.replace("for role in ['off','primary','on','compat56']}","for role in ['off','primary','on','compat56','control','raw-borrow']}")
before="   enabled=role!='off';compat=role=='compat56';isolated=False;vector=role in ['on','compat56']";after="   profile={'control':'on','raw-borrow':'off'}.get(role,role);enabled=profile!='off';compat=profile=='compat56';isolated=False;vector=profile in ['on','compat56']\n   assert ('owned_storage_local'in call['backend'])==(role in ['primary','on','raw-borrow'])";assert v.count(before)==1;v=v.replace(before,after).replace("selective=role in ['primary','on']","selective=profile in ['primary','on']")
v=v.replace('==22792','==22867').replace('offline_malformed_copies_rejected=22792','offline_malformed_copies_rejected=22867')
v=v.replace("assert len(load(d/'build-host-dependencies-before-dispatch.json')['files'])==1875","assert len(load(d/'build-host-dependencies-before-dispatch.json')['files'])>1800")
v=v.replace('source_files=1721','source_files=1722').replace('native_CTest_pass=54','native_CTest_pass=80').replace('exact57_three_policy_imports_inverse_to71=True','exact_four_path_inverse_to74=True').replace('original_guard74_to71_to70_to56','original_guard77_to74_to71_to70_to56')
anchor="programs={role:{kind:";at=v.index(anchor)
lifetime='''# Independently recompute all new lifetime stream headers, words and EOF.
reference_lifetime=None
for role in roles:
 data=(d/('native-'+role+'-owned_storage_projection_words.bin')).read_bytes();offset=0;records=words=0
 for raw_mode in [False,True]:
  for mode in range(5):
   shapes=[(rows,columns)for rows in [1,2,16,32,64]for columns in [260,1024]]+[(rows,100352)for rows in [1,2]]
   for rows,columns in shapes:
    count=rows*columns;assert struct.unpack_from('<6Q',data,offset)==(rows,128,columns,int(raw_mode),mode,count);offset+=48
    expected=b''.join(struct.pack('<f',8*(row+1))*columns for row in range(rows));assert data[offset:offset+count*4]==expected;offset+=count*4;records+=1;words+=count
 assert offset==len(data)and records==120 and words==4487160
 if reference_lifetime is None:reference_lifetime=data
 assert reference_lifetime==data
 summary=load(d/('native-'+role+'-owned_storage_projection_words.bin.summary.json'));packed=role not in ['off','raw-borrow']
 assert summary['complete_records']==records and summary['complete_FP32_words']==words and summary['owned_storage_local_enabled']==(role in ['primary','on','fault','raw-borrow'])and summary['packed_cases']==(60 if packed else 0)and summary['raw_cases']==(60 if packed else 120)and summary['invalid_cases_rejected']==72 and summary['prepared_cases']==40 and summary['live_logical_accounting_restored']
host=load(d/'host-codegen-review.json');assert host['selected_local_ownership_removed_both_raw_and_packed']
for role,functions in host['details'].items():
 for kind,record in functions.items():
  p=d/(role+'-'+kind+'.arm64.txt');assert sha(p)==record['disassembly_sha256'];lines=p.read_text().splitlines();owner=next(i for i,line in enumerate(lines)if re.search(r'fmov\\s+x\\d+, d0',line));last=next(i for i in range(owner+1,len(lines))if '_objc_msgSend'in lines[i]);window=lines[owner:last]
  assert window==record['selection_window'];enabled=role in ['raw-borrow','on'];assert(sum('_objc_retain'in x for x in window),sum('_objc_release'in x for x in window))==((0,0)if enabled else((1,0)if role=='off'else(2,1)))
  assert any('0xf8290108'in x for x in lines[max(0,owner-12):owner]);release=next(i for i in range(last+1,len(lines))if '__shared_weak_count14__release_weakEv'in lines[i]);assert any('0xf8e90108'in x for x in lines[release-10:release])
  if kind=='encode':assert any('_objc_msgSend$endEncoding'in x for x in lines[last:release])
'''
v=v[:at]+lifetime+v[at:]
# Every host-codegen consumed object/tool/flag/library identity remains explicit.
anchor="assert len(load(d/'build-host-dependencies-before-dispatch.json')['files'])>1800";assert v.count(anchor)==1
v=v.replace(anchor,"for path,digest in load(d/'host-codegen-review.json')['inputs'].items():assert resolution[path]['sha256']==digest\n"+anchor)
v=v.replace('original186_complete_projection_streams_and_new20records432804words_exact=True','original186_complete_projection_streams_and_new20records432804words_exact=True,new120records4487160words_each_eight_roles_exact=True,actual_HOST_selection_ownership_removed_with_owner_kept=True')
ast.parse(v);p=f/'verify_archive_ws77.py';assert not p.exists();p.write_text(v)
print('Prepared WS77 canonical tools for a possible valid WS10-speed reject: source1722/17overlays, eight roles/native80/lifetime120/fulllearned18/guard chain/new marker22867negatives. No copies, model calls or outcome claim.')
