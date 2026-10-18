"""Import only the immutable graph-local policy into the minimal FULL source."""
from pathlib import Path
import json,hashlib,shutil,ast,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-minimal-selective-defaults';old=w.parent/'worldsense-minimal-full-prefill';s=r/'out/research/worldsense-minimal-selective-defaults/source';seed=r/'out/research/worldsense-minimal-full-prefill/source';policy=r/'out/research/worldsense-packed-only-defaults/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'protocol.json');m=load(w/'seed-source-manifest.json');assert p['state']=='DESIGN_BEFORE_SOURCE'and p['design_sha256']==sha(w/'native-design-before-implementation.txt')and not s.exists()
assert len(m)==1721 and all(sha(seed/k)==v for k,v in m.items());audit=load(w/'read-only-consumer-prior-audit.json');assert all(sha(policy/k)==v for k,v in audit['policy_source_hashes'].items())
shutil.copytree(seed,s,ignore=shutil.ignore_patterns('__pycache__','.git'));changes=[]
def insert(rel,before,addition,position='after'):
 path=s/rel;t=path.read_text();assert t.count(before)==1,(rel,before,t.count(before));replacement=before+addition if position=='after'else addition+before;result=t.replace(before,replacement);assert result.replace(replacement,before)==t;path.write_text(result);changes.append(dict(path=rel,before=before,after=replacement,seed_sha256=m[rel],result_sha256=sha(path)))
cmake=(policy/'cmake/GarnetMetal.cmake').read_text();start=cmake.index('option(GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS');block=cmake[start:];assert block.endswith('endif()\n')and block.count('option(')==1
path=s/'cmake/GarnetMetal.cmake';before=path.read_text();assert 'PRIMARY_ONLY_LAZY_DEFAULTS'not in before;path.write_text(before+'\n'+block);changes.append(dict(path='cmake/GarnetMetal.cmake',before='',after='\n'+block,append=True,seed_sha256=m['cmake/GarnetMetal.cmake'],result_sha256=sha(path)))
guard=(w/'immutable57-policy-guard.txt').read_text();assert guard.strip()in(policy/'src/backends/metal/compiled_graph.cpp').read_text()
insert('src/backends/metal/compiled_graph.cpp','        if(!ready_||outputParts_.size()!=3||inputs_.size()<3||residentOps_.size()!=ops_.size())return;\n',guard)
entry=(policy/'src/entry/metal_main.cpp').read_text();start=entry.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS');marker=entry[start:entry.index('#endif',start)+len('#endif')]+ '\n'
insert('src/entry/metal_main.cpp','        return info;\n',marker,'before')
actual={str(p.relative_to(s)):sha(p)for p in s.rglob('*')if p.is_file()};changed=sorted(k for k in actual if actual[k]!=m[k]);assert actual.keys()==m.keys()and changed==sorted(v['path']for v in changes)and len(changed)==3
for change in changes:
 t=(s/change['path']).read_text()
 if change.get('append'):assert t.endswith(change['after']);inverse=t[:-len(change['after'])]
 else:assert t.count(change['after'])==1;inverse=t.replace(change['after'],change['before'])
 assert hashlib.sha256(inverse.encode()).hexdigest()==m[change['path']]
(w/'correctness-source-manifest.json').write_text(json.dumps(actual,indent=2)+'\n');(w/'source-transformations.json').write_text(json.dumps(changes,indent=2)+'\n');(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1721,source_files=1721,changed_or_added=changed,exact57_guard_cmake_marker_import=True,all_three_inverse_exact_to_published71=True,all_other1718files_byte_exact=True,shader_FP_CUDA_engine_original_tests_unchanged=True),indent=2)+'\n')
shutil.copytree(old/'build-profile',w/'build-profile')
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','primary_core_isolation_test.cpp']:shutil.copy2(old/name,w/name)
t=(old/'build.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-minimal-selective-defaults');key=" flags['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']='OFF'";assert t.count(key)==1;t=t.replace(key,key+"\n flags['GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS']='ON'if role in ['primary','on','fault']else'OFF'");ast.parse(t);(w/'build.py').write_text(t)
(w/'build-driver-adaptation.json').write_text(json.dumps(dict(seed=str(old/'build.py'),seed_sha256=sha(old/'build.py'),path_replacement=['worldsense-minimal-full-prefill','worldsense-minimal-selective-defaults'],added_flag='PRIMARY_ONLY_LAZY_DEFAULTS=ON only PRIMARY/ON/FAULT, OFF RAW/COMPAT56',result_sha256=sha(w/'build.py')),indent=2)+'\n')
d=load(b/'work/worldsense-next.json');d.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='SOURCE_READY_BUILD_NEXT',source_implemented=True,source_files=1721,inference_source_delta=3);(b/'work/worldsense-next.json').write_text(json.dumps(d,indent=2)+'\n')
print('WS74 source1721, exact57 policy3file import and inverse71 verified; sixrole build driver ready, no model yet.')
