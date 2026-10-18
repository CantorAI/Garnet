"""Clone immutable source, change build configuration only, no source mutation."""
from pathlib import Path
import json,hashlib,shutil,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-minimal-full-prefill';old=w.parent/'worldsense-primary-core-isolation';s=r/'out/research/worldsense-minimal-full-prefill/source';seed=r/'out/research/worldsense-primary-core-isolation/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=load(w/'protocol.json');m=load(w/'seed-source-manifest.json');assert p['state']=='DESIGN_BEFORE_SOURCE'and p['design_sha256']==sha(w/'native-design-before-implementation.txt')and not s.exists()
assert all(sha(seed/k)==v for k,v in m.items());shutil.copytree(seed,s,ignore=shutil.ignore_patterns('__pycache__','.git'))
actual={str(p.relative_to(s)):sha(p)for p in s.rglob('*')if p.is_file()};assert actual==m and len(m)==1721
(w/'correctness-source-manifest.json').write_text(json.dumps(actual,indent=2)+'\n');(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1721,source_files=1721,changed_or_added=[],all_source_byte_exact_to_published70=True,inference_CMake_runtime_shader_FP_CUDA_engine_original_tests_unchanged=True),indent=2)+'\n')
shutil.copytree(old/'build-profile',w/'build-profile')
for name in ['restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib']:shutil.copy2(old/name,w/name)
text=(old/'build.py').read_text().replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill')
before="flags.update({key:full for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_CACHED_MATRIX_METADATA']})"
after="""flags['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']='OFF'
 flags.update({key:'ON'if compat or role in ['on','fault']else'OFF'for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']})
 flags.update({key:full for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_CACHED_MATRIX_METADATA']})"""
assert text.count(before)==1;text=text.replace(before,after);ast.parse(text);(w/'build.py').write_text(text)
(w/'build-driver-adaptation.json').write_text(json.dumps(dict(seed=str(old/'build.py'),seed_sha256=sha(old/'build.py'),path_replacement=['worldsense-primary-core-isolation','worldsense-minimal-full-prefill'],before=before,after=after,result_sha256=sha(w/'build.py')),indent=2)+'\n')
d=load(b/'work/worldsense-next.json');d.update(state='EXACT_SOURCE_READY_BUILD_NEXT',source_implemented=True,source_files=1721,inference_source_delta=0);(b/'work/worldsense-next.json').write_text(json.dumps(d,indent=2)+'\n')
print('WS71 exact1721source cloned; zerochanged; sixrole existingflags driver ready, no build/model yet.')
