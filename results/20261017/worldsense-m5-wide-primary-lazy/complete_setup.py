"""Resume unlaunched setup after an unrelated legacy file failed UTF8 decoding."""
from pathlib import Path
import json,hashlib,shutil,ast
w=Path(__file__).resolve().parent;cwd=w.parent.parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-wide-primary-lazy/source';old=w.parent/'worldsense-wide-storage';old_source=r/'out/research/worldsense-wide-storage/source'
assert not(r/'out/build/garnet-metal-worldsense-wide-primary-lazy-on').exists()
assert hashlib.sha256((w/'native-design-before-implementation.txt').read_bytes()).hexdigest()==json.loads((w/'design-identity.json').read_text())['design_sha256']
seed=json.loads((w/'classifier-seed-source-manifest.json').read_text());canonical=json.loads((w/'seed-source-manifest.json').read_text())
targets={rel for rel,h in seed.items()if canonical.get(rel)!=h};assert len(targets)==8
before={rel:hashlib.sha256((s/rel).read_bytes()).hexdigest()for rel in seed}
assert all(before[rel]==h for rel,h in seed.items()if rel not in targets)
(w/'setup-source-before-repair.json').write_text(json.dumps(before,indent=2)+'\n')
for rel in targets:
 p=old_source/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==seed[rel];shutil.copy2(p,s/rel)
p=s/'src/backends/metal/compiled_graph.cpp';t=p.read_text();early='#ifdef GARNET_METAL_WIDE_CHECKPOINT_STORAGE\n    if(!wideCheckpoint_)return;\n#endif\n';assert t.count(early)==1;p.write_text(t.replace(early,'',1))
for rel in targets:
 p=s/rel;t=p.read_text();t=t.replace('GARNET_METAL_WIDE_CHECKPOINT_STORAGE','GARNET_METAL_WIDE_PRIMARY_STORAGE').replace('WideStorageEligible','WidePrimaryEligible').replace('wide_storage_eligible','wide_primary_eligible').replace('wide_checkpoint_storage','wide_primary_storage')
 t=t.replace('Scope primary and lazy default storage to consistent wide checkpoints','Pack consistent wide checkpoints with global safe default retirement').replace('Wide checkpoint storage requires primary weights and lazy defaults','Wide primary storage profile requires primary weights and global lazy defaults')
 p.write_text(t)
manifest={rel:hashlib.sha256((s/rel).read_bytes()).hexdigest()for rel in seed};assert len(manifest)==1716
changes=[rel for rel,h in manifest.items()if canonical.get(rel)!=h];assert set(changes)==targets
for rel,h in canonical.items():
 if '/cuda/'in rel or rel in ['src/backends/metal/primary_block_shader.h','src/backends/metal/primary_block_codec.h']:assert manifest[rel]==h
for name in ['correctness-source-manifest.json','starting-source-manifest.json']:(w/name).write_text(json.dumps(manifest,indent=2)+'\n')
(w/'implementation-audit.json').write_text(json.dumps({'changed_files':changes,'source_files':1716,'scope':'Classifier/admission/shared proof from44; canonical global retirement plan; design predates source changes. No native or performance tests yet.','primary_shader_CPUcodec_FP32_CUDA_unchanged':True},indent=2)+'\n')
for name in ['build.py','build_fault.py','build_corpus.py','wait_cantor.py','restore_normal.py','primary_corpus_test.cpp']:
 t=(old/name).read_text().replace('worldsense-wide-storage','worldsense-wide-primary-lazy').replace('GARNET_METAL_WIDE_CHECKPOINT_STORAGE','GARNET_METAL_WIDE_PRIMARY_STORAGE')
 if name.endswith('.py'):ast.parse(t)
 (w/name).write_text(t)
shutil.copy2(old/'before-libgarnet.dylib',w/'before-libgarnet.dylib');shutil.copy2(old/'cases.json',w/'cases.json')
(w/'prospective-storage-counters.json').write_text(json.dumps({'native_owner_tuples':{'201':[8401715200,6402163712,32819200]},'retained_primary_and_small_raw_bytes':9850930176,'prior_global_default_retirement_bytes':3036086272,'projected_combined_Metal_bytes':16867738624,'projection_not_new_measurement':True},indent=2)+'\n')
(w/'setup-development-failure.txt').write_text('Initial source-renaming setup tried UTF8 decoding every C++/CMake source, including unrelated legacy files. It aborted on UnicodeDecodeError before build.py or any native test existed. The shell then attempted the absent build.py and exited2. No native build/test/timing was launched. Complete setup now reads only the eight known UTF8 overlay files, restores those exact tested WS44 seed bytes before applying the declared change, verifies all other1716-8 source hashes untouched, and freezes the final1716 closure. The partial pre-repair manifest and failed build-controller log are retained. Numerical/performance gates unchanged.\n')
print('WS46 setup completed,8known overlay files only; unrelated source bytes preserved; source1716 frozen.')
