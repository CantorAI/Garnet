from pathlib import Path
import hashlib,json,subprocess,tempfile,shutil,difflib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-wide-cached-prefill/source';seed=r/'out/research/worldsense-primary-cached-lazy/source';wide=r/'out/research/worldsense-wide-primary-lazy/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((w/'correctness-source-manifest.json').read_text());old=json.loads((w/'ws52-seed-source-manifest.json').read_text());assert len(m)==1717 and all(sha(s/k)==h for k,h in m.items())and all(sha(seed/k)==h for k,h in old.items())
rels=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm']
with tempfile.TemporaryDirectory(prefix='ws53-inverse-',dir=w.parent)as folder:
 t=Path(folder)
 for rel in rels:
  (t/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/rel,t/rel)
 proc=subprocess.run(['git','apply','--exclude=src/entry/metal_main.cpp','--exclude=test2026/metal/primary_storage_test.cpp',str(w/'wide46-over-f21.patch')],cwd=t,text=True,capture_output=True)
 assert proc.returncode==0,proc.stderr
 for rel in rels+['src/backends/metal/wide_checkpoint_policy.h']:
  actual=(s/rel).read_text()
  if rel.endswith('metal_context.h'):
   declaration='    std::vector<uint32_t> RawBF16ReadWords(const MetalBF16Weight&,size_t start,size_t count);\n';assert actual.count(declaration)==1;actual=actual.replace(declaration,'',1)
  if rel.endswith('metal_context.mm'):
   marker='\nnamespace Garnet {\nstd::vector<uint32_t> MetalContext::RawBF16ReadWords('
   assert actual.count(marker)==1;original,reader=actual.split(marker,1);actual=original
   assert reader.endswith('\n}\n}\n')and 'std::memcpy(&word,data+2*(start+i),2)'in reader and 'count>total-start'in reader and 'count>8*1024*1024'in reader
   assert all(x not in reader for x in ['newBuffer','commit]','waitUntilCompleted','Submit('])
  assert actual==(t/rel).read_text(),rel
entry=(s/'src/entry/metal_main.cpp').read_text();marker='#ifdef GARNET_METAL_WIDE_PRIMARY_STORAGE\n        info+="; wide_primary_storage";\n#endif\n';assert entry.count(marker)==1 and entry.replace(marker,'',1)==(seed/'src/entry/metal_main.cpp').read_text()
assert (s/'src/backends/metal/wide_checkpoint_policy.h').read_bytes()==(wide/'src/backends/metal/wide_checkpoint_policy.h').read_bytes()
base=subprocess.check_output(['git','-C',str(r/'Garnet'),'show','f21c1a2584031ee9d1c129d1e58818f9d1c826d2:test2026/metal/primary_storage_test.cpp'],text=True).splitlines(keepends=True)
previous=(wide/'test2026/metal/primary_storage_test.cpp').read_text().splitlines(keepends=True);test=(s/'test2026/metal/primary_storage_test.cpp').read_text()
for op,i,j,k,l in difflib.SequenceMatcher(None,base,previous,autojunk=False).get_opcodes():
 if op=='equal':continue
 assert op=='insert';addition=''.join(previous[k:l]);assert test.count(addition)==1;test=test.replace(addition,'',1)
a=' MetalContext gpu;auto initial=MetalContext::PrimaryStats();';b=' need(MetalContext::PrimaryStats().activeWeights==initial.activeWeights,"Raw reader retained primary owners");\n';assert test.count(a)==test.count(b)==1;start=test.index(a)+len(a);end=test.index(b,start)+len(b);test=test[:start]+test[end:];assert test==(seed/'test2026/metal/primary_storage_test.cpp').read_text()
proof={'wide_admission_scope_verified':True,'all_six_wide_changes_reconstructed_from_immutable52_plus46_delta':True,'entire_compiled_loader_exact_reconstructed_merge':True,'bounded_HOST_only_raw_reader_is_only_additional_context_delta':True,'entry_only_adds_wide_capability_marker':True,'native_test_original52_preserved_plus_original46_and_new_raw_read_cases':True,'protected_shader_codec_wrapper_projection_CUDA_unchanged':True,'source_files':1717,'canonical_overlays':9,'scope':'Mechanical source audit, plus actual native/model proofs required; host raw-storage readback is not a GPU reader or a speed improvement.'}
assert all(m[k]==h for k,h in old.items()if '/cuda/'in k or k in ['runtime/metal_serving.py','src/backends/metal/primary_block_shader.h','src/backends/metal/primary_vector_prefill_shader.h','test2026/metal/primary_projection_test.cpp'])
(w/'wide-scope-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
