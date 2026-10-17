from pathlib import Path
p=Path('work/worldsense-opt');a=(p/'archive_primary_cached_lazy.py').read_text().replace('WS52','WS53').replace('WS-M5-52','WS-M5-53').replace('worldsense-primary-cached-lazy','worldsense-wide-cached-prefill').replace('worldsense-m5-primary-cached-lazy','worldsense-m5-wide-cached-prefill').replace('archive_primary_cached_lazy','archive_wide_cached_prefill').replace('verify_primary_cached_lazy','verify_wide_cached_prefill')
a=a.replace("'.ll','.cmake']", "'.ll','.cmake','.patch']")
a=a.replace("for name in ['archive_wide_cached_prefill.py','verify_wide_cached_prefill.py']:","for name in ['archive_wide_cached_prefill.py','verify_wide_cached_prefill.py','seed_ws53.py','merge_ws53.py','drivers_ws53.py','benchmark_ws53.py','finalize_ws53.py','archive_drivers_ws53.py']:")
a=a.replace('Matched primary-only stage adds prospective5%VLM TTFT/2%memory preservation, other original gates unchanged.','Actual WS10 screen precedes matched storage, per prospective53 order; other original gates unchanged.')
(p/'archive_wide_cached_prefill.py').write_text(a)
v=(p/'verify_primary_cached_lazy.py').read_text().replace('WS52','WS53').replace('WS-M5-52','WS-M5-53').replace('worldsense-primary-cached-lazy','worldsense-wide-cached-prefill').replace('worldsense-m5-primary-cached-lazy','worldsense-m5-wide-cached-prefill')
v=v.replace("==7 and len(closure)==1716", "==9 and len(closure)==1717").replace("'source_overlay':7", "'source_overlay':9").replace('source_overlay=7','source_overlay=9')
v=v.replace("t['packed_matrices']==398 and t['raw_small_matrices']==48","t['packed_matrices']==201 and t['raw_small_matrices']==245").replace('5921308672','4200857600').replace('9033678080','9850930176')
v=v.replace("assert t['retained_primary_and_small_raw_bytes']==9850930176", "assert t['retained_primary_and_small_raw_bytes']==9850930176\nassert t['HOST_raw_FP32_words_checked']==1724383232\nassert sum(m['HOST_raw_FP32_words_checked']for m in corpus['matrices'])==1724383232\nassert all(m['HOST_raw_FP32_words_checked']==(0 if m['packed']else m['rows']*m['inner'])for m in corpus['matrices'])")
v=v.replace("assert closure==load(d/'ws51-seed-source-manifest.json')", "assert load(d/'wide-scope-proof.json')['wide_admission_scope_verified']\nassert load(d/'wide-scope-proof.json')['entire_compiled_loader_exact_reconstructed_merge']")
v=v.replace(" for cap in ['GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_LAZY_RESIDENT_DEFAULTS']:assert flags[mode][cap]==('ON'if mode=='on'else'OFF')", " assert flags[mode]['GARNET_METAL_CACHED_MATRIX_METADATA']==('ON'if mode=='on'else'OFF')\n assert all(flags[mode][cap]==('OFF'if mode=='off'else'ON')for cap in ['GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_WIDE_PRIMARY_STORAGE'])")
a=v.index("original=subprocess.check_output(['git'",v.index('# New direct staging'))
b=v.index("for mode in ['on','fault']:",a)
v=v[:a]+'''# Independently reconstruct wide changes against immutable archived WS52 source.
import tempfile,shutil
seed52=load(d/'ws52-seed-source-manifest.json');ref52='44eb9a49088e6b1457057572561218d53481f484'
recon52=json.loads(subprocess.check_output(['git','-C',str(r),'show',ref52+':results/20261017/worldsense-m5-primary-cached-lazy/source-reconstruction.json']))
rels=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/compiled_graph.h','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm']
def bytes52(rel):
 key=('results/20261017/worldsense-m5-primary-cached-lazy/source/'+rel)if rel in recon52['overlay_files']else rel
 raw=subprocess.check_output(['git','-C',str(r),'show',(ref52 if rel in recon52['overlay_files']else base)+':'+key]);assert hashlib.sha256(raw).hexdigest()==seed52[rel];return raw
with tempfile.TemporaryDirectory(prefix='ws53-archive-inverse-')as folder:
 tdir=Path(folder)
 for rel in rels:
  (tdir/rel).parent.mkdir(parents=True,exist_ok=True);(tdir/rel).write_bytes(bytes52(rel))
 subprocess.run(['git','apply','--exclude=src/entry/metal_main.cpp','--exclude=test2026/metal/primary_storage_test.cpp',str(d/'wide46-over-f21.patch')],cwd=tdir,check=True)
 for rel in rels+['src/backends/metal/wide_checkpoint_policy.h']:
  text=(d/'source'/rel).read_text()
  if rel.endswith('metal_context.h'):text=text.replace('    std::vector<uint32_t> RawBF16ReadWords(const MetalBF16Weight&,size_t start,size_t count);\\n','',1)
  if rel.endswith('metal_context.mm'):
   marker='\\nnamespace Garnet {\\nstd::vector<uint32_t> MetalContext::RawBF16ReadWords(';assert text.count(marker)==1;text,reader=text.split(marker,1)
   assert 'std::memcpy(&word,data+2*(start+i),2)'in reader and 'count>total-start'in reader
  assert text==(tdir/rel).read_text(),rel
entry=(d/'source/src/entry/metal_main.cpp').read_text();marker='#ifdef GARNET_METAL_WIDE_PRIMARY_STORAGE\\n        info+="; wide_primary_storage";\\n#endif\\n';assert entry.count(marker)==1 and entry.replace(marker,'',1).encode()==bytes52('src/entry/metal_main.cpp')
''' +v[b:]
v=v.replace("storage=load(d/'prospective-storage-counters.json')['native_owner_tuples']", "storage={'201':[8401715200,6402163712,32819200]}")
v=v.replace("'primary_u32_metadata']", "'primary_u32_metadata','wide_primary_storage']")
v=v.replace("('lazy_resident_defaults'in c['backend'])==enabled", "('lazy_resident_defaults'in c['backend'])==packed")
v=v.replace("==enabled and (tt['lazy_default_planned_slots']>0)==enabled", "==packed and (tt['lazy_default_planned_slots']>0)==packed")
v=v.replace("if not enabled:assert tt['lazy_default_retired_bytes']==tt['lazy_default_planned_slots']==0", "if not packed:assert tt['lazy_default_retired_bytes']==tt['lazy_default_planned_slots']==0")
a=v.index("                            assert count in ([201,398]")
b=v.index("                    identity=",a)
v=v[:a]+'''                            assert count==201
                            assert tt['lazy_default_retired_bytes']>0 and tt['lazy_default_planned_slots']>0
                            if c['model_id']=='Qwen3.5-4B':
                                assert delta['prefill_encoded']>0 and delta['head_encoded']>0
                                assert delta['vector_decode_encoded']==delta['decode_encoded']>0 and delta['vector_head_encoded']==delta['head_encoded']
                            else:assert c['model_id']=='Qwen3-1.7B'and all(v==0 for v in delta.values())
''' +v[b:]
# Archive is a valid early rejection, not a complete 116-call prototype.
v=v.replace("assert decision['stage_outcomes']==outcomes", "assert decision['stage_outcomes']==outcomes=={'resource':True,'production-speed':False}\nassert load(d/'production-speed-terminal.json')['controller_session']==80721\nassert load(d/'resource-terminal.json')['controller_session']==25263\nassert not any((d/stage/'analysis.json').exists()for stage in ['primary-speed','speed','confirmation'])")
import ast
ast.parse(a if False else v)
(p/'verify_wide_cached_prefill.py').write_text(v)
print('WS53 archive/audit drivers prepared')
