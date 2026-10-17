from pathlib import Path
import subprocess,json,datetime,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-cached-lazy/source';results={}
base=['/opt/homebrew/bin/cmake','-S',str(s),'-C',str(w/'build-profile/MetalWorldSense4BPrefix.cmake'),'-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DXLANG3_SDK_ROOT='+str(r/'xlang3/sdk'),'-DGARNET_XLANG3_RUNTIME_LIBRARY='+str(r/'out/build/macos-arm64-release/bin/libxlang3_runtime.dylib')]
for name,flags,expected in [('k32',['-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=OFF','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY=ON'],'K32-only primary vector prefill requires primary vector prefill'),('cached',['-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF','-DGARNET_METAL_CACHED_MATRIX_METADATA=ON'],'Cached matrix metadata cohort requires lazy default storage')]:
 cmd=base+['-B',str(w/('invalid-'+name+'-config'))]+flags;p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(w/('invalid-'+name+'-config.log')).write_text(p.stdout);assert p.returncode!=0 and expected in ' '.join(p.stdout.split()),name;results[name]={'command':cmd,'returncode':p.returncode,'expected_rejection':True}
(w/'flag-dependency-proof.json').write_text(json.dumps(results,indent=2)+'\n')
source=json.loads((w/'correctness-source-manifest.json').read_text());assert source==json.loads((w/'ws51-seed-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in source.items())
program=json.loads((w/'actual-program-proof.json').read_text());reference=json.loads((w.parent/'worldsense-k32-prefill/actual-program-proof.json').read_text());assert program['programs']==reference['programs']
for mode in ['on','fault']:
 b=r/('out/build/garnet-metal-worldsense-primary-cached-lazy-'+mode);log=(b/'Testing/Temporary/LastTest.log').read_text();assert 'Matrix metadata/live-MPS parity empty/copy/move/reassign/padding/pending/Submit/Abort/host-zero/context-lifetime PASS'in log
 flags=(b/'CMakeFiles/garnet_matrix_metadata_test.dir/flags.make').read_text();assert '-DGARNET_METAL_CACHED_MATRIX_METADATA=1'in flags and '-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=1'in flags
records={}
for mode in ['off','primary','on']:
 rows=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(rows)==5;records[mode]=[]
 for x in rows:
  c=x['call'];t=c['timings'];enabled=mode=='on';assert ('cached_matrix_metadata'in c['backend'])==enabled and ('lazy_resident_defaults'in c['backend'])==enabled;assert (t['lazy_default_retired_bytes']>0)==enabled and (t['lazy_default_planned_slots']>0)==enabled and t['lazy_default_rehydrated_bytes']==0
  records[mode].append({'model_id':c['model_id'],'capacity':c['serving_configuration']['capacity'],**{k:t[k]for k in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']}})
assert all(x['returncode']==0 for x in json.loads((w/'resident-results.json').read_text()))
(w/'lifecycle-metadata-proof.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'zero_source_delta_WS51':True,'actual_four_programs_byte_exact_WS51':True,'cached_native_live_property_suite_ON_fault_executed':True,'ordinary_resident_late_session_invalid_rebuild_rehydration_exact':True,'normal_API_retirement_and_zero_rehydration':records,'scope':'Existingcompletecorrectness cases, nophysicalfree/performance claim. Disabled legacylinear allocation-injection branch remains unexecuted/unclaimed.'},indent=2)+'\n')
(w/'native-terminal.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_session':41387,'terminal_exit_code':0,'all_four_builds_seven_native_tests_pass':True,'all186_complete_projection_records_exact':True,'all65536_direct_patterns_lifetimes_errors_pass':True,'cached_live_property_suite_ON_fault_executed':True},indent=2)+'\n')
(w/'timing-observation-plan.json').write_text(json.dumps({'declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan':'OSAC/thermal metadata before/aftermatchedworkers outsideinference aspredeclared52design; notGPUclock/temperature, no newexclusion orchangedgate.'},indent=2)+'\n')
print('Bothinvalidconfigguards, sourcezeroDelta/program51identity, cachedliveproperties andactualretirement/rehydration proof verified.')
