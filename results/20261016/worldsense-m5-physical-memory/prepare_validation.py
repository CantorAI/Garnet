from pathlib import Path
import shutil,ast
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-cached-matrix-metadata'
for f in ['fixture-1-8','fixture-2-64']:shutil.copytree(old/f,w/f)
s=(old/'resident_guard.py').read_text();needle="        assert (graph.lazy_default_retired_bytes>0)";idx=s.index(needle)
block='''        if index==0:
            before=snapshot(session);other_before=snapshot(other);before_hidden=session.hidden().tolist()
            descriptors=[dict(object=graph,tag='layer')];sessions=[dict(object=session,tag='session')]
            one=json.loads(garnet.memory_inventory_json(descriptors,session));many=json.loads(garnet.memory_inventory_many_json(descriptors,sessions))
            for key in ['buffers','references','unique_allocated_bytes']:assert one[key]==many[key], 'Many-session API differs from existing single-session collection'
            dup=json.loads(garnet.memory_inventory_many_json(descriptors+descriptors,sessions+sessions))
            assert dup['buffers']==many['buffers'] and dup['unique_allocated_bytes']==many['unique_allocated_bytes'] and len(dup['references'])==2*len(many['references'])
            extra=json.loads(garnet.memory_inventory_json([],other));combined=json.loads(garnet.memory_inventory_many_json(descriptors,sessions+[dict(object=other,tag='other')]))
            assert combined['unique_allocated_bytes']==many['unique_allocated_bytes']+extra['unique_allocated_bytes'] and len(combined['buffers'])==len(many['buffers'])+len(extra['buffers'])
            for bad_graphs,bad_sessions in [(descriptors,None),([dict(object=session,tag='wrong')],sessions),(descriptors,[dict(object=graph,tag='wrong')]),(descriptors,[None])]:
                rejected=False
                try:garnet.memory_inventory_many_json(bad_graphs,bad_sessions)
                except Exception:rejected=True
                assert rejected, 'Malformed inventory collection accepted'
            assert before==snapshot(session) and other_before==snapshot(other) and before_hidden==session.hidden().tolist() and host_before==[v.tolist()for v in control], 'Inventory mutated native or host values'
            print('Global inventory old-API parity/duplicate graph and session dedup/two-private-session union/invalid types/read-only states PASS',flush=True)
'''
s=s[:idx]+block+s[idx:];(w/'resident_guard.py').write_text(s)
(w/'resident_controller.py').write_text((old/'resident_controller.py').read_text().replace('worldsense-cached-matrix-metadata','worldsense-physical-memory'))
s=(old/'correctness.py').read_text().replace('worldsense-cached-matrix-metadata','worldsense-physical-memory');a=s.index('try:\n');b=s.index('\nfinally:',a)
s=s[:a]+'''try:
 results={}
 for mode in ['off','on']:
  run(['/opt/homebrew/bin/ctest','--test-dir','/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-physical-memory-'+mode,'--output-on-failure'],'native-'+mode+'.log');results[mode]=0
 (w/'native-results.json').write_text(json.dumps(results,indent=2))
 run(['python3',str(w/'resident_controller.py')],'inventory-correctness.log')
 (w/'correctness-result.json').write_text(json.dumps({'both_native_references_pass':True,'trained_both_fixtures_off_on_global_inventory_read_only_dedup_and_types':True,'all_execution_sources_unchanged_from_WS34':True,'no_timing_claim':True},indent=2));print('BOTH GLOBAL INVENTORY API AND TRAINED STATE CHECKS PASS',flush=True)
'''+s[b:];(w/'correctness.py').write_text(s)
for p in w.glob('*.py'):ast.parse(p.read_text())
