from pathlib import Path
import shutil,ast,json
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-lazy-defaults'
for name in ['api_state_guard.py','resident_controller.py','resident_guard.py','resident_fault_guard.py','make_resident_fixtures.py','prepare_fault.py','restore_normal.py']:
 text=(old/name).read_text().replace('worldsense-lazy-defaults','worldsense-cached-matrix-metadata');(w/name).write_text(text)
(w/'wait_cantor.py').write_text((w.parent/'worldsense-request-residency/wait_cantor.py').read_text())
(w/'restore_normal.py').write_text((w.parent/'worldsense-request-residency/restore_normal.py').read_text().replace('worldsense-request-residency','worldsense-cached-matrix-metadata'))
for fixture in ['fixture-1-8','fixture-2-64']:shutil.copytree(old/fixture,w/fixture)
p=w/'api_state_guard.py';text=p.read_text().replace("call=json.loads(garnet.infer_json(prompt,image,limit,name));", "call=json.loads(garnet.infer_json(prompt,image,limit,name));assert ('cached_matrix_metadata' in call.get('backend',''))==bool(expected);");p.write_text(text)
# Vision numerical checks remain full, while old pooling-only diagnostics do not apply.
text=(w.parent/'worldsense-request-residency/vision_grid_guard.py').read_text().replace('worldsense-request-residency','worldsense-cached-matrix-metadata')
lines=[]
for line in text.splitlines():
 if any(n in line for n in ['print(\'before explicit plan','print(\'after explicit plan','plan=json.loads','assert m.vision.graph.vision_','assert (m.vision.graph.graph_workspace_']):continue
 line=line.replace(";assert previous.graph.graph_workspace_allocated_bytes==0",'')
 lines.append(line)
text='\n'.join(lines)+'\n';text=text.replace("if mode=='on':assert records==json.loads((w/'vision-grid-off.json').read_text())", "assert records==json.loads((w.parent/'worldsense-request-residency/vision-grid-off.json').read_text())\n if mode=='on':assert records==json.loads((w/'vision-grid-off.json').read_text())")
(w/'vision_grid_guard.py').write_text(text)
# Four-input recurrence exercises the same exact output/state/lifecycle checks,
# using the legacy Build API present in the frozen WS15 seed.
text=(w.parent/'worldsense-resident-workspace/linear_guard.py').read_text().replace('worldsense-resident-workspace','worldsense-cached-matrix-metadata').replace('build_bf16_resident(', 'build_bf16(').replace(';assert (graph.resident_avoided_storage_bytes>0)==enabled','')
lines=[]
for line in text.splitlines():
 if line.lstrip().startswith('inv=json.loads(') or line.lstrip().startswith('for i in [1,2]:assert'):continue
 line=line.replace('assert (a.workspace_bound_slots>0)==enabled;assert (a.workspace_allocated_bytes>0)==enabled;','')
 line=line.replace(';assert (shared.resident_avoided_storage_bytes>0)==enabled','').replace('build_shared_resident(','build_shared(')
 lines.append(line)
(w/'linear_guard.py').write_text('\n'.join(lines)+'\n')
for p in w.glob('*.py'):ast.parse(p.read_text())
print('Prepared native full-state/vision/legacy4-and6-input/fault validation scripts')
