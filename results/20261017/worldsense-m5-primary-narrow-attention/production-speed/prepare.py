from pathlib import Path
import json,shutil,hashlib,ast
w=Path(__file__).resolve().parent;p=w.parent;r=Path('/Users/shawn/CantorAI')
assert json.loads((p/'speed/analysis.json').read_text())['passed']
assert json.loads((p/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert not(w/'control-1.json').exists()and not(w/'candidate-1.json').exists()
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
control=r/'out/research/worldsense-prefix-cache/source'
manifest=json.loads((p.parent/'worldsense-primary-u32-integration/control-source-before-dispatch.json').read_text())
assert all(sha(control/rel)==h for rel,h in manifest.items())
(w/'control-source-before-dispatch.json').write_text(json.dumps(manifest,indent=2))
env=json.loads((p/'environment.json').read_text());env['libraries']['off']=sha(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib');env['control_source']=str(control);env['control_serving_sha256']=sha(control/'runtime/metal_serving.py')
(w/'environment.json').write_text(json.dumps(env,indent=2))
for name in ['starting-source-manifest.json','cases.json']:shutil.copy2(p/'speed'/name,w/name)
t=(p/'speed/controller.py').read_text().replace('worldsense-primary-narrow-attention/speed','worldsense-primary-narrow-attention/production-speed')
needle="library=root/('out/build/garnet-metal-worldsense-primary-narrow-attention-'+('on'if label.startswith('candidate')else'off'))/'libgarnet.dylib'"
assert t.count(needle)==1
t=t.replace(needle,"library=(root/'out/build/garnet-metal-worldsense-primary-narrow-attention-on/libgarnet.dylib')if label.startswith('candidate')else(root/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')",1)
needle="sha(root/'out/research/worldsense-primary-narrow-attention/source/runtime/metal_serving.py')"
assert t.count(needle)==1
t=t.replace(needle,"sha(root/('out/research/worldsense-primary-narrow-attention/source/runtime/metal_serving.py'if label.startswith('candidate')else'out/research/worldsense-prefix-cache/source/runtime/metal_serving.py'))",1)
ast.parse(t);(w/'controller.py').write_text(t)

# Quantitative/semantic/input checks remain the original screen AST. Only the
# explicitly declared actual-control identity/telemetry differences are allowed.
a=ast.parse((p/'speed/analyze.py').read_text());b=ast.parse((w/'analyze.py').read_text())
for variable in ['models','metrics','comparisons','wall','wall_gain']:
 def assignments(tree):return [n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id==variable for x in n.targets)]
 assert [ast.dump(n,include_attributes=False)for n in assignments(a)]==[ast.dump(n,include_attributes=False)for n in assignments(b)],variable
tree=ast.parse((w/'analyze.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(w/'analyze.py'),'exec'),namespace)
old=json.loads((p.parent/'worldsense-primary-u32-integration/production/control-full-1.json').read_text())
calls=[c for i in old['items']for c in i['result']['model_response'].values()if isinstance(c,dict)and'model_id'in c]+[q['result']['benchmark_model_response']for q in old['search_plans']]
assert len(calls)==12 and all(namespace['native_route'](c,False)for c in calls)
new=json.loads((p/'api-state-progress-on.json').read_text())['records'];assert len(new)==5
assert all(namespace['native_route'](i['call'],True)for i in new)
(w/'preflight.json').write_text(json.dumps(dict(actual_production_library_and_control_source_pinned=True,model_metrics_comparisons_and_wall_AST_unchanged=True,old_memory_unavailable_not_zero=True,actual_old_control_native_calls=12,actual_candidate_native_calls=5,worker_dispatched=False),indent=2))
