from pathlib import Path
import ast,shutil
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-native-text-worker'
for name in ['text_model_guard.py','text_guard_controller.py','invalid_image_loaded_guard.py']:
 t=(old/name).read_text().replace('worldsense-native-text-worker','worldsense-validated-text-worker')
 if name=='text_guard_controller.py':t=t.replace("['all_three_parent_roles_pass']","['all_three_integrated_parent_roles_pass']")
 ast.parse(t);(w/name).write_text(t)
p=w/'text_model_guard.py';t=p.read_text();needle="def infer(label,prompt):return capture(label,lambda:json.loads(garnet.infer_json(prompt,'',8,name)))";assert t.count(needle)==1
t=t.replace(needle,"from validated_call_guard import validated_call\ndef infer(label,prompt):\n value=capture(label,lambda:json.loads(garnet.infer_json(prompt,'',8,name)))\n if value.get('status')=='ok':validated_call(value,mode)\n return value")
ast.parse(t);p.write_text(t)
p=w/'invalid_image_loaded_guard.py';t=p.read_text().replace('import garnet\n','import garnet\nfrom validated_call_guard import validated_call\n');needle="row=dict(kind=kind,call=call);";assert t.count(needle)==1
t=t.replace(needle,"row=dict(kind=kind,call=call);")
needle="assert call['status']=='ok';state=";assert t.count(needle)==1;t=t.replace(needle,"assert call['status']=='ok';validated_call(call,'fault');state=")
ast.parse(t);p.write_text(t)
shutil.copy2(old/'model-assets.json',w/'model-assets.json')
print('Original text state/cache/capacity/reload and six loaded typed-image guards plus explicit validated provenance prepared')
