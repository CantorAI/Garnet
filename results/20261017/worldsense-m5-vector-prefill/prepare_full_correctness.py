from pathlib import Path
import shutil,ast,json
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-primary-u32'
for name in ['api_state_guard.py','linear_guard.py','vision_grid_guard.py','resident_guard.py','resident_controller.py','make_resident_fixtures.py']:
 t=(old/name).read_text().replace('worldsense-primary-u32','worldsense-vector-prefill')
 if name=='api_state_guard.py':
  a="mode=sys.argv[1];assert mode in ['off','on'];expected=1 if mode=='on'else 0";assert t.count(a)==1
  t=t.replace(a,"mode=sys.argv[1];assert mode in ['off','primary','on'];expected=0 if mode=='off'else 1;vector_expected=mode=='on'",1)
  a="call=json.loads(garnet.infer_json(prompt,image,limit,name));";assert t.count(a)==1;t=t.replace(a,a+"assert ('primary_vector_prefill' in call.get('backend',''))==vector_expected;",1)
  t=t.replace('primary_expected=expected,lazy_defaults_expected=False','primary_expected=expected,vector_prefill_expected=vector_expected,lazy_defaults_expected=False')
 if name=='resident_controller.py':
  a="for mode in ['off','on']:";assert t.count(a)==1;t=t.replace(a,"for mode in ['off','primary','on']:",1)
 ast.parse(t);(w/name).write_text(t)
for name in ['fixture-1-8','fixture-2-64']:
 shutil.copytree(old/name,w/name);p=w/name/'fixture.json';p.write_text(p.read_text().replace(str(old),str(w)))
(w/'model-guard-adaptation.json').write_text(json.dumps({'origin':'WS40/u32 original guards','changes':['paths only, actual prior-primary control added','primary enabled in primary/on; vector-prefill capability onlyon','original numeric/state/token/prefix/capacity assertions unchanged','residentfixtures run OFF/priorprimary/ON; linear originalOFF/ON/fault;visionOFF/ON'],'disabled_lazy_fault_branch_unclaimed':True},indent=2)+'\n')
print('Full OFF/priorprimary/ON actual model guard matrix ready; original bounds and golden bits unchanged.')
