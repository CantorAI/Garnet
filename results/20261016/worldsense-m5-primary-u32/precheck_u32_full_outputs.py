"""Read-only output precheck for completed workers; no timing decision."""
from pathlib import Path
import ast,hashlib,json,sys
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-u32/confirmation')
g=json.loads(Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text())
fn=next(n for n in ast.parse((w/'analyze.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_route')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native_route','exec'),ns)
def exact(c,ref,enabled):
    assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
    assert c['timings']['input_tokens']==ref['timings']['input_tokens']
    assert ns['native_route'](c,enabled)
stable=lambda r:{k:v for k,v in r.items()if k not in ['model_response','model_response_json']}
for label in sys.argv[1:]:
    p=w/(label+'.json');d=json.loads(p.read_text());enabled=label.startswith('candidate')
    assert d['valid']and d['assignments_restored']and len(d['items'])==6 and len(d['search_plans'])==2
    count=0
    for item in d['items']:
        ref=next(r for r in g['items']if r['policy']=='candidate'and r['observation_id']==item['observation_id']and r['skill_ref']==item['skill_ref'])
        assert stable(item['result'])==stable(ref['result'])and item['image_sha256']==ref['image_sha256']and item['skill_sha256']==ref['skill_sha256']
        for phase,c in item['result']['model_response'].items():
            if isinstance(c,dict)and c.get('status')=='ok':exact(c,ref['result']['model_response'][phase],enabled);count+=1
    for query in d['search_plans']:
        ref=next(q for q in g['search_plans']if q['policy']=='candidate'and q['query']==query['query'])
        exact(query['result']['benchmark_model_response'],ref['result']['benchmark_model_response'],enabled)
        assert {k:v for k,v in query['result'].items()if k!='benchmark_model_response'}=={k:v for k,v in ref['result'].items()if k!='benchmark_model_response'}
        count+=1
    assert count==12
    (w/(label+'-output-precheck.json')).write_text(json.dumps({'raw_artifact':p.name,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'all12_model_calls_exact_to_golden':True,'native_route_valid':True,'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Completed-worker output precheck only; no paired timing or post-stage SDK decision.'},indent=2))
    print(label,'all12calls exact/golden/semantics/input/native route',flush=True)
