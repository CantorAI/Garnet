from pathlib import Path
import json,statistics,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-weight-residency/speed');rows=[];summary={};identities={}
for label in ['control-1','candidate-1']:
 p=r/(label+'.json');identities[label]=hashlib.sha256(p.read_bytes()).hexdigest();d=json.loads(p.read_text())
 for item in d['items']:
  for phase,c in item['result']['model_response'].items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   t=c['timings'];rows.append({'worker':label,'model':c['model_id'],'warmup':item['warmup'],'repetition':item['repetition'],'phase':phase,'tokens':c['generated_tokens'],'decode_seconds':t['decode_seconds'],'decode_profile':t['decode_profile'],'prefill_profile':t['prefill_profile'],'residency':t['weight_residency']})
 summary[label]={}
 for model in ['Qwen3.5-4B','Qwen3-1.7B']:
  rr=[x for x in rows if x['worker']==label and x['model']==model and not x['warmup']];assert len(rr)==4
  summary[label][model]={'samples':4,'decode_seconds_p50':statistics.median(x['decode_seconds']for x in rr),'decode_profile_p50':{k:statistics.median(x['decode_profile'][k]for x in rr)for k in ['encode_ms','prepare_ms','gpu_ms','wait_ms']},'prefill_profile_p50':{k:statistics.median(x['prefill_profile'][k]for x in rr)for k in ['encode_ms','prepare_ms','gpu_ms','wait_ms']}}
(w/'prior-phase-analysis.json').write_text(json.dumps({'raw_sha256':identities,'all_first_warm_rows':rows,'warm_p50':summary,'scope':'Read-onlyposthocWS32diagnostic,no newinference/confirmation. GPUandwaitoverlap;neveradd. Profilevaluesreportednativebatchspans,sumoversequentialsteps,notperoperatorGPUactivecycles. No stepcountinferencefromgeneratedtokens/EOS. Wholeprofilecomparisonnotisolatedresidencycausality.'},indent=2));print(json.dumps(summary,indent=2))
