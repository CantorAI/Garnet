from pathlib import Path
import json,sys
r=Path('D:/CantorAI/5090-study-20261023');sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from validate_native_mtp_http import post
out=r/'results/native-mtp3-decode-only-workflow/live-repeat';out.mkdir(exist_ok=False);d=r/'results/shared-decode-only-http-diagnostic'
rows=json.loads((out.parent/'prior-id-diagnostic.json').read_text(encoding='utf8'));keys=sorted({v['key'] for v in rows if not v['generated_exact']});assert len(keys)==5
report={'scope':'Untimed same-live-worker repeat of five prior-qualified identity failures; no gate waiver or benchmark.','cases':[],'status':'checking'}
try:
 for k in keys:
  for mode in ('target','mtp'):
   body=json.loads((d/(k+'-'+mode+'-request.json')).read_text(encoding='utf8'));original=json.loads((d/(k+'-'+mode+'-response.json')).read_text(encoding='utf8'))['choices'][0]
   current=post('http://127.0.0.1:18082',body,out,k+'-'+mode)['choices'][0]
   item={'key':k,'mode':mode,'prompt_exact':current['prompt_token_ids']==original['prompt_token_ids'],'generated_exact':current['response_token_ids']==original['response_token_ids']};report['cases'].append(item)
   (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');assert item['prompt_exact'] and item['generated_exact'],'Live repeat changed; preserve raw response'
 report['status']='repeat_exact'
finally:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
