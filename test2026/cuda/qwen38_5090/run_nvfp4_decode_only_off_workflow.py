from pathlib import Path
import json,sys,time,zipfile
from urllib.request import urlopen
r=Path('D:/CantorAI/5090-study-20261023');sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from run_logged import run
from validate_native_mtp_http import validate_worker_profile
out=r/'results/native-mtp3-decode-only-off-workflow';out.mkdir(exist_ok=False);d=r/'results/shared-decode-only-off-http-diagnostic';assert not d.exists();deadline=time.monotonic()+1800
while True:
 try:
  with urlopen('http://127.0.0.1:18082/health',timeout=5) as response:health=json.load(response)
 except Exception:
  if time.monotonic()>=deadline:raise TimeoutError('Inspect original server handle; never restart solely on deadline')
  time.sleep(5);continue
 validate_worker_profile(health,3,4,'direct',4,128,True,False,True);break
(out/'health.json').write_text(json.dumps(health,indent=2),encoding='utf8')
p=r/'Garnet/test2026/cuda/qwen38_5090'
code=run([sys.executable,str(p/'probe_shared_mtp_sequences.py'),str(r),str(d),'--joined-mlp'],out/'diagnostic-ledger','correctness','v3-option-off-twenty-pairs',[r/'build/bin/garnet.dll',r/'serve-nvfp4-decode-only-off-worker.bat'])
if code:raise SystemExit(code)
rows=[]
with zipfile.ZipFile(r/'results/native-mtp3-shared-joined-evidence.zip') as old:
 for f in sorted(d.glob('*-target-response.json')):
  key=f.name.removesuffix('-target-response.json');expected=json.loads(old.read('results/shared-joined-mlp-http-diagnostic/'+key+'-target-response.json'))['choices'][0]
  for mode in ('target','mtp'):
   current=json.loads((d/(key+'-'+mode+'-response.json')).read_text(encoding='utf8'))['choices'][0];a=expected['response_token_ids'];b=current['response_token_ids']
   rows.append({'key':key,'mode':mode,'prompt_exact':expected['prompt_token_ids']==current['prompt_token_ids'],'generated_exact':a==b,'first_different_index':next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None),'old_length':len(a),'new_length':len(b)})
assert len(rows)==40
report={'status':'off_control_exact' if all(v['prompt_exact'] and v['generated_exact'] for v in rows) else 'off_control_differs','cases':rows,'scope':'Same rebuilt DLL option-off20 HTTP target/MTP pairs versus prior qualified outputs. Untimed control, not a benchmark; no cause or harmlessness assumed.'}
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if report['status']!='off_control_exact':raise SystemExit(1)
