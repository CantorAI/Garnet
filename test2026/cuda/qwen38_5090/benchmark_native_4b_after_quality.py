from pathlib import Path
import hashlib,json,subprocess,sys
from urllib.request import urlopen
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';out=r/'results/native-4b-http-pilot'
# Never time while the CPU/IO-heavy tool extraction is still running.
for name in ('nsight-2026-tool-setup-direct-ledger','nsight-2026-cab-extraction-ledger','native-4b-http-quality-ledger'):
 events=[json.loads(l) for l in (r/'results'/name/'commands.jsonl').read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished','Observe original setup/quality terminal handles before timing'
 if name=='native-4b-http-quality-ledger':assert events[-1]['exit_code']==0
q=json.loads((out/'quality.json').read_text(encoding='utf8'));assert q['status']=='passed'
assert len(q['old_smoke_cases'])==4 and all(v['exact'] for v in q['old_smoke_cases'])
assert len(q['semantic_cases'])==4 and all(v['semantic_pass'] for v in q['semantic_cases'])
assert len(q['full128_pairs'])==16 and all(v['repeat_exact'] and v['generated']==128 for v in q['full128_pairs']) and set(q['stream_gates'])=={'text','image'}
metadata=out/'metadata.json';m=json.loads(metadata.read_text(encoding='utf8'))
for name,h in m['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'Qualified artifact changed: '+name
with urlopen('http://127.0.0.1:18082/health',timeout=5) as response:health=json.load(response)
assert health==m['health'],'Original4B worker profile changed';assert not (out/'benchmark').exists(),'Preserve prior timing attempt'
subprocess.run([sys.executable,str(p/'benchmark_stream.py'),'--url','http://127.0.0.1:18082','--engine','garnet','--model','qwen35-4b-bf16','--output',str(out/'benchmark'),'--metadata',str(metadata),'--concurrency','1','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)
