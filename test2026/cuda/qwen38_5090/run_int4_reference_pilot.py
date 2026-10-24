from pathlib import Path
import argparse,hashlib,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--engine',choices=['vllm','sglang'],required=True);a=p.parse_args();r=a.study;source=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(source))
from run_logged import source_identity
out=r/('results/'+a.engine+'-int4-http-pilot');out.mkdir(exist_ok=False)
quality=r/('results/'+a.engine+'-int4-quality');candidate=json.loads((quality/'candidate.json').read_text());greedy=json.loads((quality/'greedy/equivalence.json').read_text());vlm=json.loads((quality/'correctness/oracle.json').read_text())
assert candidate['status']=='quality_capture_completed' and candidate['model']=='qwen38-27b-int4'
assert greedy['status']=='passed' and greedy['model']=='qwen38-27b-int4' and len(greedy['cases'])==16
assert all(len(c['generated_ids'])==128 for c in greedy['cases'])
assert vlm['model']=='qwen38-27b-int4' and len(vlm['cases'])==4
for c in vlm['cases']:
 assert c['reference_text'].strip().lower().rstrip('.')==c['expected_color'] and c['prompt_ids'].count(248056)==64
log=r/('results/'+a.engine+'-int4-server.log');raw=log.read_bytes();(out/'server-log-before-pilot.log').write_bytes(raw)
if a.engine=='vllm':assert b'quantization=compressed-tensors' in raw and b'Qwen3_5ForConditionalGeneration' in raw
files=[quality/'candidate.json',quality/'greedy/equivalence.json',quality/'correctness/oracle.json',source/('serve_'+a.engine+'_int4.sh'),r/'models/Qwen3.8-27B-INT4/config.json',r/'models/Qwen3.8-27B-INT4/preprocessor_config.json']
metadata={'source':source_identity(),'model':'qwen38-27b-int4','engine':a.engine,'quality_capture':candidate,'vlm_cases':4,'full128_greedy_cases':16,'server_log_before_sha256':hashlib.sha256(raw).hexdigest(),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'scope':'Exploratory pinned INT4 reference candidate after4 semantic VLM cases and16 full greedy captures. Startup snapshot retained; ordinary configuration, not fully tuned fastest proof. No broad/cross-engine quality, native concurrency or2x qualification.'}
path=out/'metadata.json';path.write_text(json.dumps(metadata,indent=2))
url='http://127.0.0.1:'+('18080' if a.engine=='vllm' else '18081')
subprocess.run([sys.executable,str(source/'benchmark_stream.py'),'--url',url,'--engine',a.engine,'--model','qwen38-27b-int4','--output',str(out/'benchmark'),'--metadata',str(path),'--concurrency','1','2','4','8','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)
(out/'completed.json').write_text(json.dumps({'status':'completed','scope':metadata['scope']}))
