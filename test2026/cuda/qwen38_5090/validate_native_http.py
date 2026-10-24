"""Gate native HTTP transport against retained native VLM IDs, then exploratory B1 timing."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from run_logged import source_identity
from benchmark_stream import payload,request
from native_http_gates import validate_saved_vlm

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();root=a.study
 out=a.output or root/'results/native-http';out.mkdir(exist_ok=True);url='http://127.0.0.1:18082';deadline=time.monotonic()+1200
 while True:
  try:
   with urlopen(url+'/health',timeout=5) as response:
    if response.status==200:health=response.read();break
  except Exception as error:
   if time.monotonic()>=deadline:raise TimeoutError('Native readiness deadline; inspect live server before restart') from error
  time.sleep(5)
 (out/'health.json').write_bytes(health)
 rows=validate_saved_vlm(root,out,url)
 for case,prompt_count in [('text',41),('image',104)]:
  body,_=payload(case,'s3-'+case+'-c1-r0',300000+(20000 if case=='image' else 0)+1000,'qwen38-27b-nvfp4',128,'jpeg','image-first')
  result=request(url,body,out,'stream-gate-'+case,128)
  if result['status']!='completed' or result['prompt_tokens']!=prompt_count or not result['single_token_arrivals']:raise ValueError('Native streaming gate failed: '+str(result))
  print('NATIVE_STREAM_GATE_PASS',case,flush=True)
 files=[root/'build/bin/garnet.dll',root/'build/bin/xlang3.exe',root/'serve-native-worker.bat',root/'models/Qwen3.8-27B-NVFP4/config.json']
 metadata={'source':source_identity(),'health':json.loads(health),'precision':'mixed NVFP4/FP8, native W4A4 head, BF16 vision; F32 state/cache','context_capacity':4096,'batch':1,
  'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'correctness':rows,'scope':'Research-only native single-slot HTTP bridge; broader quality and exact reference VLM parity remain open'}
 path=out/'metadata.json';path.write_text(json.dumps(metadata,indent=2))
 subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark_stream.py')),'--url',url,'--engine','garnet','--output',str(out/'benchmark'),'--metadata',str(path),'--concurrency','1','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first'],check=True)
if __name__=='__main__':main()
