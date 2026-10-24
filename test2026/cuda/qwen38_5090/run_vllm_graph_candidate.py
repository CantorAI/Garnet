"""Bounded vLLM graph readiness, actual JPEG token-ID smoke, then raw HTTP pilot.

A completed run is still a candidate until runtime graph/admission/quality review.
"""
import argparse,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True)
 p.add_argument('--output',type=Path);p.add_argument('--metadata',type=Path)
 a=p.parse_args();root=a.study;scripts=Path(__file__).parent
 out=a.output or root/'results/vllm-graph-jpeg-grid'
 if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve prior attempt; choose a fresh explicit output path')
 out.mkdir(parents=True,exist_ok=True);metadata=a.metadata or root/'results/vllm-graph-metadata.json';url='http://127.0.0.1:18080';deadline=time.monotonic()+1200
 while True:
  try:
   with urlopen(url+'/health',timeout=5) as response:
    if response.status==200:health=response.read();break
  except Exception as error:
   if time.monotonic()>=deadline:raise TimeoutError('vLLM readiness deadline; inspect live server before any restart') from error
  time.sleep(5)
 (out/'health-response.bin').write_bytes(health);print('VLLM_HEALTHY',flush=True)
 subprocess.run([sys.executable,str(scripts/'prepare_vlm_reference.py'),'--engine','vllm','--url',url,'--output',str(out/'correctness')],check=True)
 print('VLLM_FOUR_JPEG_TOKEN_SMOKES_PASSED',flush=True)
 subprocess.run([sys.executable,str(scripts/'benchmark_stream.py'),'--url',url,'--engine','vllm','--output',str(out/'benchmark'),'--metadata',str(metadata),
  '--concurrency','1','2','4','8','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)
if __name__=='__main__':main()
