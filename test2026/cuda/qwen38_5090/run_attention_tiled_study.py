"""Correctness first, then a separate warm attention-core CUDA-event A/B experiment."""
import argparse,hashlib,json,math,shutil,subprocess,time
from pathlib import Path
from run_logged import run,source_identity
from sample_vram import Sampler

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',required=True,type=Path);a=p.parse_args();root=a.study;out=root/'results/attention-compact-tiled';out.mkdir(exist_ok=True)
 manifest=out/'manifest.json'
 if manifest.exists():raise ValueError('Preserve existing component experiment')
 exe=root/'build/bin/garnet_attention_compact_test.exe';prefix=out/'qualification-context';command=[str(exe),str(prefix)]
 files=[exe,Path(__file__),root/'Garnet/src/cuda/qwen35_attention_core.cu',root/'Garnet/src/cuda/qwen35_attention_core.cuh',Path(__file__).with_name('attention_compact_test.cu'),root/'build/CMakeCache.txt']
 report={'status':'running','source':source_identity(),'file_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
  'scope':'Warm standalone FP32 attention core, including KV copies. Not model TTFT/decode/concurrency/VRAM or2x evidence.',
  'measurement':'CUDA events around20 graph launches/sample, six samples/variant/profile, ABCCBA repeated3 times; five warmups/variant. CPU/reference checks and downloads outside event intervals.',
  'environment_limit':'Desktop applications retain GPU memory; isolated component experiment must be repeated in controlled full-model runs.'}
 manifest.write_text(json.dumps(report,indent=2));shutil.copyfile(root/'build/CMakeCache.txt',out/'CMakeCache.txt')
 with (out/'nvidia-smi-before.xml').open('wb') as f:subprocess.run(['nvidia-smi','-q','-x'],stdout=f,check=True)
 code=run(command,root/'results/attention-compact-tiled-correctness-ledger','correctness','compact-independent-cpu-and-full-cache-graph-gates',[str(exe)])
 report['correctness_exit']=code
 if code:report['status']='correctness_failed';manifest.write_text(json.dumps(report,indent=2));return code
 sampler=Sampler(out/'vram.jsonl');sampler.phase='whole-component-oracle-and-timing-job';sampler.start();rows=[];code=1
 try:
  report['gpu']=sampler.device
  with (out/'timing-raw.log').open('wb') as f:
   process=subprocess.Popen([str(exe),str(out/'timing-context'),'--timing'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   for line in process.stdout:
    f.write(line);f.flush()
    if line.startswith(b'{'):
     row=json.loads(line);rows.append(row);print(json.dumps(row),flush=True)
   code=process.wait()
  report['timing_exit']=code;report['rows']=rows
  if code or len(rows)!=7 or not all(r['order']=='ABCCBAx3' and r['bit_exact_baseline'] and r['full_kv_exact'] and len(r['baseline_ms'])==len(r['compact_ms'])==len(r['tiled_ms'])==6 and all(math.isfinite(v) and v>0 for key in ('baseline_ms','compact_ms','tiled_ms') for v in r[key]) for r in rows):raise ValueError('Incomplete/mismatched component measurement')
  report['status']='component_completed'
 except Exception as error:
  report['status']='failed';report['error']=repr(error);code=1
 finally:
  sampler.close();report['whole_component_sampled_memory']=sampler.peak('whole-component-oracle-and-timing-job');report['sampling_errors']=sampler.errors
  report['source_after']=source_identity();manifest.write_text(json.dumps(report,indent=2))
 return code
if __name__=='__main__':raise SystemExit('Three-variant experiment archived. Current binary requires run_attention_split_study.py; reproduce the earlier experiment from source39a393c8a.')
