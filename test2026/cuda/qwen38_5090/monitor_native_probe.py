"""Record adapter memory around an untimed native correctness subprocess."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 command=a.command[1:] if a.command[:1]==['--'] else a.command
 if not command:p.error('Correctness subprocess required')
 a.output.mkdir(exist_ok=True);path=a.output/'resource-monitor.json'
 if path.exists():raise FileExistsError('Preserve previous resource observation')
 sys.path.insert(0,str(a.study/'Garnet/test2026/cuda/qwen38_5090'))
 from sample_vram import Sampler
 sampler=Sampler(a.output/'resource-vram.jsonl');sampler.phase='construction-and-correctness';code=None
 try:
  sampler.start();code=subprocess.run(command).returncode
 finally:
  sampler.close()
  report={'child_exit_code':code,'command':command,'gpu':sampler.device,'sampled_memory':sampler.peak('construction-and-correctness'),
   'scope':'Adapter-wide memory throughout construction and correctness, including background allocations. Not serving peak, TTFT or model throughput.'}
  path.write_text(json.dumps(report,indent=2));print('NATIVE_CORRECTNESS_RESOURCE_REPORT_SAVED',flush=True)
 return code

if __name__=='__main__':raise SystemExit(main())
