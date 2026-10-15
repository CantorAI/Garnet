"""CPU-only exact-input checks and ABBA fresh preparation; no inference claim."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from frame_preprocessing import prepare_frame as control
from frame_preprocessing_compact import prepare_frame as candidate
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);output=out
s=root/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_guard','exec'),globals())
def check():
 guard()
 for line in snapshot().splitlines():
  fields=line.split()
  if fields and fields[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+fields[0])
check()
command=[sys.executable,str(s/'test_compact_frame_preprocessing.py'),str(out/'compatibility')]
with (out/'compatibility.log').open('w') as stream:code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
(out/'compatibility-status.json').write_bytes((json.dumps(dict(command=command,exit_code=code))+'\n').encode());assert code==0
check();model=root/'models/Qwen3.5-0.8B';cfg=json.loads((model/'config.json').read_text())['vision_config'];processor=json.loads((model/'preprocessor_config.json').read_text());records=[];cases={};warmups=[]
for batch in (1,2):
 case=json.loads((root/'logs/global-alpha-vllm-20261014/jobs'/('Qwen3.5-0.8B-b'+str(batch)+'-0/case.json')).read_text());meta=case['image_metadata'];cases[str(batch)]=case
 image=Path(meta['image']);assert hashlib.sha256(image.read_bytes()).hexdigest()==meta['image_sha256']
 def run(policy):
  begin=time.perf_counter()
  with Image.open(image) as opened:patches,rgb,hw=(candidate if policy else control)(opened,cfg,processor,batch)
  patchfile=out/'frame.f32';rgbfile=out/'frame.rgb';patches.tofile(patchfile);rgbfile.write_bytes(rgb)
  patch_sha=hashlib.sha256(patchfile.read_bytes()).hexdigest();rgb_sha=hashlib.sha256(rgb).hexdigest();elapsed=(time.perf_counter()-begin)*1000
  assert patch_sha==meta['patch_sha256'] and list(patches.shape)==meta['shape'] and list(hw)==meta['resized_hw']
  return dict(batch=batch,candidate=policy,elapsed_ms=elapsed,patch_sha256=patch_sha,rgb_sha256=rgb_sha)
 begin=time.perf_counter();count=0
 while time.perf_counter()-begin<.25:check();run(False);run(True);count+=2
 warmups.append(dict(batch=batch,executions=count,seconds=time.perf_counter()-begin))
 for repeat in (0,1):
  for window,policy in enumerate((False,True,True,False)):
   check();run(policy)
   for sample in range(12):check();records.append(dict(repeat=repeat,window=window,sample=sample,**run(policy)))
assert len(records)==192 and len({r['rgb_sha256'] for r in records})==1
paths=[Path(__file__),s/'frame_preprocessing.py',s/'frame_preprocessing_compact.py',s/'test_compact_frame_preprocessing.py',s/'compare_vllm_metal.py']
snapshot_dir=out/'source-snapshot';snapshot_dir.mkdir()
for p in paths:(snapshot_dir/p.name).write_bytes(p.read_bytes())
summary=dict(scope=__doc__,numpy=np.__version__,pillow=Image.__version__,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},cases=cases,warmups=warmups,records=records,results=[])
for batch in (1,2):
 means={str(policy):statistics.mean(r['elapsed_ms'] for r in records if r['batch']==batch and r['candidate']==policy) for policy in (False,True)}
 summary['results'].append(dict(batch=batch,control_ms=means['False'],candidate_ms=means['True'],fraction=means['True']/means['False']))
(out/'summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode());check();print(json.dumps(summary['results']),flush=True)
