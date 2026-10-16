"""Native ABBA: validated split-key chunk 32 against selected chunk 16."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
tree=ast.parse((scripts/'compare_vllm_metal.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=functions,type_ignores=[]),'owned_guard','exec'),globals())
assert os.environ.get('GARNET_METAL_PROFILE_OPS')!='1'
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
candidate=root/'build-dense-single-prefill';control=root/'build-linear-chunk'
def digest(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
metadata=dict(scope='Native ABBA; no vLLM/2x claim',candidate_sha256=digest(candidate/'libgarnet.dylib'),control_sha256=digest(control/'libgarnet.dylib'),worker_sha256=digest(scripts/'run_matched_garnet.py'),runner_sha256=digest(Path(__file__)))
assert metadata['control_sha256']=='26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671'
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
baseline=root/'logs/equivalent-f32-linear-chunk-text-20261013/jobs';records=[];groups=[];statuses=[]
for batch in (1,2):
 for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
  label=name+'-b'+str(batch);case=json.loads((baseline/(label+'-case.json')).read_text())
  frozen=next(json.loads(v.split(' ',1)[1]) for v in (baseline/(label+'-garnet-0/output.log')).read_text().splitlines() if v.startswith('matched_result '))
  for repeat in range(2):
   for backend in (('candidate','control') if repeat==0 else ('control','candidate')):
    folder=output/(label+'-'+backend+'-'+str(repeat));folder.mkdir();work=dict(case);work['chunk']=32 if backend=='candidate' else 16
    casefile=folder/'case.json';casefile.write_text(json.dumps(work,indent=2)+'\n')
    guard();(folder/'before-processes.txt').write_text(snapshot())
    command=[sys.executable,str(scripts/'benchmark_matched_process.py'),runtime,str(scripts/'run_matched_garnet.py'),str(candidate if backend=='candidate' else control),str(root/'source'),str(root/'models'/name),str(casefile)]
    with (folder/'output.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    statuses.append(dict(folder=folder.name,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    assert result.returncode==0,folder
    guard();(folder/'after-processes.txt').write_text(snapshot())
    lines=(folder/'output.log').read_text().splitlines()
    values=[json.loads(v.split(' ',1)[1]) for v in lines if v.startswith('matched_result ')];assert len(values)==3
    memory=next(json.loads(v.split(' ',1)[1]) for v in lines if v.startswith('matched_process '));assert memory['exit_code']==0
    for v in values:
     for key in ('token_ids','finished','prompt_token_ids'):assert v[key]==frozen[key],(folder,key)
     assert all(v['finished']) and v['decode_steps_after_first']==max(map(len,v['token_ids']))-1
     assert v['prefill_chunk_plan']==([32] if backend=='candidate' else [16,16])
     v.update(backend=backend,repeat=repeat,profile_model=name,process=memory);records.append(v)
  selected=[v for v in records if v['profile_model']==name and v['case']['batch']==batch]
  def mean(backend):
   a=[v for v in selected if v['backend']==backend and v['request']>0]
   return {k:statistics.mean(v[k] for v in a) for k in ('ttft_seconds','decode_wall_steps_per_second','total_seconds')}
  c=mean('candidate');b=mean('control')
  group=dict(model=name,batch=batch,candidate=c,control=b,ttft_fraction=c['ttft_seconds']/b['ttft_seconds'],decode_ratio=c['decode_wall_steps_per_second']/b['decode_wall_steps_per_second'],exact_complete_outputs=True,
   first_use={p:statistics.mean(v['ttft_seconds'] for v in selected if v['backend']==p and v['request']==0) for p in ('candidate','control')},
   initialization={p:statistics.mean(v['initialization_seconds'] for v in selected if v['backend']==p and v['request']==0) for p in ('candidate','control')},
   peak_child_rss_bytes={p:max(v['process']['peak_rss_bytes'] for v in selected if v['backend']==p) for p in ('candidate','control')})
  groups.append(group);(output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],groups=groups,records=records),indent=2)+'\n');print(json.dumps(group),flush=True)
