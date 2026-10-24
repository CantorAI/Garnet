from pathlib import Path
import hashlib,json,os,shutil,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
out=r/'results/nvfp4-down-candidate-attempt1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dll=r/'build/bin/garnet.dll';digest=sha(dll);assert digest=='2d902723b85f4353db3c3feccf2088b2ab412af6deb23b529caa3bb089d7708c'
assert shutil.disk_usage(r).free>1<<30,'Require1GiB headroom for bounded fixture/build/evidence'
fixture=out/'fixture'
code=run([sys.executable,str(p/'prepare_nvfp4_down_fixture.py'),str(r/'models/Qwen3.8-27B-NVFP4'),str(fixture)],out/'fixture-ledger','correctness','actual-down-fixture',[p/'prepare_nvfp4_down_fixture.py',r/'models/Qwen3.8-27B-NVFP4/model.safetensors.index.json'])
if code:raise SystemExit(code)
manifest=json.loads((fixture/'manifest.json').read_text(encoding='utf8'))
for name,h in manifest['files'].items():assert sha(fixture/name)==h
code=run(['cmd.exe','/c',str(r/'build-operand-swap.bat')],out/'build-ledger','build','standalone-down-candidate',[r/'build-operand-swap.bat',r/'nvfp4-down-source.zip'])
assert sha(dll)==digest
if code:raise SystemExit(code)
os.environ['PATH']=str(r/'tools/cuda13/nvidia/cu13/bin')+';'+str(r/'tools/cuda13/nvidia/cu13/bin/x64')+';'+str(r/'tools/cuda13/nvidia/cu13/bin/x86_64')+';'+os.environ['PATH']
exe=r/'build/bin/garnet_nvfp4_operand_swap_test.exe'
(out/'identity.json').write_text(json.dumps({'dll_before_sha256':digest,'dll_after_build_sha256':sha(dll),'exe_sha256':sha(exe),'source':json.loads((r/'study-state.json').read_text(encoding='utf8'))},indent=2))
for name,args in [('oracle',[]),('checkpoint',['--checkpoint-fixture',str(fixture)])]:
 cmd=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/(name+'-resources')),'--',str(exe),str(out/name/'report.json'),*args]
 code=run(cmd,out/(name+'-ledger'),'correctness',name,[exe,dll,fixture/'manifest.json'])
 if code:raise SystemExit(code)
assert sha(dll)==digest
for name,h in manifest['files'].items():assert sha(fixture/name)==h
(out/'workflow-passed.json').write_text(json.dumps({'status':'passed','dll_final_sha256':sha(dll),'timing_admitted':False,'scope':'One projection correctness gate only; no timing performed.'}))
