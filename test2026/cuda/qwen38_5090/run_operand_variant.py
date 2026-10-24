from pathlib import Path
import hashlib,subprocess,os,json,sys
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/operand-variant-attempt1';out.mkdir(exist_ok=False)
p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dll=r/'build/bin/garnet.dll';before=sha(dll);assert before=='2d902723b85f4353db3c3feccf2088b2ab412af6deb23b529caa3bb089d7708c'
code=run(['cmd.exe','/c',str(r/'build-operand-swap.bat')],out/'build-ledger','build','standalone-operand-swap',[r/'build-operand-swap.bat',r/'operand-variant-source.zip'])
assert sha(dll)==before,'Production DLL changed'
if code:raise SystemExit(code)
os.environ['PATH']=str(r/'tools/cuda13/nvidia/cu13/bin')+';'+str(r/'tools/cuda13/nvidia/cu13/bin/x64')+';'+str(r/'tools/cuda13/nvidia/cu13/bin/x86_64')+';'+os.environ['PATH']
exe=r/'build/bin/garnet_nvfp4_operand_swap_test.exe'
identity={'dll_before_sha256':before,'dll_after_build_sha256':sha(dll),'exe_sha256':sha(exe),'source':json.loads((r/'study-state.json').read_text())}
(out/'identity.json').write_text(json.dumps(identity,indent=2))
for name,args in [('oracle',[]),('algorithms',['--algorithm-timing'])]:
 cmd=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/(name+'-resources')),'--',str(exe),str(out/name/'report.json'),*args]
 code=run(cmd,out/(name+'-ledger'),'correctness',name,[exe,dll])
 if code:raise SystemExit(code)
assert sha(dll)==before
(out/'workflow-passed.json').write_text(json.dumps({'status':'passed','dll_final_sha256':sha(dll)}))
