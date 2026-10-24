from pathlib import Path
import hashlib,json,os,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
out=r/'results/nvfp4-decode-only-attempt1';out.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dll=r/'build/bin/garnet.dll';old=sha(dll);assert old=='e82279af698dd7798be3f158ace999eef29be4ed609e8cd50cf1ae36c90cd750'
fixture=r/'results/nvfp4-down-candidate-attempt1/fixture';manifest=json.loads((fixture/'manifest.json').read_text(encoding='utf8'))
for name,h in manifest['files'].items():assert sha(fixture/name)==h
code=run(['cmd.exe','/c',str(r/'build-nvfp4-down-runtime-tests.bat')],out/'build-tests-ledger','build','down-runtime-tests',[r/'build-nvfp4-down-runtime-tests.bat',r/'nvfp4-decode-only-source.zip'])
assert sha(dll)==old
if code:raise SystemExit(code)
os.environ['PATH']=str(r/'tools/cuda13/nvidia/cu13/bin')+';'+str(r/'tools/cuda13/nvidia/cu13/bin/x64')+';'+str(r/'tools/cuda13/nvidia/cu13/bin/x86_64')+';'+str(r/'tools/TensorRT/bin')+';'+os.environ['PATH']
plugin=r/'build/bin/garnet_modelopt_plugin_test.exe';probe=r/'build/bin/garnet_nvfp4_operand_swap_test.exe'
identity={'dll_before_sha256':old,'plugin_test_sha256':sha(plugin),'probe_sha256':sha(probe),'source':json.loads((r/'study-state.json').read_text(encoding='utf8'))};(out/'identity.json').write_text(json.dumps(identity,indent=2))
stages=[('contracts',[str(plugin),'--candidate-contracts']),('legacy-plugin',[str(plugin)]),('oracle',[str(probe),str(out/'oracle/report.json')]),('checkpoint',[str(probe),str(out/'checkpoint/report.json'),'--checkpoint-fixture',str(fixture)])]
for name,args in stages:
 command=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/(name+'-resources')),'--',*args]
 code=run(command,out/(name+'-ledger'),'correctness',name,[plugin,probe,dll,fixture/'manifest.json'])
 if code:raise SystemExit(code)
assert sha(dll)==old
for name,h in manifest['files'].items():assert sha(fixture/name)==h
code=run(['cmd.exe','/c',str(r/'build-nvfp4-down-runtime-native.bat')],out/'build-native-ledger','build','down-runtime-native',[r/'build-nvfp4-down-runtime-native.bat',r/'nvfp4-decode-only-source.zip'])
if code:raise SystemExit(code)
identity['dll_after_native_build_sha256']=sha(dll);assert identity['dll_after_native_build_sha256']!=old
(out/'identity.json').write_text(json.dumps(identity,indent=2))
(out/'workflow-passed.json').write_text(json.dumps({'status':'component_gates_and_native_build_passed','library_sha256':sha(dll),'full_model_qualification':'pending','scope':'Opt-in/default-off runtime policy; no full-model correctness/performance or2x claim.'}))
