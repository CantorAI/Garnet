from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
out=r/'results/layer0-real-stage-attempt2';assert not out.exists();events=[json.loads(l) for l in (r/'results/layer0-stage-build-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
build=r/'results/layer0-stage-build/manifest.json';v=json.loads(build.read_text());assert v['status']=='native_build_completed_unqualified' and v['library_sha256']==hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
fix=r/'results/layer0-stage-probe-fix-install/receipt.json';f=json.loads(fix.read_text());assert f['status']=='test_setup_fixed_same_native_binary' and f['library_sha256']==v['library_sha256']
for n,item in v['source_files'].items():
 if not n.endswith('probe_layer0_real_stages.py'):assert hashlib.sha256((r/n).read_bytes()).hexdigest()==item['sha256']
assert hashlib.sha256((p/'probe_layer0_real_stages.py').read_bytes()).hexdigest()==f['probe_sha256']
witness=r/'results/layer0-stage-witness/manifest.json';saved=json.loads(witness.read_text());assert saved['archive_sha256']=='32586aea53c0fd77b1d96935f700abe0061bd24f2cc1915400454929aac10564'
for n,h in saved['files'].items():assert hashlib.sha256((witness.parent/n).read_bytes()).hexdigest()==h
assert hashlib.sha256((r/'results/layer0-real-stage-attempt1-evidence.zip').read_bytes()).hexdigest()==f['failure_archive_sha256']
out.mkdir()
code=run([sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(r/'test-layer0-real-stages-warmed.bat')],out/'probe-ledger','correctness','actual-layer0-b1-b2-stage-primary-witness',[r/'test-layer0-real-stages-warmed.bat',p/'probe_layer0_real_stages.py',r/'build/bin/garnet.dll',build,witness,fix])
report={'status':'process_failed' if code else 'process_finished_needs_independent_array_replay','exit_code':code,'library_sha256':v['library_sha256'],'scope':'Actual layer0 intermediate witness/negative/repeat/row diagnostics only; no full64-layer batch admission, serving or timing claim.'};(out/'workflow.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));raise SystemExit(code)
