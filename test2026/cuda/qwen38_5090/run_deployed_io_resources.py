"""Read actual deployed CUDA resource metadata after the frozen ABBA sequence."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,zipfile
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
proof=json.loads((r/'results/native-controls-abba-verified.json').read_text());assert proof['status']=='four_leg_controls_abba_evidence_verified'
assert proof['archive_sha256']==hashlib.sha256((r/'results/native-controls-abba-evidence.zip').read_bytes()).hexdigest()
events=[json.loads(l) for l in (r/'results/native-controls-abba-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
out=r/'results/deployed-io-resources-attempt1';assert not out.exists();library=r/'build/bin/garnet.dll';before=hashlib.sha256(library.read_bytes()).hexdigest();assert before=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
tooldir=r/'tools/cuobjdump132';assert tooldir.resolve().is_relative_to((r/'tools').resolve());tool=tooldir/'cuobjdump.exe'
with zipfile.ZipFile(r/'cuobjdump132-tool.zip') as z:
 assert set(z.namelist())=={'cuobjdump.exe','manifest.json'};receipt=json.loads(z.read('manifest.json'));data=z.read('cuobjdump.exe');assert hashlib.sha256(data).hexdigest()==receipt['sha256']=='7405a7a6f0c64bc7d962bb5cfc30ce24d122c8c6ac6d89c0636f12f60bf3c6fe'
 tooldir.mkdir(exist_ok=True)
 if tool.exists():assert tool.read_bytes()==data
 else:tool.write_bytes(data)
out.mkdir();signature=subprocess.run(['powershell','-NoProfile','-Command',"$s=Get-AuthenticodeSignature -LiteralPath '"+str(tool)+"';[PSCustomObject]@{Status=$s.Status.ToString();Signer=$s.SignerCertificate.Subject}|ConvertTo-Json -Compress"],capture_output=True,text=True);(out/'signature.json').write_text(signature.stdout);sig=json.loads(signature.stdout);assert signature.returncode==0 and sig['Status']=='Valid' and 'NVIDIA Corporation' in sig['Signer']
version=subprocess.run([str(tool),'--version'],capture_output=True,text=True);(out/'version.log').write_text(version.stdout+version.stderr);assert version.returncode==0
code=run([str(tool),'--dump-resource-usage',str(library)],out/'resource-ledger','correctness','read-only-deployed-cuda-resource-inventory',[tool,library,r/'run-deployed-io-resources.py']);assert code==0
# run_logged preserves the unmodified raw command output separately.
text=(out/'resource-ledger/commands.log').read_bytes().decode('utf8',errors='replace');matches=list(re.finditer(r'Function\s*:\s*(\S+)',text));rows=[]
for i,m in enumerate(matches):
 if not any(k in m[1] for k in ('GatherChunk','WriteRequest')):continue
 end=matches[i+1].start() if i+1<len(matches) else len(text);block=text[m.start():end];architectures=re.findall(r'arch\s*=\s*(sm_\d+)',text[:m.start()]);local=re.search(r'LOCAL:\s*(\d+)',block);stack=re.search(r'STACK:\s*(\d+)',block)
 rows.append({'function':m[1],'architecture':architectures[-1] if architectures else None,'local_bytes_per_thread':int(local[1]) if local else None,'stack_bytes_per_thread':int(stack[1]) if stack else None,'raw':block})
assert rows and all(row['local_bytes_per_thread'] is not None for row in rows),'Resource format unsupported; preserve raw output, no local-memory conclusion'
sm120=[row for row in rows if row['architecture']=='sm_120'];status='deployed_sm120_static_resources_read' if any('GatherChunk' in v['function'] for v in sm120) and any('WriteRequest' in v['function'] for v in sm120) else 'embedded_other_arch_resources_only_no_runtime_jit_claim'
selected_cache=[line for line in (r/'build/CMakeCache.txt').read_text().splitlines() if line.startswith(('CMAKE_CUDA_ARCHITECTURES:','CMAKE_CUDA_COMPILER:'))]
assert hashlib.sha256(library.read_bytes()).hexdigest()==before
report={'status':status,'library_sha256':before,'tool':receipt,'version':version.stdout,'signature':sig,'selected_cmake_cache':selected_cache,'kernels':rows,'scope':'Offline metadata of actual unchanged deployed DLL, no GPU/kernel timing. Static per-thread local resources are not measured global VRAM use or proof of allocation/pause cause. Other-arch rows cannot prove actual SM120 JIT resources. No default/source/binary/driver change or speed claim.'};(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
