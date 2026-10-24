from pathlib import Path
import hashlib,json,re,sys
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/deployed-io-resources-attempt1';dest=out/'corrected-parse.json';assert not dest.exists()
events=[json.loads(l) for l in (r/'results/deployed-io-resources-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==1
commands=[json.loads(l) for l in (out/'resource-ledger/commands.jsonl').read_text().splitlines()];assert commands[-1]['exit_code']==0
text=(out/'resource-ledger/commands.log').read_bytes().decode('utf8',errors='replace');matches=list(re.finditer(r'^\s*Function\s*:?\s+([^\r\n]+)',text,re.M));rows=[]
for i,m in enumerate(matches):
 name=m[1].strip().rstrip(':')
 if not any(k in name for k in ('GatherChunk','WriteRequest')):continue
 end=matches[i+1].start() if i+1<len(matches) else len(text);block=text[m.start():end];arches=re.findall(r'arch\s*=\s*(sm_\d+)',text[:m.start()]);local=re.search(r'LOCAL:\s*(\d+)',block);stack=re.search(r'STACK:\s*(\d+)',block);constant=re.search(r'CONSTANT\[0\]:\s*(\d+)',block)
 assert local and stack
 rows.append({'function':name,'architecture':arches[-1] if arches else None,'local_bytes_per_thread':int(local[1]),'stack_bytes_per_thread':int(stack[1]),'constant0_bytes':int(constant[1]) if constant else None,'raw':block})
assert len(rows)==2 and all(v['architecture']=='sm_120' for v in rows) and sum('GatherChunk' in v['function'] for v in rows)==sum('WriteRequest' in v['function'] for v in rows)==1
library=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();assert library=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
cache=[l for l in (r/'build/CMakeCache.txt').read_text().splitlines() if l.startswith(('CMAKE_CUDA_ARCHITECTURES:','CMAKE_CUDA_COMPILER:'))]
report={'status':'deployed_sm120_static_resources_read','original_controller_exit':1,'resource_tool_exit':0,'library_sha256':library,'resource_log_sha256':hashlib.sha256((out/'resource-ledger/commands.log').read_bytes()).hexdigest(),'selected_cmake_cache':cache,'kernels':rows,'scope':'Corrected offline parser for actual Function NAME: format; raw tool output not regenerated. Static compiled local/stack bytes are not global VRAM attribution or model timing. Original parser failure retained. No source/runtime/driver or GPU execution change.'};dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
