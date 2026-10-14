'Exclusive original BF16 kernel with row-two work made compile-time dead.'
import ast,hashlib,json,mmap,os,signal,struct,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
s=root/'source/test2026/metal';output=out;statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def check(owned=None):
 if shutil.disk_usage(root).free<2*2**30:raise OSError('Disk free space below 2 GiB')
 strict_guard()
 for line in snapshot().splitlines():
  fields=line.split()
  if fields and fields[-1].lower().endswith(('_gpu','_probe')) and int(fields[0])!=owned:raise RuntimeError('Other GPU probe PID '+fields[0])
def run(name,command):
 check();samples=[];begin=time.monotonic()
 with (out/(name+'.log')).open('w') as stream:
  process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while process.poll() is None:
    check(process.pid);samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
    if time.monotonic()-begin>1200:raise TimeoutError('Owned real projection gate exceeded deadline')
    time.sleep(.5)
  except BaseException:
   if process.poll() is None:
    assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
   raise
 (out/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n')
 statuses.append(dict(name=name,command=command,exit_code=process.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
 assert process.returncode==0,name
 check()
native=root/'source/src/backends/metal/metal_context.mm'
paths=[native,s/'indirect_bf16_projection.mm',s/'make_single_row_bf16_probe.py',s/'make_constant_bf16_probe.py',s/'run_constant_bf16_probe.py',Path(__file__),s/'compare_vllm_metal.py']
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},
 numerical_bounds='Bit-exact versus current native shader; gamma(ceil(K/32)+32)*sumabs+1e-7 versus independent FP64',
 hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
 versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)),assets={})
for p in paths:
 target=out/'source-snapshot'/p.relative_to(root/'source');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
native_source=native.read_text();shader=out/'native-decode.metal'
shader.write_bytes(('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native_source[native_source.index('kernel void bf16_decode('):native_source.index('kernel void bf16_gemm(')]).encode())
shader_text=shader.read_text()
single=shader_text.replace('kernel void bf16_decode(', 'kernel void single_bf16_decode(').replace('kernel void bf16_decode_small(', 'kernel void single_bf16_decode_small(').replace('p.rows==2','false')
single=single.replace('#include <metal_stdlib>','').replace('using namespace metal;','').replace('struct BF16Params {ulong rows,inner,columns,aStride,oStride;};','')
shader.write_bytes((shader_text+single).encode());metadata['shader_sha256']=digest(shader);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
probe=out/'single_row_bf16_gpu.mm'
run('harness',[sys.executable,str(s/'make_single_row_bf16_probe.py'),str(probe)])
binary=out/'single_row_bf16_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(probe),'-o',str(binary)])
metadata['binary_sha256']=digest(binary)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets')
 info=json.loads((assets/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors'
 assert digest(checkpoint)==info['checkpoint_sha256']
 for case in info['cases']:assert digest(assets/(case['name']+'.bin'))==case['sha256']
 with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  header_bytes=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+header_bytes]);base=8+header_bytes
  for case in info['cases']:
   if case['tensor'].startswith('synthetic'):continue
   tensor=header[case['tensor']];assert tensor['dtype']=='BF16' and tensor['shape']==[case['columns'],case['inner']]
   start,end=tensor['data_offsets'];assert hashlib.sha256(mapping[base+start:base+end]).hexdigest()==case['sha256']
 metadata['assets'][model]=info;(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 check();run(model+'-gpu',[str(binary),str(shader),str(assets)])
print('All row-specialized BF16 projections passed fresh-input/padding and FP64 gates',flush=True)
