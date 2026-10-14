"""Guarded CPU-only build, exactness and fresh-source ABBA preparation timings."""
import ast,hashlib,json,os,subprocess,sys,time,statistics
from pathlib import Path
import numpy as np
from PIL import Image,_webp,features
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';sys.path.insert(0,str(s));background_checks=[]
from frame_preprocessing import prepare_frame
from frame_webp_threaded import WebPDecoder
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def check():
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+f[0])
check();library=Path(_webp.__file__).parent/'.dylibs/libwebp.7.dylib';assert library.is_file()
target=output/'libwebp_thread.dylib'
command=['/usr/bin/xcrun','clang++','-std=c++17','-O3','-dynamiclib','-I/opt/homebrew/include',str(s/'webp_thread_decode.cpp'),str(library),'-Wl,-rpath,'+str(library.parent),'-o',str(target)]
with (output/'compile.log').open('w') as stream:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
(output/'compile-status.json').write_text(json.dumps(dict(command=command,exit_code=code),indent=2)+'\n');assert code==0
decoder=WebPDecoder(target)
expected_version=tuple(map(int,features.version('webp').split('.')));actual=(decoder.version>>16,(decoder.version>>8)&255,decoder.version&255);assert actual==expected_version
model=r/'models/Qwen3.5-0.8B';config=json.loads((model/'config.json').read_text())['vision_config'];processor=json.loads((model/'preprocessor_config.json').read_text())
image=r/'images/RealWorld-04.png';raw=image.read_bytes()
with Image.open(image) as opened:opened.load();reference=np.asarray(opened.convert('RGB')).copy()
checks=[]
for threads in (0,1):
    pixels=decoder.pixels(raw,threads);assert pixels is not None and np.array_equal(pixels,reference)
    checks.append(dict(threads=threads,full_rgb_exact=True,rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
records=[];cases={}
for batch in (1,2):
    path=r/'logs/selected-attention-vllm-20261013/jobs'/('Qwen3.5-0.8B-b'+str(batch)+'-0/case.json');cases[str(batch)]=json.loads(path.read_text());meta=cases[str(batch)]['image_metadata']
    def run(policy):
        start=time.perf_counter()
        if policy=='pillow':
            with Image.open(image) as opened:patches,rgb,hw=prepare_frame(opened,config,processor,batch)
        else:patches,rgb,hw=decoder.prepare(image,config,processor,batch,policy=='threaded')
        patch=output/'frame.f32';rgbfile=output/'frame.rgb';patches.tofile(patch);rgbfile.write_bytes(rgb)
        digest=hashlib.sha256(patch.read_bytes()).hexdigest();rgb_hash=hashlib.sha256(rgb).hexdigest();elapsed=(time.perf_counter()-start)*1000
        assert digest==meta['patch_sha256'] and list(patches.shape)==meta['shape'] and list(hw)==meta['resized_hw']
        return dict(batch=batch,policy=policy,elapsed_ms=elapsed,patch_sha256=digest,rgb_sha256=rgb_hash)
    for candidate in ('native-single','threaded'):
        for repeat in range(2):
            for policy in ('pillow',candidate,candidate,'pillow'):
                check();run(policy)
                for sample in range(12):records.append(dict(candidate=candidate,repeat=repeat,sample=sample,**run(policy)))
metadata=dict(scope='CPU preprocessing only; fresh file/decode/resize/normalize/patch/write/hash each sample; no inference speed claim',
    pillow=Image.__version__,webp=features.version('webp'),native_version=actual,image_sha256=hashlib.sha256(raw).hexdigest(),library_path=str(library),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),cases=cases,
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),s/'webp_thread_decode.cpp',s/'frame_webp_threaded.py',s/'frame_preprocessing.py',s/'frame_frontend_candidates.py',s/'compare_vllm_metal.py')})
copy=output/'source-snapshot';copy.mkdir()
for name in metadata['source_sha256']:(copy/name).write_bytes((s/name).read_bytes())
results=[]
for batch in (1,2):
    for candidate in ('native-single','threaded'):
        control=statistics.median(v['elapsed_ms'] for v in records if v['batch']==batch and v['candidate']==candidate and v['policy']=='pillow')
        value=statistics.median(v['elapsed_ms'] for v in records if v['batch']==batch and v['candidate']==candidate and v['policy']==candidate)
        results.append(dict(batch=batch,candidate=candidate,pillow_ms=control,candidate_ms=value,speedup=control/value))
assert len(records)==384 and len({v['rgb_sha256'] for v in records})==1
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,checks=checks,records=records,results=results),indent=2)+'\n');check();print(json.dumps(results),flush=True)
