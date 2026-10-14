"""CPU-only phase attribution of the frozen fresh-frame frontend."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import math
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard','exec'),globals())
def check():
    guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+fields[0])
model=r/'models/Qwen3.5-0.8B';config=json.loads((model/'config.json').read_text())['vision_config'];processor=json.loads((model/'preprocessor_config.json').read_text())
source=(s/'frame_preprocessing_control.py').read_text();code=source[source.index('def prepare_frame('):]
code=code.replace('def prepare_frame(', 'def traced_prepare_frame(').replace('    image=ImageOps.',"    stamps=[('start',time.perf_counter())]\n    image=ImageOps.",1)
for label,needle in [('decode_exif_convert',"    patch=config['patch_size']"),('resize','    pixels=np.asarray(resized'),('normalize','    gh,gw='),('patch_layout','    return np.tile')]:
    code=code.replace(needle,f"    stamps.append(('{label}',time.perf_counter()))\n"+needle,1)
code=code.replace("    return np.tile(patches,(batch,1,1)).astype('<f4'),resized.tobytes(),(rh,rw)","    result=np.tile(patches,(batch,1,1)).astype('<f4');rgb=resized.tobytes()\n    stamps.append(('tile_cast_rgb',time.perf_counter()))\n    return result,rgb,(rh,rw),stamps")
assert 'stamps' in code;exec(compile(code,'traced_actual_frontend','exec'),globals())
results=[]
for batch in (1,2):
    case=r/f'logs/joined-prefill-native-20261013/jobs/Qwen3.5-0.8B-b{batch}-attempt0/Qwen3.5-0.8B-b{batch}-case.json'
    workload=json.loads(case.read_text());meta=workload['image_metadata'];check()
    for repeat in range(12):
        check();begin=time.perf_counter()
        with Image.open(meta['image']) as image:
            mode=image.mode;size=image.size;patches,rgb,resized,stamps=traced_prepare_frame(image,config,processor,batch)
        path=output/f'b{batch}-frame.f32';rgb_path=output/f'b{batch}-frame.rgb';written=time.perf_counter()
        patches.tofile(path);rgb_path.write_bytes(rgb);hashed=time.perf_counter()
        digest=hashlib.sha256(path.read_bytes()).hexdigest();end=time.perf_counter();assert digest==meta['patch_sha256'] and list(resized)==meta['resized_hw']
        phases={stamps[i][0]:(stamps[i][1]-stamps[i-1][1])*1000 for i in range(1,len(stamps))}
        phases.update(open_ms=(stamps[0][1]-begin)*1000,write_ms=(hashed-written)*1000,hash_ms=(end-hashed)*1000,total_ms=(end-begin)*1000)
        results.append(dict(batch=batch,repeat=repeat,source_mode=mode,source_wh=size,shape=list(patches.shape),strides=patches.strides,c_contiguous=patches.flags.c_contiguous,patch_sha256=digest,rgb_sha256=hashlib.sha256(rgb).hexdigest(),phases=phases))
    check()
summary={str(b):{k:statistics.mean(v['phases'][k] for v in results if v['batch']==b and v['repeat']>0) for k in results[0]['phases']} for b in (1,2)}
metadata=dict(scope=__doc__,source_sha256=hashlib.file_digest((s/'frame_preprocessing_control.py').open('rb'),'sha256').hexdigest(),self_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),image_sha256=hashlib.file_digest(Path(meta['image']).open('rb'),'sha256').hexdigest(),versions=dict(python=sys.version,numpy=np.__version__,pillow=Image.__version__))
(output/'traced-source.py').write_text(code)
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,means=summary,records=results),indent=2))
print(json.dumps(dict(means=summary,source_mode=results[0]['source_mode'],c_contiguous=results[0]['c_contiguous']),indent=2),flush=True)
