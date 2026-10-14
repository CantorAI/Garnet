"""Freshly opened inputs: exact RGB/patch parity and conservative WebP fallback."""
import ast,hashlib,io,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
from PIL import Image
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';sys.path.insert(0,str(s));background_checks=[]
from frame_preprocessing import prepare_frame
from frame_webp_threaded import WebPDecoder
from frame_frontend_candidates import simple_webp_eligible
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def check():
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+f[0])
library=Path(sys.argv[3]).resolve();decoder=WebPDecoder(library)
model=r/'models/Qwen3.5-0.8B';config=json.loads((model/'config.json').read_text())['vision_config'];processor=json.loads((model/'preprocessor_config.json').read_text());rng=np.random.default_rng(949)
records=[];checks=0;branches=0
with tempfile.TemporaryDirectory(prefix='garnet-threaded-webp-') as temporary:
    path=Path(temporary)/'input.webp'
    def verify(label):
        global checks,branches
        raw=path.read_bytes();native=decoder.pixels(raw,1);applied=native is not None
        branches+=int(applied)
        for batch in (1,2):
            with Image.open(path) as control:expected=prepare_frame(control,config,processor,batch)
            for threads in (0,1):
                actual=decoder.prepare(path,config,processor,batch,threads)
                assert np.array_equal(actual[0],expected[0]) and actual[0].tobytes()==expected[0].tobytes()
                assert actual[1:]==expected[1:];checks+=1
        records.append(dict(case=label,native_applied=applied,container_eligible=simple_webp_eligible(raw),source_sha256=hashlib.sha256(raw).hexdigest()))
    for width,height in ((32,48),(128,160),(511,257),(1024,768)):
        check();base=Image.fromarray(rng.integers(0,256,(height,width,3),dtype=np.uint8))
        for mode in ('RGB','RGBA','L'):
            image=base.convert(mode)
            if mode=='RGBA':image.putalpha(Image.fromarray(rng.integers(0,256,(height,width),dtype=np.uint8)))
            for lossless in (False,True):
                for orientation in (None,1,2,3,4,5,6,7,8):
                    options=dict(format='WEBP',lossless=lossless,quality=80)
                    if orientation is not None:
                        exif=Image.Exif();exif[274]=orientation;options['exif']=exif
                    image.save(path,**options);verify(f'{width}x{height}-{mode}-{lossless}-{orientation}')
                    if orientation is not None:assert not records[-1]['native_applied']
                    if mode=='RGBA':assert not records[-1]['native_applied']
        base.save(path,format='WEBP',save_all=True,append_images=[base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)],duration=30);verify('animated');assert not records[-1]['native_applied']
        for fmt in ('PNG','JPEG','BMP'):
            base.save(path,format=fmt);verify(fmt+'-fallback');assert not records[-1]['native_applied']
    # Loading must still expose orientation in a PNG EXIF chunk after IDAT.
    import struct,zlib
    for orientation in range(2,9):
        exif=Image.Exif();exif[274]=orientation;buffer=io.BytesIO();base.save(buffer,format='PNG');raw=buffer.getvalue()
        data=exif.tobytes()[6:];chunk=b'eXIf'+data;encoded=struct.pack('>I',len(data))+chunk+struct.pack('>I',zlib.crc32(chunk)&0xffffffff)
        assert raw[-8:-4]==b'IEND';path.write_bytes(raw[:-12]+encoded+raw[-12:]);verify('late-exif-'+str(orientation));assert not records[-1]['native_applied']
check();assert checks==956 and branches>0
metadata=dict(checks=checks,fixtures=len(records),native_fixtures=branches,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),s/'frame_webp_threaded.py',s/'frame_preprocessing.py')})
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,records=records),indent=2)+'\n');print(json.dumps(metadata),flush=True)
