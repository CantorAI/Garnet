"""Exact RGB/patch compatibility across image modes, EXIF and late PNG metadata."""
import hashlib,io,json,struct,sys,zlib
from pathlib import Path
import numpy as np
from PIL import Image
from frame_preprocessing_control import prepare_frame
from frame_preprocessing import prepare_frame as prepare_selected
from frame_frontend_candidates import prepare_borrowed,prepare_cv_frame
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
config=dict(patch_size=2,spatial_merge_size=2,temporal_patch_size=2);processor=dict(image_mean=[.5]*3,image_std=[.5]*3);checks=[]
def compare(image,label):
    for batch in (1,2):
        control=prepare_frame(image,config,processor,batch,64);candidate=prepare_borrowed(image,config,processor,batch,64)
        selected=prepare_selected(image,config,processor,batch,64)
        assert control[0].tobytes()==candidate[0].tobytes() and control[1:]==candidate[1:],label
        assert control[0].tobytes()==selected[0].tobytes() and control[1:]==selected[1:],label
        checks.append(dict(label=label,batch=batch,patch_sha256=hashlib.sha256(control[0].tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(control[1]).hexdigest()))
for height,width in ((1,1),(3,7),(9,5),(17,19)):
    values=np.random.default_rng(height*width).integers(0,256,(height,width,3),dtype=np.uint8)
    for mode in ('RGB','RGBA','L','P','I','CMYK'):
        original=Image.fromarray(values).convert(mode)
        for orientation in range(1,9):
            image=original.copy();image.getexif()[274]=orientation;compare(image,f'{mode}-{width}x{height}-exif{orientation}')
        compare(original,f'{mode}-{width}x{height}-noexif')
# Encode EXIF after IDAT to verify the borrowed path loads late metadata.
base=Image.fromarray(np.random.default_rng(22).integers(0,256,(9,13,3),dtype=np.uint8));stream=io.BytesIO();base.save(stream,format='PNG');raw=stream.getvalue()
for orientation in range(2,9):
    exif=Image.Exif();exif[274]=orientation;payload=exif.tobytes()[6:];kind=b'eXIf'
    chunk=struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    late=raw[:-12]+chunk+raw[-12:];path=output/f'late-exif-{orientation}.png';path.write_bytes(late)
    for batch in (1,2):
        with Image.open(path) as image:control=prepare_frame(image,config,processor,batch,64)
        with Image.open(path) as image:candidate=prepare_borrowed(image,config,processor,batch,64)
        with Image.open(path) as image:selected=prepare_selected(image,config,processor,batch,64)
        assert control[0].tobytes()==candidate[0].tobytes() and control[1:]==candidate[1:]
        assert control[0].tobytes()==selected[0].tobytes() and control[1:]==selected[1:]
        checks.append(dict(label=f'late-png-exif{orientation}-fresh-open',batch=batch,patch_sha256=hashlib.sha256(control[0].tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(control[1]).hexdigest()))
    # Metadata images deliberately use the conservative CV fallback.
    for batch in (1,2):
        with Image.open(path) as image:control=prepare_frame(image,config,processor,batch,64)
        candidate=prepare_cv_frame(path,config,processor,batch,64)
        assert control[0].tobytes()==candidate[0].tobytes() and control[1:]==candidate[1:]
# Real frozen asset, fresh opening/decoding at both batch sizes.
model=r/'models/Qwen3.5-0.8B';cfg=json.loads((model/'config.json').read_text())['vision_config'];proc=json.loads((model/'preprocessor_config.json').read_text())
for batch in (1,2):
    case=json.loads((r/f'logs/joined-prefill-native-20261013/jobs/Qwen3.5-0.8B-b{batch}-attempt0/Qwen3.5-0.8B-b{batch}-case.json').read_text());meta=case['image_metadata']
    with Image.open(meta['image']) as image:control=prepare_frame(image,cfg,proc,batch)
    with Image.open(meta['image']) as image:candidate=prepare_borrowed(image,cfg,proc,batch)
    with Image.open(meta['image']) as image:selected=prepare_selected(image,cfg,proc,batch)
    assert control[0].tobytes()==candidate[0].tobytes() and control[1:]==candidate[1:]
    assert control[0].tobytes()==selected[0].tobytes() and control[1:]==selected[1:]
    assert hashlib.sha256(candidate[0].tobytes()).hexdigest()==meta['patch_sha256']
    checks.append(dict(label='frozen-real-asset',batch=batch,patch_sha256=meta['patch_sha256'],rgb_sha256=hashlib.sha256(control[1]).hexdigest()))
(output/'summary.json').write_text(json.dumps(dict(scope=__doc__,checks=checks,late_exif_cv_fallback_checks=14,exact=True),indent=2))
print(json.dumps(dict(exact_checks=len(checks),late_exif_cv_fallback_checks=14)),flush=True)
