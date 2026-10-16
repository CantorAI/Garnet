"""Compare complete RGB and FP32 patch bytes on independent image fixtures."""
import hashlib,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from frame_preprocessing import prepare_frame as control
from frame_preprocessing_compact import prepare_frame as candidate

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);checks=[]
for size in ((1,1),(31,47),(64,96),(513,257)):
    w,h=size;rgb=(np.arange(w*h*3,dtype=np.uint32)*37%256).astype(np.uint8).reshape(h,w,3)
    for mode in ('RGB','RGBA','L'):
        base=Image.fromarray(rgb).convert(mode)
        for orientation in range(1,9):
            exif=base.getexif();exif[274]=orientation
            stream=io.BytesIO();base.save(stream,format='PNG',exif=exif);raw=stream.getvalue()
            for batch in (1,2):
                for temporal in (1,2):
                    cfg=dict(patch_size=16,spatial_merge_size=2,temporal_patch_size=temporal)
                    for mean,std in (([.5]*3,[.5]*3),([.123,.234,.345],[.271,.382,.493])):
                        processor=dict(image_mean=mean,image_std=std)
                        with Image.open(io.BytesIO(raw)) as opened:a,ar,ah=control(opened,cfg,processor,batch)
                        with Image.open(io.BytesIO(raw)) as opened:b,br,bh=candidate(opened,cfg,processor,batch)
                        assert ah==bh and ar==br and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
                        assert b.flags.c_contiguous and b.dtype==np.dtype('<f4')
                        checks.append(dict(size=size,mode=mode,orientation=orientation,batch=batch,temporal=temporal,mean=mean,std=std,rgb_sha256=hashlib.sha256(br).hexdigest(),patch_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
report=dict(scope=__doc__,numpy=np.__version__,pillow=Image.__version__,comparisons=len(checks),all_complete_bytes_match=True,checks=checks)
(out/'verification.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
