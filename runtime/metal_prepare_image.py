# SPDX-License-Identifier: Apache-2.0
"""Common CPU-only frame preparation; no learned operators."""
import math
import numpy as np
from PIL import ImageOps,Image

def prepare_frame(image,config,processor,batch,budget=65536):
    # Load before reading orientation, including EXIF chunks after image data.
    # RGB frames with no transform can be borrowed without two full-image copies.
    image.load()
    if image.getexif().get(274) in (2,3,4,5,6,7,8):image=ImageOps.exif_transpose(image)
    if image.mode!='RGB':image=image.convert('RGB')
    width,height=image.size
    patch=config['patch_size'];merge=config['spatial_merge_size'];factor=patch*merge
    rh=max(factor,round(height/factor)*factor);rw=max(factor,round(width/factor)*factor)
    if rh*rw>budget:
        beta=math.sqrt(height*width/budget)
        rh=max(factor,math.floor(height/beta/factor)*factor);rw=max(factor,math.floor(width/beta/factor)*factor)
    elif rh*rw<budget:
        beta=math.sqrt(budget/(height*width));rh=math.ceil(height*beta/factor)*factor;rw=math.ceil(width*beta/factor)*factor
    resized=image.resize((rw,rh),Image.Resampling.BICUBIC)
    pixels=np.asarray(resized,dtype=np.float32)/np.float32(255)
    pixels=(pixels-np.asarray(processor['image_mean'],np.float32))/np.asarray(processor['image_std'],np.float32)
    gh,gw=rh//patch,rw//patch;temporal=config['temporal_patch_size']
    values=np.repeat(pixels.transpose(2,0,1)[:,None,:,:],temporal,axis=1)
    values=values.reshape(3,temporal,gh//merge,merge,patch,gw//merge,merge,patch)
    patches=values.transpose(2,5,3,6,0,1,4,7).reshape(gh*gw,-1)
    return np.tile(patches,(batch,1,1)).astype('<f4'),resized.tobytes(),(rh,rw)


if __name__ == '__main__':
    import sys, json
    from pathlib import Path
    model, image_path, stem = sys.argv[1:]
    config = json.loads((Path(model)/'config.json').read_text())
    processor = json.loads((Path(model)/'preprocessor_config.json').read_text())
    with Image.open(image_path) as image:
        values, rgb, size = prepare_frame(image, config['vision_config'], processor, 1)
    values.tofile(stem + '.f32')
    patch = config['vision_config']['patch_size']
    Path(stem + '.json').write_text(json.dumps({'grid':[size[0]//patch,size[1]//patch]}))
