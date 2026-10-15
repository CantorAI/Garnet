"""Common CPU-only frame preparation; no learned operators."""
import math
import numpy as np
from PIL import ImageOps,Image

def prepare_frame(image,config,processor,batch,budget=65536):
    image=ImageOps.exif_transpose(image).convert('RGB');width,height=image.size
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
