"""Exact common frame preparation with fewer intermediate arrays; optional trial."""
import math
import numpy as np
from PIL import ImageOps,Image

def prepare_frame(image,config,processor,batch,budget=65536):
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
    pixels=np.asarray(resized,dtype=np.float32)
    # Keep all three separately rounded FP32 operations and their order.
    np.divide(pixels,np.float32(255),out=pixels)
    np.subtract(pixels,np.asarray(processor['image_mean'],np.float32),out=pixels)
    np.divide(pixels,np.asarray(processor['image_std'],np.float32),out=pixels)
    gh,gw=rh//patch,rw//patch;temporal=config['temporal_patch_size']
    values=np.broadcast_to(pixels.transpose(2,0,1)[:,None,:,:],(3,temporal,rh,rw))
    values=values.reshape(3,temporal,gh//merge,merge,patch,gw//merge,merge,patch)
    patches=values.transpose(2,5,3,6,0,1,4,7).reshape(gh*gw,-1)
    result=patches[None,:,:] if batch==1 else np.broadcast_to(patches,(batch,*patches.shape))
    return np.ascontiguousarray(result,dtype='<f4'),resized.tobytes(),(rh,rw)
