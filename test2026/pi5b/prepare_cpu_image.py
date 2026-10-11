"""CPU still-image preprocessing for Garnet's fixed image graph.

Pillow bicubic resize, official 0.5 mean/std, duplicated temporal frame,
spatial-merge patch order. This does not run any learned model operators.
"""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps

def prepare(image,config,processor,min_pixels=65536,max_pixels=65536):
    if min_pixels<=0 or max_pixels<min_pixels:raise ValueError('invalid pixel bounds')
    image=ImageOps.exif_transpose(image).convert('RGB');width,height=image.size
    patch=config['patch_size'];merge=config['spatial_merge_size'];factor=patch*merge
    if max(height,width)/min(height,width)>200:raise ValueError('image aspect ratio exceeds 200')
    rh=max(factor,round(height/factor)*factor);rw=max(factor,round(width/factor)*factor)
    if rh*rw>max_pixels:
        beta=math.sqrt(height*width/max_pixels);rh=max(factor,math.floor(height/beta/factor)*factor);rw=max(factor,math.floor(width/beta/factor)*factor)
    elif rh*rw<min_pixels:
        beta=math.sqrt(min_pixels/(height*width));rh=math.ceil(height*beta/factor)*factor;rw=math.ceil(width*beta/factor)*factor
    resized=image.resize((rw,rh),Image.Resampling.BICUBIC)
    pixels=np.asarray(resized,dtype=np.float32)/np.float32(255)
    pixels=(pixels-np.asarray(processor['image_mean'],dtype=np.float32))/np.asarray(processor['image_std'],dtype=np.float32)
    gh,gw=rh//patch,rw//patch;patches=[]
    for bh in range(0,gh,merge):
        for bw in range(0,gw,merge):
            for ih in range(merge):
                for iw in range(merge):
                    h,w=bh+ih,bw+iw
                    tile=pixels[h*patch:(h+1)*patch,w*patch:(w+1)*patch].transpose(2,0,1)
                    patches.append(np.repeat(tile[:,None,:,:],config['temporal_patch_size'],axis=1).reshape(-1))
    return np.asarray(patches,dtype=np.float32),(gh,gw),(height,width),(rh,rw)

if __name__=='__main__':
    image_path,model_folder,output=map(Path,sys.argv[1:4]);batch=int(sys.argv[4]) if len(sys.argv)>4 else 1
    pixel_budget=int(sys.argv[5]) if len(sys.argv)>5 else 65536
    if batch not in (1,2):raise ValueError('batch must be 1 or 2')
    config=json.loads((model_folder/'config.json').read_text())['vision_config']
    processor=json.loads((model_folder/'preprocessor_config.json').read_text())
    with Image.open(image_path) as image:patches,grid,source,resized=prepare(image,config,processor,pixel_budget,pixel_budget)
    values=np.repeat(patches[None,:,:],batch,axis=0).astype('<f4');values.tofile(output)
    metadata=dict(image=str(image_path),image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        patch_file=str(output),patch_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),shape=list(values.shape),
        grid_thw=[1,*grid],source_hw=list(source),resized_hw=list(resized),resize='Pillow bicubic',
        min_pixels=pixel_budget,max_pixels=pixel_budget,mean=processor['image_mean'],std=processor['image_std'])
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
