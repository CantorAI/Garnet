"""Exact-output CPU frontend candidates; fresh decode on every call."""
import math,struct
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from frame_preprocessing import prepare_frame

def prepare_borrowed(image,config,processor,batch,budget=65536):
    # Match the control's metadata availability before deciding orientation;
    # PNG can place EXIF after compressed image data.
    image.load()
    orientation=image.getexif().get(274)
    if orientation in (2,3,4,5,6,7,8):image=ImageOps.exif_transpose(image)
    if image.mode!='RGB':image=image.convert('RGB')
    width,height=image.size;patch=config['patch_size'];merge=config['spatial_merge_size'];factor=patch*merge
    rh=max(factor,round(height/factor)*factor);rw=max(factor,round(width/factor)*factor)
    if rh*rw>budget:
        beta=math.sqrt(height*width/budget);rh=max(factor,math.floor(height/beta/factor)*factor);rw=max(factor,math.floor(width/beta/factor)*factor)
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

def png_fast_eligible(raw):
    # Conservatively fall back for metadata/animation that can affect orientation
    # or image selection. Do not assume metadata precedes IDAT.
    if raw[:8]!=b'\x89PNG\r\n\x1a\n':return False
    offset=8;header=False;end=False
    allowed={b'IHDR',b'IDAT',b'IEND',b'sRGB',b'gAMA',b'cHRM',b'pHYs',b'iCCP',b'sBIT',b'bKGD',b'tIME'}
    while offset+12<=len(raw):
        length=struct.unpack_from('>I',raw,offset)[0];kind=raw[offset+4:offset+8]
        if length>len(raw)-offset-12 or kind not in allowed:return False
        if kind==b'IHDR':
            if header or offset!=8 or length!=13:return False
            width,height,depth,color,compression,filtering,interlace=struct.unpack_from('>IIBBBBB',raw,offset+8)
            if not width or not height or depth!=8 or color!=2 or compression or filtering or interlace not in (0,1):return False
            header=True
        if kind==b'IEND':
            if length:return False
            end=True;offset+=12;break
        offset+=length+12
    return header and end and offset==len(raw)

def simple_webp_eligible(raw):
    if len(raw)<20 or raw[:4]!=b'RIFF' or raw[8:12]!=b'WEBP' or struct.unpack_from('<I',raw,4)[0]+8!=len(raw):return False
    offset=12;images=0
    while offset+8<=len(raw):
        kind=raw[offset:offset+4];length=struct.unpack_from('<I',raw,offset+4)[0]
        if kind not in (b'VP8 ',b'VP8L') or length>len(raw)-offset-8:return False
        images+=1;offset+=8+length+(length&1)
    return images==1 and offset==len(raw)

def prepare_cv_frame(path,config,processor,batch,budget=65536):
    raw=Path(path).read_bytes()
    if png_fast_eligible(raw) or simple_webp_eligible(raw):
        import cv2
        cv2.ocl.setUseOpenCL(False)
        decoded=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
        if decoded is not None and decoded.dtype==np.uint8 and decoded.ndim==3 and decoded.shape[2]==3:
            rgb=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
            image=Image.fromarray(rgb)
            return prepare_borrowed(image,config,processor,batch,budget)
    with Image.open(path) as image:return prepare_frame(image,config,processor,batch,budget)
