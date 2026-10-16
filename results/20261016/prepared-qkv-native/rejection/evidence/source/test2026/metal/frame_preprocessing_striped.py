"""Optional exact-coordinate Pillow 12.3 RGB resize strips; fresh pixels every call."""
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageOps

class StripedFramePreparer:
    def __init__(self,workers=4):
        if workers not in (2,4):raise ValueError('Supported strip counts are 2 and 4')
        self.workers=workers;self.pool=ThreadPoolExecutor(max_workers=workers)
        self.last_route=None;self.parallel_calls=0;self.fallback_calls=0
    def close(self):self.pool.shutdown(wait=True)
    def resize(self,image,size):
        width,height=image.size;rw,rh=size;g=math.gcd(height,rh);unit=rh//g
        # Integer box edges fit Pillow's float32 boxes. A dyadic reduced scale
        # makes global/strip center arithmetic exact in binary64 at these bounds.
        eligible=(Image.__version__=='12.3.0' and image.mode=='RGB' and
            min(width,height,rw,rh)>0 and max(width,height,rw,rh)<=2**20 and
            width*height>=2**18 and height!=rh and g>=2 and unit&(unit-1)==0)
        if not eligible:
            self.last_route='fallback';self.fallback_calls+=1
            return image.resize(size,Image.Resampling.BICUBIC)
        image.load();count=min(self.workers,g);jobs=[]
        for i in range(count):
            top=(g*i//count)*unit;bottom=(g*(i+1)//count)*unit
            box=(0,(top//unit)*(height//g),width,(bottom//unit)*(height//g))
            jobs.append((top,self.pool.submit(image.resize,(rw,bottom-top),Image.Resampling.BICUBIC,box)))
        output=Image.new('RGB',size)
        try:
            for top,future in jobs:
                tile=future.result()
                try:output.paste(tile,(0,top))
                finally:tile.close()
        except BaseException:
            # Drain every reader before the caller can close its source image.
            for _,future in jobs:
                try:future.result().close()
                except BaseException:pass
            output.close();raise
        self.last_route='parallel';self.parallel_calls+=1
        return output
    def prepare(self,image,config,processor,batch,budget=65536):
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
        resized=self.resize(image,(rw,rh))
        pixels=np.asarray(resized,dtype=np.float32)/np.float32(255)
        pixels=(pixels-np.asarray(processor['image_mean'],np.float32))/np.asarray(processor['image_std'],np.float32)
        gh,gw=rh//patch,rw//patch;temporal=config['temporal_patch_size']
        values=np.repeat(pixels.transpose(2,0,1)[:,None,:,:],temporal,axis=1)
        values=values.reshape(3,temporal,gh//merge,merge,patch,gw//merge,merge,patch)
        patches=values.transpose(2,5,3,6,0,1,4,7).reshape(gh*gw,-1)
        return np.tile(patches,(batch,1,1)).astype('<f4'),resized.tobytes(),(rh,rw)
