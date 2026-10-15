"""Conservative static-WebP CPU experiment; unchanged Pillow preprocessing."""
import ctypes
from pathlib import Path
import numpy as np
from PIL import Image
from frame_frontend_candidates import simple_webp_eligible
from frame_preprocessing import prepare_frame

class WebPDecoder:
    def __init__(self,library):
        self.library=ctypes.CDLL(str(library));self.info=self.library.garnet_webp_info;self.decode=self.library.garnet_webp_decode
        self.info.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)];self.info.restype=ctypes.c_int
        self.decode.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];self.decode.restype=ctypes.c_int
        self.library.garnet_webp_version.restype=ctypes.c_int
        self.version=self.library.garnet_webp_version()
    def pixels(self,raw,threads):
        if not simple_webp_eligible(raw):return None
        width=ctypes.c_int();height=ctypes.c_int()
        if self.info(raw,len(raw),ctypes.byref(width),ctypes.byref(height)):return None
        values=np.empty((height.value,width.value,3),np.uint8)
        if self.decode(raw,len(raw),int(threads),values.ctypes.data,values.nbytes):return None
        return values
    def prepare(self,path,config,processor,batch,threads):
        raw=Path(path).read_bytes();pixels=self.pixels(raw,threads)
        if pixels is None:
            with Image.open(path) as image:return prepare_frame(image,config,processor,batch)
        with Image.fromarray(pixels) as image:return prepare_frame(image,config,processor,batch)
