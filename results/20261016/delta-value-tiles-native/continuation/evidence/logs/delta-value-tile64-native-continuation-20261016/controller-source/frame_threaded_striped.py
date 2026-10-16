"""Optional composition of independently exact static-WebP decode and strip resize."""
from pathlib import Path
from PIL import Image
from frame_webp_threaded import WebPDecoder
from frame_preprocessing_striped import StripedFramePreparer

class ThreadedStripedDecoder(WebPDecoder):
    def __init__(self,library):
        super().__init__(library);self.striped=StripedFramePreparer();self.last_decode_route=None
    def prepare(self,path,config,processor,batch,threads=1):
        raw=Path(path).read_bytes();pixels=self.pixels(raw,threads)
        if pixels is None:
            self.last_decode_route='pillow'
            with Image.open(path) as image:return self.striped.prepare(image,config,processor,batch)
        self.last_decode_route='native-threaded' if threads else 'native-single'
        with Image.fromarray(pixels) as image:return self.striped.prepare(image,config,processor,batch)
    def close(self):self.striped.close()
