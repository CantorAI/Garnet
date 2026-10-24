"""Lossy JPEG nonce/cache audit; generated fixtures are not benchmark results."""
import base64,hashlib,unittest
from io import BytesIO
from types import SimpleNamespace
from PIL import Image
from benchmark_stream import prepare_workloads,payload
class ImageNonceTests(unittest.TestCase):
 def test_actual_seed3_cohorts_keep_encoded_and_decoded_images_unique(self):
  a=SimpleNamespace(cases=['text','image'],concurrency=[1,2,4,8],warmups=2,requests=8,workload_seed=3,model='qwen38-27b-nvfp4',max_tokens=128,image_format='jpeg',image_order='image-first',image_nonce_pattern='binary-grid')
  groups,images=prepare_workloads(a)
  self.assertEqual(len(images),62);self.assertEqual(len({x['encoded_sha256'] for x in images}),62);self.assertEqual(len({x['decoded_rgb_sha256'] for x in images}),62)
  body=groups[('image',1)]['measured'][0][1];self.assertEqual(body,payload('image','s3-image-c1-r0',321000,a.model,128,'jpeg','image-first','binary-grid')[0])
 def test_jpeg_preserves_nonce_bit_boundaries_in_visible_pixels(self):
  for seed in (0,1,255,256,65535,65536,2**24-1):
   _,raw=payload('image','fixture',seed,'model',128,'jpeg','image-first','binary-grid')
   with Image.open(BytesIO(raw)) as image:
    image=image.convert('RGB')
    self.assertEqual(image.size,(224,224))
    for bit in range(24):
     pixel=image.getpixel((160+(bit%8)*8+4,200+(bit//8)*8+4))
     self.assertTrue(min(pixel)>223 if seed&(1<<bit) else max(pixel)<32)
  with self.assertRaises(ValueError):payload('image','fixture',2**24,'model',128,'jpeg','image-first','binary-grid')
if __name__=='__main__':unittest.main()
