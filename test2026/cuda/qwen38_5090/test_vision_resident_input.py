"""Actual encoder control with independent CPU data buffers; no GPU replay claim."""
import ast,types,unittest
from pathlib import Path

def fixture():
 p=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_decoder.py'
 cls=next(n for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,ast.ClassDef) and n.name=='CUDAImageEncoder')
 methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('enable_resident_input','encode_patches','close')]
 scope={};exec(compile(ast.Module(body=methods,type_ignores=[]),str(p),'exec'),scope)
 class Encoder:pass
 for m in methods:setattr(Encoder,m.name,scope[m.name])
 encoder=Encoder();encoder.shape=[1,2,3];encoder.resident_io=None;encoder.resident_patches=None;encoder.resident_input_bytes=0
 calls=[];closed=[]
 def run(values):calls.append(values[0]);return list(values[0]['values'])
 encoder.graph=types.SimpleNamespace(run_device=run,close=lambda:closed.append(True))
 class IO:
  def zeros(self,shape):return {'shape':list(shape),'values':[0]*6}
  def tensor_bytes(self,p):return 24
  def copy_into(self,a,b):
   if a['shape']!=b['shape']:raise ValueError('shape mismatch')
   b['values'][:]=a['values']
 io=IO();return encoder,io,calls,closed

def image(value):return {'shape':[1,2,3],'values':[value]*6}
class VisionResidentInputTests(unittest.TestCase):
 def test_new_images_use_same_binding_without_stale_content(self):
  e,io,calls,_=fixture();e.enable_resident_input(io)
  first=image(3);self.assertEqual(e.encode_patches(first),[3]*6)
  first['values'][:]=[99]*6
  self.assertEqual(e.encode_patches(image(7)),[7]*6)
  self.assertIs(calls[0],calls[1]);self.assertIsNot(calls[0],first)
  self.assertEqual(e.resident_input_bytes,24)
 def test_bad_copy_cannot_run_graph_with_previous_image(self):
  e,io,calls,_=fixture();e.enable_resident_input(io);e.encode_patches(image(3))
  with self.assertRaisesRegex(ValueError,'shape'):e.encode_patches({'shape':[6],'values':[8]*6})
  self.assertEqual(len(calls),1)
 def test_default_direct_path_and_caller_owned_io_lifetime(self):
  e,io,calls,closed=fixture();first=image(3);e.encode_patches(first);self.assertIs(calls[0],first)
  e.enable_resident_input(io)
  with self.assertRaises(ValueError):e.enable_resident_input(io)
  e.close();self.assertTrue(closed);self.assertIsNone(e.resident_io);self.assertEqual(e.resident_input_bytes,0)
  self.assertEqual(io.zeros([1,2,3])['values'],[0]*6)
if __name__=='__main__':unittest.main()
