"""Diagnostic native JPEG patches for an independent same-input reference.

Run after Garnet has unloaded, never during measured inference.
"""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
io=garnet.cuda_tensor_io()
prepared=io.prepare_image(sys.argv[1],65536)
data={'image':sys.argv[1],'grid_thw':[1,prepared[1],prepared[2]],
      'patches':garnet.tensor_to_cpu(prepared[0]).tolist(),
      'source':'Garnet native JPEG/NPP, FP32 normalized patches; diagnostic export'}
with open(sys.argv[2],'w') as f:f.write(json.dumps(data))
print('QWEN35_PATCH_EXPORT_COMPLETE',data['grid_thw'],flush=True)
