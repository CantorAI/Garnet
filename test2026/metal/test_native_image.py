# SPDX-License-Identifier: Apache-2.0
"""XLang3 native image parity runner. Offline reference data is an input, not a runtime dependency.
Usage: xlang3 -B test_native_image.py NATIVE_BUILD MODEL_ROOT CASE_MANIFEST OUTPUT_JSON
Each case supplies image/reference paths, pixel budget, and expected grid.
"""
import sys,json,pathlib,struct,hashlib,time
build,model,manifest,output=sys.argv[1:5];sys.path.insert(0,build)
import garnet
root=pathlib.Path(model);config=json.loads((root/'config.json').read_text())['vision_config'];processor=json.loads((root/'preprocessor_config.json').read_text());source=pathlib.Path(manifest);records=[]
def flatten(values):
 for v in values:
  if isinstance(v,list):yield from flatten(v)
  else:yield v
for case in json.loads(source.read_text()):
 image=pathlib.Path(case['image']);reference=pathlib.Path(case['reference'])
 if not image.is_absolute():image=source.parent/image
 if not reference.is_absolute():reference=source.parent/reference
 options={'patch_size':config['patch_size'],'merge_size':config['spatial_merge_size'],'temporal_patch_size':config['temporal_patch_size'],'pixels':case['budget'],'image_mean':processor['image_mean'],'image_std':processor['image_std']}
 started=time.monotonic();frame=garnet.prepare_image(str(image),json.dumps(options));seconds=time.monotonic()-started
 data=b''.join(struct.pack('<f',v)for v in flatten(frame[0].tolist()));expected=reference.read_bytes();assert data==expected and list(frame[1:3])==case['grid'],str(image)
 records.append({'image':image.name,'bitwise_equal':True,'fp32_values':len(data)//4,'sha256':hashlib.sha256(data).hexdigest(),'seconds':seconds})
pathlib.Path(output).write_text(json.dumps({'executor':'XLang3','cases':records},indent=2));print('Native image parity PASS',len(records),flush=True)
