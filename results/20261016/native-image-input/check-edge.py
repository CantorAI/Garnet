import sys,json,pathlib,struct,time
sys.path.insert(0,sys.argv[1])
import garnet
r=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/native-image');options={'patch_size':16,'merge_size':2,'temporal_patch_size':2,'pixels':4096,'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]};records=[]
def flat(v):
 for x in v:
  if isinstance(x,list):yield from flat(x)
  else:yield x
for case in json.loads((r/'references.json').read_text()):
 x=garnet.prepare_image(case['image'],json.dumps(options));a=b''.join(struct.pack('<f',v) for v in flat(x[0].tolist()));b=pathlib.Path(case['reference']).read_bytes();equal=a==b and list(x[1:3])==case['grid'];print(pathlib.Path(case['image']).name,equal,flush=True);records.append({'image':pathlib.Path(case['image']).name,'bitwise_equal':equal,'bytes':len(a)})
for name in ['bad.jpg','truncated.jpg']:
 try:garnet.prepare_image(str(r/'fixtures'/name),json.dumps(options))
 except Exception:records.append({'invalid':name,'rejected':True})
 else:raise AssertionError('Corrupt image admitted')
bad=dict(options);bad['image_std']=[0,.5,.5]
try:garnet.prepare_image(str(r/'fixtures/rgb.png'),json.dumps(bad))
except Exception:records.append({'invalid':'zero normalization std','rejected':True})
else:raise AssertionError('Invalid normalization admitted')
(r/'edge-checks.json').write_text(json.dumps(records,indent=2));assert all(x.get('bitwise_equal',True)for x in records)
