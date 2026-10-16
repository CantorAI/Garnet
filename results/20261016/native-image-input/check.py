import sys,json,pathlib,struct,hashlib,time,statistics
sys.path.insert(0,sys.argv[1])
import garnet
root=pathlib.Path('/Users/shawn/CantorAI/out/models/Qwen3.5-4B');v=json.loads((root/'config.json').read_text())['vision_config'];p=json.loads((root/'preprocessor_config.json').read_text());options=json.dumps({'patch_size':v['patch_size'],'merge_size':v['spatial_merge_size'],'temporal_patch_size':v['temporal_patch_size'],'pixels':65536,'image_mean':p['image_mean'],'image_std':p['image_std']});records=[]
def flat(v):
 for x in v:
  if isinstance(x,list):yield from flat(x)
  else:yield x
for name in ['bunny','car','traffic']:
 image='/Users/shawn/Documents/Codex/2026-10-15/i/work/skill-expansion/'+name+'.jpg';base=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/4b-vision')/name
 r=garnet.prepare_image(image,options);actual=b''.join(struct.pack('<f',x) for x in flat(r[0].tolist()));expected=base.with_suffix('.f32').read_bytes();a=struct.unpack('<'+str(len(actual)//4)+'f',actual);b=struct.unpack('<'+str(len(expected)//4)+'f',expected);different=sum(x!=y for x,y in zip(a,b));largest=max((abs(x-y)for x,y in zip(a,b)),default=0);print('PARITY',name,list(r[1:3]),len(actual),len(expected),different,largest,flush=True)
 times=[]
 for i in range(10):
  start=time.monotonic();q=garnet.prepare_image(image,options);times.append(time.monotonic()-start)
 records.append({'image':name,'grid':[r[1],r[2]],'values':len(a),'bitwise_equal':actual==expected,'different_values':different,'max_absolute_error':largest,'native_median_seconds':statistics.median(times),'native_seconds':times,'phases_ms':list(r[3:]),'native_sha256':hashlib.sha256(actual).hexdigest(),'reference_sha256':hashlib.sha256(expected).hexdigest()})
pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/native-image/parity.json').write_text(json.dumps(records,indent=2));assert all(x['bitwise_equal']for x in records)
