import sys,os,json,time,pathlib,hashlib
build,source,label=sys.argv[1:4]
sys.path.insert(0,build)
import garnet
sys.path.insert(0,source)
from runtime.metal_serving import Model
import runtime.metal_serving as serving
assert str(pathlib.Path(serving.__file__).resolve()).startswith(source)
profile={'maxInputTokens':1536,'maxOutputTokens':256,'imagePixels':65536,'prefillChunk':16,'linearChunk':True}
start=time.monotonic();model=Model('/Users/shawn/CantorAI/out/models/Qwen3.5-4B',source+'/xModel/qwen35/vl_4b','/Users/shawn/Documents/Codex/2026-10-15/i/work/4b-shape/cache-'+label,profile,'Qwen3.5-4B')
print(json.dumps({'label':label,'load_seconds':time.monotonic()-start,'backend':garnet.backend_info(),'library_sha256':hashlib.sha256(pathlib.Path(build,'libgarnet.dylib').read_bytes()).hexdigest(),'runtime_source':serving.__file__}),flush=True)
prompt='Return only a compact JSON object with description (one factual sentence), entities (at most three visible objects), rabbit_visible (boolean), car_visible (boolean), elephant_visible (boolean). Do not describe absent objects.'
for i,image in enumerate(['bunny','car','traffic','bunny']):
 result=model.infer(prompt,'/Users/shawn/Documents/Codex/2026-10-15/i/work/skill-expansion/'+image+'.jpg',180)
 shape_count=model.decoder.session.body_shape_dispatches
 misses=model.decoder.session.body_shape_misses
 if label.startswith('candidate'):
  assert shape_count==248 and misses==0,(shape_count,misses)
 else:assert shape_count==0
 result['body_shape_dispatches']=shape_count
 result['body_shape_misses']=misses
 result['body_shape_pipeline_count']=model.decoder.session.body_shape_pipeline_count
 print(json.dumps({'first_after_init':i==0,'image':image,'call':result}),flush=True)
