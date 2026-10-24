"""Isolated checkpoint prefixes at B1/B2; not a full-model correctness gate."""
import hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from test2026.cuda.qwen38_5090.run_logged import source_identity

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
capture_internals='--internals' in sys.argv[3:]
prefixes=tuple(int(v) for v in sys.argv[3:] if v!='--internals') or (1,4)
if len(set(prefixes))!=len(prefixes) or any(v not in (1,2,3,4) for v in prefixes):raise ValueError('Diagnostic prefixes must be unique values1..4')
if capture_internals and max(prefixes)>3:raise ValueError('Full internal capture is bounded to GDN-only prefixes1..3')
path=out/'report.json'
if path.exists():raise FileExistsError('Preserve prior prefix diagnostic')
checkpoint=root/'models/Qwen3.8-27B-NVFP4'
config_path=checkpoint/'config.json';original=json.loads(config_path.read_text(encoding='utf8'))
oracle=root/'results/native-mtp3-http/s3-text-c1-r0-greedy-response.json'
ids=json.loads(oracle.read_text(encoding='utf8'))['choices'][0]['prompt_token_ids']
assert len(ids)==41
report={'status':'starting','source':source_identity(),'capacity':4096,'chunk':16,'prefixes':prefixes,'capture_internals':capture_internals,'cases':[],
 'input_prompt_ids':ids,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
 'oracle_sha256':hashlib.sha256(oracle.read_bytes()).hexdigest(),
 'scope':'Diagnostic truncated checkpoint prefixes1..4, identical text rows at B1/B2; no full-model IDs, VLM, serving, quality or timing claim. Original64-layer batch gate remains failed.'}
decoder=None;frames={}

def save_frame(io,value,prefix,batch,repeat,chunk,label):
 for row in range(batch):
  raw=bytes(io.finite_f32_bytes(io.row(value,row)))
  name='prefix-'+str(prefix)+'-batch-'+str(batch)+'-repeat-'+str(repeat)+'-chunk-'+str(chunk)+'-'+label+'-row-'+str(row)+'.f32'
  (out/name).write_bytes(raw);frames[(prefix,batch,repeat,chunk,label,row)]=name
  report.setdefault('internal_files',[]).append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
try:
 for prefix in prefixes:
  snapshots={}
  config=dict(original['text_config']);config['num_hidden_layers']=prefix;config['layer_types']=config['layer_types'][:prefix]
  for batch in (1,2):
   decoder=CUDAPackedDecoder(str(checkpoint),config,batch,4096,16,16,flat_concat=True,embedded_projection_weights=True)
   decoder.enable_cuda_graph(True);io=decoder.io;embeddings=io.embedding(ids)
   hidden=[io.stack_rows([io.row(embeddings,t)]*batch) for t in range(len(ids))]
   captures=[]
   for repeat in range(2):
    decoder.reset();decoder.prefill_mode=True
    for start in range(0,len(ids),16):
     end=min(start+16,len(ids));decoder.chunk_hidden(hidden[start:end],[[1]*batch for _ in range(start,end)],
       [[[t]*3 for t in range(start,end)] for _ in range(batch)],False)
     if capture_internals:
      save_frame(io,io.storage_view(decoder.hidden,[batch,16*5120]),prefix,batch,repeat,start,'hidden')
      for layer in decoder.programs[16].layout:
       assert layer['kind']=='linear_attention'
       width=layer['state1_width']+layer['state2_width']
       if width>1048576:raise ValueError('Internal per-row dump exceeds bounded F32 reader')
       part=io.zeros([batch,width]);io.slice_into(decoder.state,layer['offset'],part)
       save_frame(io,part,prefix,batch,repeat,start,'layer-'+str(layer['index'])+'-state')
    decoder.prefill_mode=False
    raw=bytes(io.finite_f32_bytes(decoder.last_hidden));assert len(raw)==batch*5120*4
    name='prefix-'+str(prefix)+'-batch-'+str(batch)+'-repeat-'+str(repeat)+'-hidden.f32'
    (out/name).write_bytes(raw);captures.append(raw)
   rowbytes=5120*4;snapshots[batch]=captures[0][:rowbytes]
   item={'prefix':prefix,'batch':batch,'repeat_exact':captures[0]==captures[1],
     'duplicate_rows_exact':batch==1 or captures[0][:rowbytes]==captures[0][rowbytes:],
     'hidden_sha256':hashlib.sha256(captures[0]).hexdigest(),'memory':decoder.memory_info()}
   report['cases'].append(item);path.write_text(json.dumps(report,indent=2))
   assert item['repeat_exact'] and item['duplicate_rows_exact'],'Prefix repeat/row isolation failed; arrays preserved'
   decoder.close();decoder=None
  left,right=struct.unpack('<5120f',snapshots[1]),struct.unpack('<5120f',snapshots[2])
  comparison={'prefix':prefix,'hidden_exact':snapshots[1]==snapshots[2],
    'max_abs':max(abs(a-b) for a,b in zip(left,right)),
    'relative_l2':math.sqrt(sum((a-b)**2 for a,b in zip(left,right))/max(sum(b*b for b in right),1e-30))}
  report.setdefault('b1_b2_comparisons',[]).append(comparison)
  if capture_internals:
   for key,name in list(frames.items()):
    p,b,r,chunk,label,row=key
    if p!=prefix or b!=1 or r!=0:continue
    raw=(out/name).read_bytes()
    same_profile=True
    for batch in (1,2):
     first=(out/frames[(prefix,batch,0,chunk,label,0)]).read_bytes()
     for repeat in (0,1):
      for row in range(batch):
       same_profile &= first==(out/frames[(prefix,batch,repeat,chunk,label,row)]).read_bytes()
    other=(out/frames[(prefix,2,0,chunk,label,0)]).read_bytes()
    check={'prefix':prefix,'chunk_start':chunk,'part':label,'bytes_per_row':len(raw),
      'same_profile_repeat_and_duplicate_rows_exact':same_profile,'b1_b2_exact':raw==other}
    report.setdefault('internal_comparisons',[]).append(check)
    path.write_text(json.dumps(report,indent=2))
    assert same_profile,'Internal repeat/duplicate-row instability; raw arrays preserved'
  path.write_text(json.dumps(report,indent=2));print('PREFIX_BATCH_DIAGNOSTIC',json.dumps(comparison),flush=True)
 report['status']='diagnostic_completed'
except Exception as error:
 report['status']='failed';report['error']=repr(error);raise
finally:
 path.write_text(json.dumps(report,indent=2))
 if decoder is not None:decoder.close()
