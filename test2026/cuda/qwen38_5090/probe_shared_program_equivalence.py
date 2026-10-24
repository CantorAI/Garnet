"""Separate-process same-profile full decoder comparison; no performance claim."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity

root,out=map(Path,sys.argv[1:3]);mode=sys.argv[3]
assert mode in ('embedded','external');out.mkdir(exist_ok=True)
path=out/(mode+'-report.json')
if path.exists():raise FileExistsError('Preserve prior equivalence attempt')
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_text())
report={'status':'starting','mode':mode,'source':source_identity(),'cases':[],
 'scope':'Same B1/T128/cap4096 full64 decoder, identical saved real text/image inputs and zero initial state. Full packed output chunk hashes and repeat equality. Not generation quality, MTP or timing.'}
io=garnet.cuda_tensor_io();program=vision=None
def save():path.write_text(json.dumps(report,indent=2))
save()
try:
 io.open(str(checkpoint));inputs={}
 for case in ('text','image'):
  stem='s3-'+case+'-c1-r0-greedy';oracle=root/'results/native-mtp3-http'
  response=json.loads((oracle/(stem+'-response.json')).read_text());ids=response['choices'][0]['prompt_token_ids']
  saved=out/(case+'-input.f32')
  if mode=='embedded':
   hidden=io.zeros([1,128,5120]);embedding=io.embedding(ids);visual=None;start=-1
   if case=='image':
    body=json.loads((oracle/(stem+'-request.json')).read_text());content=body['messages'][0]['content']
    data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1]);image=out/'input.jpg';image.write_bytes(data)
    vision=create_cuda_vision(str(checkpoint),config,16,16)
    prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
    visual=vision.encode_patches(prepared[0]);start=ids.index(config['image_token_id'])
   for i in range(len(ids)):io.write_chunk_token(io.visual_token(visual,i-start) if start<=i<start+64 and visual is not None else io.row(embedding,i),hidden,i)
   saved.write_bytes(bytes(io.finite_f32_bytes(hidden)))
   if vision is not None:vision.close();vision=None
  else:hidden=io.read_f32(str(saved),[1,128,5120])
  positions,_=image_positions(len(ids),ids.index(config['image_token_id']),64,8) if case=='image' else ([[i]*3 for i in range(len(ids))],len(ids))
  inputs[case]=[hidden,None,io.from_host([1.0]*len(ids)+[0.0]*(128-len(ids)),[1,128]),
    io.from_host([float(v) for row in positions for v in row]+[0.0]*(3*(128-len(ids))),[1,128,3]),
    io.from_host([float(i) for i in range(len(ids))]+[0.0]*(128-len(ids)),[1,128])]
  report.setdefault('inputs',{})[case]={'sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),'prompt_ids':ids}
 report['status']='constructing';save()
 program=CUDAPackedDecoderProgram(str(checkpoint),config['text_config'],io,1,4096,128,flat_concat=True,
     embedded_projection_weights=mode=='embedded',external_compressed_weights=mode=='external')
 state=io.zeros([1,program.state_width])
 for case,values in inputs.items():
  values[1]=state;baseline=None
  for repeat in range(3):
   packed=program.run(values);chunks=[]
   for offset in range(0,program.output_width,1048576):
    width=min(1048576,program.output_width-offset)
    raw=bytes(io.finite_f32_bytes(io.slice(packed,offset,[1,width])))
    chunks.append({'offset':offset,'words':width,'sha256':hashlib.sha256(raw).hexdigest()})
    if offset==0:(out/(mode+'-'+case+'-'+str(repeat)+'-first.f32')).write_bytes(raw)
   if baseline is None:baseline=chunks
   else:assert chunks==baseline,'Same-profile repeat differs'
   report['cases'].append({'case':case,'repeat':repeat,'chunks':chunks});save()
   print('FULL_PROGRAM_HASHED',mode,case,repeat,len(chunks),flush=True)
 report['program']=program.memory_info();report['pool']=json.loads(io.memory_json())
 assert report['program']['graph']['graph_capture_failures']==0
 if mode=='external':
  reference=json.loads((out/'embedded-report.json').read_text());assert reference['status']=='passed'
  assert report['inputs']==reference['inputs']
  assert report['cases']==reference['cases'],'Same-profile full packed output differs'
  report['same_profile_full_output_exact']=True
 report['status']='passed';print('FULL_PROGRAM_SAME_PROFILE_PASS',mode,flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:
 save()
 if vision is not None:vision.close()
 if program is not None:program.close()
 io.close()
