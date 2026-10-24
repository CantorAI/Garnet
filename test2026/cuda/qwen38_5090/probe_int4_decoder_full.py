"""Full INT4 decoder construction/repeat/VLM smoke; never a timing benchmark."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_int4_decoder,prepare_cuda_image
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity

root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous full INT4 decoder attempt')
checkpoint=root/'models/Qwen3.8-27B-INT4';oracle=root/'results/native-mtp3-http'
config=json.loads((checkpoint/'config.json').read_text(encoding='utf8'))
report={'status':'constructing_decoder','source':source_identity(),'batch':1,'capacity':4096,'prefill_chunk':16,'decode_chunk':16,'cases':[],
 'scope':'Full64 INT4 G128 W4A16 construction, repeat determinism and four VLM semantic smoke cases. Mixed NVFP4 reference IDs are diagnostic only because the checkpoint quantization changes. Not cross-checkpoint precision equivalence, MTP, broad quality or benchmark admission.'}
decoder=vision=None
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save()
try:
 decoder=create_cuda_int4_decoder(str(checkpoint),config,4096)
 decoder.enable_cuda_graph(True);io=decoder.io
 report['status']='constructing_vision';report['decoder_after_build']=decoder.memory_info();save()
 assert sum(g.get('native_int4_g128_projection_weights',0) for g in report['decoder_after_build']['graphs'])==496
 vision=CUDAImageEncoder(str(checkpoint/'model.safetensors.index.json'),config['vision_config'],16,16,1);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 report['status']='checking_cases';save()
 wants={'red':'red','green':'green','blue':'blue','bottom-blue':'blue'}
 for key in ['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green','blue','bottom-blue']:
  request=oracle/(key+'-request.json');response=oracle/(key+'-response.json')
  body=json.loads(request.read_text(encoding='utf8'));prior=json.loads(response.read_text(encoding='utf8'))['choices'][0]
  ids=prior['prompt_token_ids'];limit=body['max_tokens'];baseline=None
  for repeat in range(2):
   decoder.reset();visual=None;content=body['messages'][0]['content'];start=-1
   if isinstance(content,list):
    item=next(x for x in content if x['type']=='image_url');data=base64.b64decode(item['image_url']['url'].split(',',1)[1],validate=True)
    image=root/'tmp/int4-full-image.jpg';image.write_bytes(data)
    prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
    visual=vision.encode_patches(prepared[0]);start=ids.index(config['image_token_id']);assert ids.count(config['image_token_id'])==64
    positions,current=image_positions(len(ids),start,64,8)
   else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
   embeddings=io.embedding(ids)
   hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
   prediction=decoder.prefill_sequence(hidden,[positions]);generated=[]
   normalized=bytes(io.finite_f32_bytes(decoder.head.run_device([decoder.last_hidden])))
   (out/(key+'-'+str(repeat)+'-normalized.f32')).write_bytes(normalized)
   for step in range(limit):
    token=prediction[0];generated.append(token)
    if token in stops or step+1==limit:break
    prediction=decoder.step([token],positions=[[current]*3]);current+=1
   text=tokenizer.decode([t for t in generated if t not in stops])
   row={'key':key,'repeat':repeat,'prompt_ids':ids,'generated_ids':generated,'text':text,'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),
    'prior_response_sha256':hashlib.sha256(response.read_bytes()).hexdigest(),'mixed_nvfp4_reference_ids_equal':generated==prior['response_token_ids'],
    'normalized_sha256':hashlib.sha256(normalized).hexdigest()}
   report['cases'].append(row);save()
   if baseline is None:baseline=(generated,normalized)
   else:assert baseline==(generated,normalized),'Same INT4 profile repeat differs'
   if key in wants:assert text.strip().lower().rstrip('.')==wants[key],'VLM semantic smoke failed'
   else:assert len(generated)==128,'Full128-output diagnostic expected'
   print('INT4_FULL_DECODER_CASE',key,repeat,len(generated),flush=True)
 report['decoder_final']=decoder.memory_info();report['vision_final']=json.loads(vision.graph.stats_json())
 assert all(g['graph_capture_failures']==0 for g in report['decoder_final']['graphs'])
 report['status']='smoke_passed';print('INT4_FULL64_REPEAT_AND_VLM_SMOKE_PASS; no timing admission',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:
 save()
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
