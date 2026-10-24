"""Fresh B1 retained-output configuration control; diagnostic, not timing."""
import base64,hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_packed_decoder,create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity
root=Path(sys.argv[1]);retain=int(sys.argv[2]);assert retain in (0,1);batch=1;out=Path(sys.argv[3]);out.mkdir(exist_ok=True)
path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous batch qualification')
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_text(encoding='utf8'))
oracle=root/'results/native-mtp3-http'
keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green']
cases=[]
for key in keys:
 request=oracle/(key+'-request.json');response=oracle/(key+'-response.json')
 body=json.loads(request.read_text(encoding='utf8'));choice=json.loads(response.read_text(encoding='utf8'))['choices'][0]
 cases.append({'key':key,'body':body,'choice':choice,'oracle_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (request,response)}})
report={'status':'starting','source':source_identity(),'batch':batch,'retain_linear_inputs':bool(retain),'capacity':4096,'chunk':16,
 'garnet_dll_sha256':hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest(),
 'cases':[],'oracles':[{k:c[k] for k in ('key','oracle_sha256')} for c in cases],
 'scope':'Fresh B1 packed backend with explicit retained-output configuration. Saved oracle IDs and B2 raw hidden comparisons are diagnostic; no broad quality, HTTP or timing claim.'}
path.write_text(json.dumps(report,indent=2));decoder=None;vision=None
try:
 decoder=create_cuda_packed_decoder(str(checkpoint),config,4096,retain_linear_inputs=bool(retain))
 decoder.enable_cuda_graph(True);io=decoder.io
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 groups=[[c] for c in cases];repeat_ids={}
 for group_index,group in enumerate(groups):
  for repeat in range(2):
   decoder.reset();rows=[]
   for row,c in enumerate(group):
    ids=c['choice']['prompt_token_ids'];content=c['body']['messages'][0]['content'];visual=None;start=-1
    if isinstance(content,list):
     item=next(x for x in content if x['type']=='image_url');data=base64.b64decode(item['image_url']['url'].split(',',1)[1],validate=True)
     image=root/'tmp'/('batch-study-image-'+str(row)+'.jpg');image.write_bytes(data)
     prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
     visual=io.clone(vision.encode_patches(prepared[0]));start=ids.index(config['image_token_id'])
     assert ids.count(config['image_token_id'])==64
     positions,current=image_positions(len(ids),start,64,8)
    else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
    embeddings=io.embedding(ids)
    hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
    rows.append({'ids':ids,'positions':positions,'current':current,'hidden':hidden,'embeddings':embeddings,'visual':visual})
   count=max(len(r['ids']) for r in rows);zero=io.zeros([1,config['text_config']['hidden_size']])
   hidden=[io.stack_rows([r['hidden'][t] if t<len(r['hidden']) else zero for r in rows]) for t in range(count)]
   lengths=[len(r['ids']) for r in rows];positions=[r['positions']+[[0,0,0]]*(count-len(r['ids'])) for r in rows]
   prediction=decoder.prefill_sequence(hidden,positions,lengths)
   raw=bytes(io.finite_f32_bytes(decoder.last_hidden));normalized=bytes(io.finite_f32_bytes(decoder.head.run_device([decoder.last_hidden])))
   name='group-'+str(group_index)+'-repeat-'+str(repeat)
   (out/(name+'-last-hidden.f32')).write_bytes(raw);(out/(name+'-normalized.f32')).write_bytes(normalized)
   b2_location={0:(0,0),1:(0,1),2:(2,1)}.get(group_index);comparison=None
   if b2_location is not None:
    g,row=b2_location;b2=(root/('results/native-batch2-prefill-controls/group-'+str(g)+'-repeat-0-normalized.f32')).read_bytes()[row*20480:(row+1)*20480]
    left=struct.unpack('<5120f',normalized);right=struct.unpack('<5120f',b2)
    comparison={'normalized_exact':normalized==b2,'max_abs':max(abs(a-b) for a,b in zip(left,right)),
      'relative_l2':math.sqrt(sum((a-b)**2 for a,b in zip(left,right))/max(sum(b*b for b in right),1e-30)),
      'b2_group':g,'b2_row':row,'b2_normalized_sha256':hashlib.sha256(b2).hexdigest()}
   generated=[[] for _ in rows];active=[1]*batch;current=[r['current'] for r in rows]
   limits=[c['body']['max_tokens'] for c in group]
   for step in range(max(limits)):
    for b in range(batch):
     if active[b]:
      generated[b].append(prediction[b])
      if prediction[b] in stops or len(generated[b])==limits[b]:active[b]=0
    if not any(active):break
    prediction=decoder.step(prediction,active,[[p]*3 for p in current])
    current=[p+flag for p,flag in zip(current,active)]
   expected=[c['choice']['response_token_ids'] for c in group]
   result={'group':group_index,'repeat':repeat,'keys':[c['key'] for c in group],'prompt_lengths':lengths,'limits':limits,
    'normalized_sha256':hashlib.sha256(normalized).hexdigest(),'b2_comparison':comparison,'generated_ids':generated,'expected_ids':expected,'exact_b1_ids_match':generated==expected,'decoder_memory':decoder.memory_info()}
   report['cases'].append(result);path.write_text(json.dumps(report,indent=2))
   if repeat==0:repeat_ids[group_index]=generated
   else:assert repeat_ids[group_index]==generated,'Fresh B1 repeat differs; raw IDs saved'
   print('B1_CONFIGURATION_CONTROL',retain,group_index,repeat,generated==expected,flush=True)
 report['status']='configuration_control_completed';report['vision_memory']=json.loads(vision.graph.stats_json());path.write_text(json.dumps(report,indent=2))
except Exception as error:
 report['status']='failed';report['error']=repr(error);path.write_text(json.dumps(report,indent=2));raise
finally:
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
