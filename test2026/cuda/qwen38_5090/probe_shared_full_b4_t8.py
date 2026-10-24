"""Unrun physical B4 T8 correctness candidate; no serving or timing admission."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_batch_study_decoder,create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity
root=Path(sys.argv[1]);batch=int(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(exist_ok=True)
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
report={'status':'starting','source':source_identity(),'batch':batch,'capacity':4096,'chunk':8,
 'garnet_dll_sha256':hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest(),
 'cases':[],'oracles':[{k:c[k] for k in ('key','oracle_sha256')} for c in cases],
 'scope':'Unqualified physical B4 T8 packed mixed backend; flattened M32 geometry alone is not numerical admission, ragged text/image/early EOS against retained B1 full IDs. Inputs use saved oracle prompt IDs; no tokenizer frontend, HTTP, MTP, broad quality or throughput claim.'}
path.write_text(json.dumps(report,indent=2));decoder=None;vision=None
try:
 assert batch == 4 and os.environ.get('GARNET_FP8_FULL_ATTN_M32_CANONICAL') == '1'
 from xModel.qwen38.vl_27b.model import validate_cuda_batch_study_profile
 from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
 validate_cuda_batch_study_profile(config,4096,batch)
 decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],batch,4096,8,8,
     flat_concat=True,external_compressed_weights=True)
 decoder.io.enable_nvfp4_head(True)
 pool=json.loads(decoder.io.memory_json()); report['external_compressed_pool']=pool
 assert pool['compressed_projection_pool_entries']==400 and pool['compressed_nvfp4_entries']==192 and pool['compressed_fp8_entries']==208
 physical=pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']
 for program in decoder.programs.values():
  stats=json.loads(program.graph.stats_json())
  assert stats['native_nvfp4_projection_weights']==192 and stats['native_fp8_projection_weights']==208
  assert stats['external_bf16_weight_inputs']==0 and stats['external_compressed_weight_inputs']==400 and stats['external_compressed_weight_bytes']==physical
 report['external_compressed_weights']=True;path.write_text(json.dumps(report,indent=2))
 decoder.enable_cuda_graph(True);io=decoder.io
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 groups=[cases] if batch==4 else [cases[:2],cases[2:]]
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
   prediction=decoder.prefill_sequence(hidden,positions,lengths);generated=[[] for _ in rows];active=[1]*batch;current=[r['current'] for r in rows]
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
    'generated_ids':generated,'expected_ids':expected,'exact_b1_ids_match':generated==expected,'decoder_memory':decoder.memory_info()}
   report['cases'].append(result);path.write_text(json.dumps(report,indent=2))
   assert generated==expected,'Batched greedy output differs from B1 oracle; raw IDs saved'
   print('BATCH_BACKEND_FULL_IDS_PASS',batch,group_index,repeat,flush=True)
 report['status']='backend_smoke_completed';report['vision_memory']=json.loads(vision.graph.stats_json());path.write_text(json.dumps(report,indent=2))
except Exception as error:
 report['status']='failed';report['error']=repr(error);path.write_text(json.dumps(report,indent=2));raise
finally:
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
