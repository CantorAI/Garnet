"""B2 prefill row-isolation diagnostics; not B1 equivalence or timing."""
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
report={'status':'starting','source':source_identity(),'batch':batch,'capacity':4096,'chunk':16,
 'garnet_dll_sha256':hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest(),
 'cases':[],'oracles':[{k:c[k] for k in ('key','oracle_sha256')} for c in cases],
 'scope':'B2 fixed-engine row isolation/repeat under different companion rows and ragged lengths. Saved B1 first IDs are diagnostic only; the original full B1-equivalence gate remains failed. No serving admission, quality or speed claim.'}
path.write_text(json.dumps(report,indent=2));decoder=None;vision=None
try:
 decoder=create_cuda_batch_study_decoder(str(checkpoint),config,4096,batch)
 decoder.enable_cuda_graph(True);io=decoder.io
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 if batch!=2:raise ValueError('Prefill controls require B2')
 groups=[cases[:2],[cases[0],cases[0]],[cases[0],cases[2]],[cases[1],cases[0]],[cases[1],cases[1]]]
 snapshots={}
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
   raw=bytes(io.finite_f32_bytes(decoder.last_hidden));name='group-'+str(group_index)+'-repeat-'+str(repeat)
   (out/(name+'-last-hidden.f32')).write_bytes(raw)
   normalized=bytes(io.finite_f32_bytes(decoder.head.run_device([decoder.last_hidden])))
   (out/(name+'-normalized.f32')).write_bytes(normalized)
   row_bytes=config['text_config']['hidden_size']*4;snapshots[(group_index,repeat)]=(raw[:row_bytes],normalized[:row_bytes])
   expected=[c['choice']['response_token_ids'][0] for c in group]
   item={'group':group_index,'repeat':repeat,'keys':[c['key'] for c in group],'prompt_lengths':lengths,
    'first_prediction_ids':list(prediction),'expected_b1_first_ids':expected,'b1_first_ids_match':list(prediction)==expected,
    'hidden_sha256':hashlib.sha256(raw).hexdigest(),'normalized_sha256':hashlib.sha256(normalized).hexdigest()}
   report['cases'].append(item);path.write_text(json.dumps(report,indent=2));print('B2_PREFILL_CONTROL_CAPTURED',group_index,repeat,list(prediction),flush=True)
 checks=[]
 for left,right,label in [((0,0),(1,0),'text with image versus text companion'),((0,0),(2,0),'text with long versus short image companion'),((3,0),(4,0),'image with text versus image companion')]+[((g,0),(g,1),'repeat group'+str(g)) for g in range(5)]:
  a,b=snapshots[left],snapshots[right];checks.append({'left':left,'right':right,'label':label,'hidden_exact':a[0]==b[0],'normalized_exact':a[1]==b[1]})
 report['row_isolation_checks']=checks;report['within_batch_isolation_pass']=all(c['hidden_exact'] and c['normalized_exact'] for c in checks)
 report['decoder_memory']=decoder.memory_info();report['status']='prefill_controls_completed';path.write_text(json.dumps(report,indent=2))
 assert report['within_batch_isolation_pass'],'Within-B2 row isolation/repeat failed; raw snapshots preserved'
except Exception as error:
 report['status']='failed';report['error']=repr(error);path.write_text(json.dumps(report,indent=2));raise
finally:
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
