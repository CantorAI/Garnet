"""Actual4B through current Model API; exploratory smoke, not matched benchmarking."""
import os,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from runtime.cuda_serving import Model
from test2026.cuda.qwen38_5090.run_logged import source_identity
root=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else root/'results/qwen35-4b-native-smoke';out.mkdir(exist_ok=True)
report_path=out/'report.json'
if report_path.exists():raise ValueError('Preserve existing4B smoke')
checkpoint=root/'models/Qwen3.5-4B';image=root/'qwen35-4b-smoke.jpg'
profile={'batch':1,'maxInputTokens':3968,'maxOutputTokens':128,'imagePixels':65536,'imageMinPixels':65536,'imageGrid':[16,16],
 'prefillChunk':16,'decodeChunk':16,'fusedDelta':True,'fusedAttention':True,'cudaGraph':True}
report={'status':'starting','source':source_identity(),'profile':profile,'checkpoint_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
 'scope':'Native4B text/image/repeat/long-input smoke with desktop GPU co-residency. Internal API times retained raw, not HTTP benchmark or matched model comparison.',
 'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [checkpoint/'config.json',image,root/'build/bin/garnet.dll']},'cases':[]}
origin=root/'results/ct-int4-capture-binary.json'
if origin.is_file():
 candidate=json.loads(origin.read_text())
 if candidate['garnet_dll_sha256']==report['files'][str(root/'build/bin/garnet.dll')]:report['compiled_binary_origin']=candidate
report_path.write_text(json.dumps(report,indent=2));model=None
try:
 model=Model(str(checkpoint),profile,'Qwen3.5-4B')
 assert model.capacity==4096 and len(model.decoder.layers)==32 and len(model.vision.graph.stats_json())>0
 report['loaded_memory']=model.memory_info();report['status']='loaded';report_path.write_text(json.dumps(report,indent=2))
 cases=[('text','What is 2 + 2? Answer with the number only.',''),('image','What is the dominant color in this image? Answer with one color word.',str(image)),
        ('text-repeat','What is 2 + 2? Answer with the number only.',''),('image-repeat','What is the dominant color in this image? Answer with one color word.',str(image))]
 for key,prompt,path in cases:
  result=model.infer(prompt,path,16)
  assert result['status']=='ok' and result['prompt_tokens'][0]+16<=4096 and result['sampled_tokens_including_stop'][0]>0
  if path:assert result['visual_tokens_per_row']==[64]
  normalized=result['text'].strip().lower().strip('.!')
  item={'key':key,'prompt':prompt,'image_sha256':report['files'].get(str(image)) if path else None,'result':result,'basic_semantic_check':normalized==('red' if path else '4')}
  report['cases'].append(item);report_path.write_text(json.dumps(report,indent=2));print('ACTUAL_4B_SMOKE',key,json.dumps({'text':result['text'],'sampled':result['sampled_tokens_including_stop'],'basic_semantic_check':item['basic_semantic_check']}),flush=True)
 assert report['cases'][0]['result']['generated_token_ids']==report['cases'][2]['result']['generated_token_ids']
 assert report['cases'][1]['result']['generated_token_ids']==report['cases'][3]['result']['generated_token_ids']
 # Actual near-capacity input, retaining early EOS instead of manufacturing length.
 suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
 def length(n):return len(model.tokenizer.encode('<|im_start|>user\n'))+len(model.tokenizer.encode('x '*n+suffix))
 lo,hi=0,3968
 while lo<hi:
  mid=(lo+hi+1)//2
  if length(mid)<=3968:lo=mid
  else:hi=mid-1
 assert length(lo)==3968
 prompt='x '*lo;result=model.infer(prompt,'',128);assert result['prompt_tokens']==[3968]
 report['cases'].append({'key':'long-input','prompt':prompt,'result':result,'scope':'3968 actual input tokens plus128 allowed outputs at4096 capacity; EOS/actual counts retained, not a claim of4096 occupied KV slots'})
 report['final_memory']=model.memory_info();report['repeat_ids_match']=True;report['basic_semantics_all_pass']=all(c['basic_semantic_check'] for c in report['cases'] if 'basic_semantic_check' in c)
 report['status']='smoke_completed';report['independent_quality_gate']='pending4B reference/expanded quality';report_path.write_text(json.dumps(report,indent=2))
 print('ACTUAL_4B_TEXT_IMAGE_REPEAT_AND_3968_INPUT_SMOKE_COMPLETE',flush=True)
except Exception as error:
 report['status']='failed';report['error']=repr(error);report_path.write_text(json.dumps(report,indent=2));raise
finally:
 if model is not None:model.close()
