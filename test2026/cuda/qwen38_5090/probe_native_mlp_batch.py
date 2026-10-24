"""Same actual embedding rows through isolated norm/MLP at B1/B2; no timing."""
import hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,mlp
from test2026.cuda.qwen38_5090.run_logged import source_identity

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous norm/MLP diagnostic')
checkpoint=root/'models/Qwen3.8-27B-NVFP4'
config=json.loads((checkpoint/'config.json').read_text(encoding='utf8'))['text_config']
oracle=root/'results/native-mtp3-http/s3-text-c1-r0-greedy-response.json'
ids=json.loads(oracle.read_text(encoding='utf8'))['choices'][0]['prompt_token_ids'][:16]
report={'status':'starting','source':source_identity(),'cases':[],'input_ids':ids,
 'oracle_sha256':hashlib.sha256(oracle.read_bytes()).hexdigest(),
 'scope':'Isolated layer0 post-attention norm/MLP on actual prompt embeddings, not actual post-GDN activations, full model, accuracy oracle or timing. Batch-invariance diagnostic only.'}
io=garnet.cuda_tensor_io();graph=None
try:
 io.open(str(checkpoint));embeddings=io.embedding(ids);flat=io.storage_view(embeddings,[1,16*5120])
 for kind in ('norm','mlp'):
  snapshots={}
  for batch in (1,2):
   value=io.storage_view(io.stack_rows([flat]*batch),[batch,16,5120])
   @T.fusion(name='qwen38_study_norm_mlp_batch',boundary='required')
   def forward(x):
    y=norm(x,'model.language_model.layers.0.post_attention_layernorm',config)
    return mlp(y,'model.language_model.layers.0.mlp') if kind=='mlp' else y
   symbol=tensor.input('mlp_batch_hidden',shape=[batch,16,5120],dtype=tensor.float32)
   graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(256<<20)
   graph.build(forward(symbol),[symbol],str(checkpoint));graph.enable_cuda_graph(True)
   raws=[]
   for repeat in range(3):
    raw=bytes(io.finite_f32_bytes(graph.run_device([value])));raws.append(raw)
    (out/(kind+'-batch-'+str(batch)+'-repeat-'+str(repeat)+'.f32')).write_bytes(raw)
   rowbytes=16*5120*4
   item={'kind':kind,'batch':batch,'repeat_exact':len(set(raws))==1,
      'duplicate_rows_exact':batch==1 or raws[0][:rowbytes]==raws[0][rowbytes:],
      'graph':json.loads(graph.stats_json()),'sha256':hashlib.sha256(raws[0]).hexdigest()}
   report['cases'].append(item);path.write_text(json.dumps(report,indent=2))
   assert item['repeat_exact'] and item['duplicate_rows_exact'],'Norm/MLP repeat/row isolation failed'
   if kind=='mlp':assert item['graph']['native_nvfp4_projection_weights']==3 and item['graph']['external_bf16_weight_inputs']==0,'Actual mixed MLP lowering required'
   snapshots[batch]=raws[0][:rowbytes];graph.close();graph=None
  a,b=struct.unpack('<81920f',snapshots[1]),struct.unpack('<81920f',snapshots[2])
  comparison={'kind':kind,'exact':snapshots[1]==snapshots[2],
    'max_abs':max(abs(x-y) for x,y in zip(a,b)),
    'relative_l2':math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(y*y for y in b),1e-30))}
  report.setdefault('comparisons',[]).append(comparison);path.write_text(json.dumps(report,indent=2))
  print('NORM_MLP_BATCH_DIAGNOSTIC',json.dumps(comparison),flush=True)
 report['status']='diagnostic_completed'
except Exception as error:
 report['status']='failed';report['error']=repr(error);raise
finally:
 path.write_text(json.dumps(report,indent=2))
 if graph is not None:graph.close()
 io.close()
