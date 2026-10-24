"""Residual+MLP branch batch-shape diagnostic on recorded/derived inputs; no timing."""
import hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,mlp,add
from test2026.cuda.qwen38_5090.run_logged import source_identity

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous residual/MLP diagnostic')
checkpoint=root/'models/Qwen3.8-27B-NVFP4'
config=json.loads((checkpoint/'config.json').read_text(encoding='utf8'))['text_config']
oracle=root/'results/native-mtp3-http/s3-text-c1-r0-greedy-response.json'
ids=json.loads(oracle.read_text(encoding='utf8'))['choices'][0]['prompt_token_ids'][:16]
report={'status':'starting','source':source_identity(),'cases':[],'input_ids':ids,
 'oracle_sha256':hashlib.sha256(oracle.read_bytes()).hexdigest(),
 'scope':'Residual plus native MLP branch on real embeddings and element-rolled recorded hidden as a diagnostic second operand. Not actual attention output/residual activations, full model, accuracy oracle or timing.'}
io=garnet.cuda_tensor_io();graph=None
try:
 io.open(str(checkpoint));embeddings=io.embedding(ids);flat=io.storage_view(embeddings,[1,16*5120])
 fixture=root/'results/native-prefix12-internals/prefix-1-batch-1-repeat-0-chunk-0-hidden-row-0.f32'
 raw_input=fixture.read_bytes();words=struct.unpack('<81920f',raw_input)
 attention_values=[words[(i+13)%len(words)] for i in range(len(words))]
 attention_raw=struct.pack('<81920f',*attention_values);(out/'derived-attention-input.f32').write_bytes(attention_raw)
 report['fixture_sha256']=hashlib.sha256(raw_input).hexdigest();report['derived_input_sha256']=hashlib.sha256(attention_raw).hexdigest()
 report['second_operand_recipe']='Recorded first-layer hidden, flattened element roll13; synthetic diagnostic attention operand'
 attention_flat=io.from_host(attention_values,[1,16*5120]);all_snapshots={}
 for kind in ('fused','residual_barrier'):
  snapshots={}
  for batch in (1,2):
   value=io.storage_view(io.stack_rows([flat]*batch),[batch,16,5120])
   attention=io.storage_view(io.stack_rows([attention_flat]*batch),[batch,16,5120])
   @T.fusion(name='qwen38_study_residual_mlp_batch',boundary='required')
   def forward(x,attention):
    residual=add(x,attention)
    if kind=='residual_barrier':residual=residual*T.unary_op('qwen35_f32_barrier')
    y=mlp(norm(residual,'model.language_model.layers.0.post_attention_layernorm',config),'model.language_model.layers.0.mlp')
    return add(residual,y)
   symbol=tensor.input('mlp_batch_hidden',shape=[batch,16,5120],dtype=tensor.float32)
   other=tensor.input('residual_batch_attention',shape=[batch,16,5120],dtype=tensor.float32)
   graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(256<<20)
   graph.build(forward(symbol,other),[symbol,other],str(checkpoint));graph.enable_cuda_graph(True)
   raws=[]
   for repeat in range(3):
    raw=bytes(io.finite_f32_bytes(graph.run_device([value,attention])));raws.append(raw)
    (out/(kind+'-batch-'+str(batch)+'-repeat-'+str(repeat)+'.f32')).write_bytes(raw)
   rowbytes=16*5120*4
   item={'kind':kind,'batch':batch,'repeat_exact':len(set(raws))==1,
      'duplicate_rows_exact':batch==1 or raws[0][:rowbytes]==raws[0][rowbytes:],
      'graph':json.loads(graph.stats_json()),'sha256':hashlib.sha256(raws[0]).hexdigest()}
   report['cases'].append(item);path.write_text(json.dumps(report,indent=2))
   assert item['repeat_exact'] and item['duplicate_rows_exact'],'Norm/MLP repeat/row isolation failed'
   assert item['graph']['native_nvfp4_projection_weights']==3 and item['graph']['external_bf16_weight_inputs']==0,'Actual mixed MLP lowering required'
   snapshots[batch]=raws[0][:rowbytes];all_snapshots[(kind,batch)]=snapshots[batch];graph.close();graph=None
  a,b=struct.unpack('<81920f',snapshots[1]),struct.unpack('<81920f',snapshots[2])
  comparison={'kind':kind,'exact':snapshots[1]==snapshots[2],
    'max_abs':max(abs(x-y) for x,y in zip(a,b)),
    'relative_l2':math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(y*y for y in b),1e-30))}
  report.setdefault('comparisons',[]).append(comparison);path.write_text(json.dumps(report,indent=2))
  print('RESIDUAL_MLP_BATCH_DIAGNOSTIC',json.dumps(comparison),flush=True)
 report['barrier_changes_each_shape']={str(b):all_snapshots[('fused',b)]!=all_snapshots[('residual_barrier',b)] for b in (1,2)}
 report['status']='diagnostic_completed'
except Exception as error:
 report['status']='failed';report['error']=repr(error);raise
finally:
 path.write_text(json.dumps(report,indent=2))
 if graph is not None:graph.close()
 io.close()
