"""Real layer0 B1/B2 stage capture; witness parity precedes attribution.

No model/batch admission, native speed or architecture quality claim.
Native diagnostic flag must first be compiled; never deploy during a live study.
"""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.decoder_layer import T
from test2026.cuda.qwen38_5090.run_logged import source_identity
assert os.environ.get('GARNET_NVFP4_DOWN_CANDIDATE','0')=='0' and not os.environ.get('GARNET_PROFILE_DECODE_LAYERS')
r,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir();path=out/'report.json';witness=r/'results/layer0-stage-witness';saved=json.loads((witness/'manifest.json').read_text());checkpoint=r/'models/Qwen3.8-27B-NVFP4';config_path=checkpoint/'config.json';original=json.loads(config_path.read_text())
assert hashlib.sha256(config_path.read_bytes()).hexdigest()==saved['config_sha256']
for n,h in saved['files'].items():assert hashlib.sha256((witness/n).read_bytes()).hexdigest()==h
c=dict(original['text_config']);c['num_hidden_layers']=1;c['layer_types']=c['layer_types'][:1]
assert c['hidden_size']==5120 and c['layer_types']==['linear_attention'] and c['linear_conv_kernel_dim']==4
ids=saved['input_prompt_ids'];assert len(ids)==41;ids=ids[:16]
normal_width=16*5120+40960+786432
stages=[('masked_input',81920),('input_norm',81920),('qkv',163840),('z',98304),('beta',768),('a',768),('gdn_gated_output',98304),('attention_output',81920),('residual',81920),('mlp_input_norm',81920),('output_hidden',81920)]
report={'status':'starting','source':source_identity(),'config_sha256':saved['config_sha256'],'historical_archive_sha256':saved['archive_sha256'],'input_ids':ids,'stages':stages,'files':[],'cases':[],'scope':'Two prefix1 B1/B2 profiles, actual first16 prompt embeddings, zero initial state. Primary output/state must match ordinary prefix graph before stage attribution. Additional exports may change TensorRT fusion; failure is retained and disqualifies attribution. No timing/fullmodel/batch2 or2x admission.'}
frames={};decoder=graph=None

def save():path.write_text(json.dumps(report,indent=2))
def capture(io,value,batch,repeat,label):
 for row in range(batch):
  raw=bytes(io.finite_f32_bytes(io.row(value,row)));name='b'+str(batch)+'-r'+str(repeat)+'-'+label+'-row'+str(row)+'.f32';(out/name).write_bytes(raw);frames[(batch,repeat,label,row)]=raw;report['files'].append({'name':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
def make_forward(flag):
 @T.fusion(name='qwen35_layer0_real_stages',boundary='required')
 def forward(hidden,history,state,active):
  bound=hidden*T.binary_op('qwen35_conv_bind_state')*history
  bound=bound*T.binary_op('qwen35_delta_bind_state')*state
  bound=bound*T.binary_op('qwen35_conv_bind_active')*active
  return bound*T.unary_op('qwen35_linear_decoder_chunk',layer_prefix='model.language_model.layers.0',key_heads=16,value_heads=48,key_dim=128,value_dim=128,eps=c['rms_norm_eps'],fused_delta=1,diagnostic_linear_stages=flag)
 return forward
save()
try:
 for batch in (1,2):
  decoder=CUDAPackedDecoder(str(checkpoint),c,batch,4096,16,16,flat_concat=True,embedded_projection_weights=True);decoder.enable_cuda_graph(True);io=decoder.io
  embedded=io.embedding(ids);tokens=[io.stack_rows([io.row(embedded,t)]*batch) for t in range(16)]
  for repeat in range(2):
   decoder.reset();decoder.chunk_hidden(tokens,[[1]*batch for _ in range(16)],[[[t]*3 for t in range(16)] for _ in range(batch)],False)
   capture(io,decoder.last_packed,batch,repeat,'ordinary')
  hidden=io.clone(decoder.hidden);history=io.zeros([batch,10240,4]);state=io.zeros([batch,48,128,128]);active=io.from_host([1.0]*(batch*16),[batch,16])
  symbols=[tensor.input('stage_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(([batch,16,5120],[batch,10240,4],[batch,48,128,128],[batch,16]))]
  if batch==1:
   rejected=[]
   for flag,bad_batch,bad_time in ((2,1,16),(-1,1,16),(1,4,16),(1,1,4)):
    bad_symbols=[tensor.input('invalid_stage_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(([bad_batch,bad_time,5120],[bad_batch,10240,4],[bad_batch,48,128,128],[bad_batch,bad_time]))]
    invalid=garnet.compiled_graph();failed=False;reason=None
    try:invalid.build(make_forward(flag)(*bad_symbols),bad_symbols,str(checkpoint))
    except Exception as error:failed=True;reason=repr(error)
    finally:invalid.close()
    rejected.append({'flag':flag,'batch':bad_batch,'time':bad_time,'rejected':failed,'error':reason});report['invalid_profiles']=rejected;save()
    assert failed,'Native binary ignored or accepted unsupported stage diagnostic profile'
  graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(256<<20);graph.build(make_forward(1)(*symbols),symbols,str(checkpoint));graph.enable_cuda_graph(True)
  for repeat in range(2):
   result=graph.run_device([hidden,history,state,active]);offset=0
   for label,width in [('primary',normal_width)]+stages:
    part=io.zeros([batch,width]);io.slice_into(result,offset,part);capture(io,part,batch,repeat,label);offset+=width
   save()
  stats=json.loads(graph.stats_json());report.setdefault('native_projection_counts',[]).append({'batch':batch,'stats':stats});save()
  assert stats['checkpoint_weight_precision']=='modelopt_mixed' and stats['native_nvfp4_projection_weights']==3 and stats['native_fp8_projection_weights']==3 and stats['native_int4_g128_projection_weights']==0,'Diagnostic projection precision/count changed'
  assert stats['device_output_bytes']==batch*(normal_width+sum(width for _,width in stages))*4 and stats['graph_capture_failures']==0 and stats['graph_captures']==1 and stats['graph_replays']>=1,'Diagnostic output/graph contract differs'
  ordinary_memory=decoder.memory_info();assert any(g['native_nvfp4_projection_weights']==3 and g['native_fp8_projection_weights']==3 for g in ordinary_memory['graphs']),'Ordinary prefix precision/count differs'
  rows=[]
  for row in range(batch):
   ordinary=frames[(batch,0,'ordinary',row)];primary=frames[(batch,0,'primary',row)];old_hidden=witness/('prefix-1-batch-'+str(batch)+'-repeat-0-chunk-0-hidden-row-'+str(row)+'.f32');old_state=witness/('prefix-1-batch-'+str(batch)+'-repeat-0-chunk-0-layer-0-state-row-'+str(row)+'.f32')
   rows.append({'row':row,'primary_matches_ordinary':primary==ordinary,'ordinary_repeat_exact':ordinary==frames[(batch,1,'ordinary',row)],'historical_hidden_exact':ordinary[:81920*4]==old_hidden.read_bytes(),'historical_state_exact':ordinary[81920*4:]==old_state.read_bytes(),'stage_repeats_exact':all(frames[(batch,0,label,row)]==frames[(batch,1,label,row)] for label in ['primary']+[n for n,_ in stages])})
  duplicate=all(frames[(batch,0,label,0)]==frames[(batch,0,label,row)] for label in ['ordinary','primary']+[n for n,_ in stages] for row in range(batch))
  report['cases'].append({'batch':batch,'rows':rows,'duplicate_rows_exact':duplicate,'ordinary_memory':ordinary_memory,'stage_graph':json.loads(graph.stats_json())});save()
  assert duplicate and all(v['primary_matches_ordinary'] and v['ordinary_repeat_exact'] and v['stage_repeats_exact'] for v in rows),'Diagnostic changes primary or repeat/row isolation; raw outputs saved, no root-cause attribution'
  graph.close();graph=None;decoder.close();decoder=None
 report['cross_batch_stage_exact']={label:frames[(1,0,label,0)]==frames[(2,0,label,0)] for label in ['ordinary','primary']+[n for n,_ in stages]};report['library_sha256']=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();report['status']='witness_passed_needs_independent_array_replay';save()
except Exception as error:report['status']='failed';report['error']=repr(error);save();raise
finally:
 if graph is not None:graph.close()
 if decoder is not None:decoder.close()
 save()
