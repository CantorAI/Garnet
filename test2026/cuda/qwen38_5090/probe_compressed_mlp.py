"""Two static MLP profiles sharing actual NVFP4 matrices; no serving benchmark."""
import hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
from test2026.cuda.qwen38_5090.run_logged import source_identity
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous shared MLP attempt')
checkpoint=root/'models/Qwen3.8-27B-NVFP4';prefix='model.language_model.layers.0.mlp';names=[prefix+'.'+n+'.weight' for n in ['gate_proj','up_proj','down_proj']]
report={'status':'starting','source':source_identity(),'cases':[], 'scope':'Actual layer0 MLP, static4/128 rows, same-profile embedded versus external full-output equality, two simultaneously resident engines sharing one compressed pool. Not complete decoder, VLM, TTFT or throughput.'}
io=garnet.cuda_tensor_io();live=[];references={};inputs={};weights=[]
def save():path.write_text(json.dumps(report,indent=2))
def raw(graph,values):return bytes(io.finite_f32_bytes(graph.run_device(values)))
try:
 io.open(str(checkpoint));metadata=[json.loads(io.compressed_weight_info(n)) for n in names]
 assert all(m['recipe']=='modelopt_nvfp4_w4a4' for m in metadata)
 width=metadata[0]['logical_shape'][1];expected_bytes=sum(m['packed_bytes']+m['scale_bytes'] for m in metadata)
 # Reference engines are released before constructing both external profiles.
 for rows in [4,128]:
  symbol=tensor.input('reference_hidden_'+str(rows),shape=[1,rows,width],dtype=tensor.float32)
  @T.fusion(name='shared_compressed_mlp_reference',boundary='required')
  def forward(x):return x*T.unary_op('qwen35_mlp',weight_prefix=prefix)
  graph=garnet.compiled_graph();live.append(graph);graph.set_builder_workspace_bytes(256<<20);graph.build(forward(symbol),[symbol],str(checkpoint));graph.enable_cuda_graph(True)
  data=io.from_host([((i%17)-8)/64 for i in range(rows*width)],[1,rows,width]);inputs[rows]=data
  baseline=raw(graph,[data]);assert raw(graph,[data])==baseline
  references[rows]=baseline;(out/('embedded-'+str(rows)+'.f32')).write_bytes(baseline)
  report['cases'].append({'mode':'embedded','rows':rows,'sha256':hashlib.sha256(baseline).hexdigest(),'stats':json.loads(graph.stats_json())});save();graph.close();live.remove(graph)
 for name in names:weights.extend([io.compressed_weight(name),io.compressed_scales(name)])
 pool=json.loads(io.memory_json());assert pool['compressed_projection_pool_entries']==3
 assert pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']==expected_bytes
 report['shared_pool_before_engines']=pool
 for rows in [4,128]:
  symbols=[tensor.input('shared_hidden_'+str(rows),shape=[1,rows,width],dtype=tensor.float32)]
  for i,m in enumerate(metadata):
   symbols.append(tensor.input('packed_'+str(rows)+'_'+str(i),shape=m['packed_shape'],dtype=tensor.int32))
   symbols.append(tensor.input('scales_'+str(rows)+'_'+str(i),shape=m['scale_shape'],dtype=tensor.int32))
  @T.fusion(name='shared_compressed_mlp_external',boundary='required')
  def forward(*args):
   bindings={name:[args[1+2*i],args[2+2*i]] for i,name in enumerate(names)}
   return args[0]*T.unary_op('qwen35_mlp',weight_prefix=prefix,external_compressed_weights=bindings)
  graph=garnet.compiled_graph();live.append(graph);graph.set_builder_workspace_bytes(256<<20);graph.build(forward(*symbols),symbols,str(checkpoint));graph.enable_cuda_graph(True)
  values=[inputs[rows]]+weights
  for repeat in range(3):
   actual=raw(graph,values);assert actual==references[rows],'External MLP differs from same-profile embedded MLP'
   (out/('external-'+str(rows)+'-'+str(repeat)+'.f32')).write_bytes(actual)
  stats=json.loads(graph.stats_json());assert stats['native_nvfp4_projection_weights']==3 and stats['external_compressed_weight_inputs']==3 and stats['external_compressed_weight_bytes']==expected_bytes
  assert stats['external_bf16_weight_inputs']==0 and stats['graph_replays']>=1 and stats['graph_capture_failures']==0
  report['cases'].append({'mode':'external','rows':rows,'sha256':hashlib.sha256(actual).hexdigest(),'stats':stats});save()
 after=json.loads(io.memory_json());assert after['compressed_projection_pool_entries']==3
 for key in ['compressed_projection_weight_bytes','compressed_projection_scale_bytes']:assert after[key]==pool[key]
 report['shared_pool_two_engines']=after
 # Interleave both still-resident engines to exercise shared allocation bindings.
 for rows,graph in zip([4,128],live):assert raw(graph,[inputs[rows]]+weights)==references[rows]
 report['interleaved_two_engine_exact']=True;report['status']='passed';print('ACTUAL_NVFP4_TWO_PROFILE_SHARED_MLP_PASS',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:
 save()
 for graph in live:graph.close()
 io.close()
