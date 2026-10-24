"""Actual complete projection through Garnet capture/lowering, not a model benchmark."""
import hashlib,json,math,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
checkpoint,fixture,output=map(Path,sys.argv[1:4]);manifest=json.loads((fixture/'manifest.json').read_text())
for name,digest in manifest['files'].items():
 if hashlib.sha256((fixture/name).read_bytes()).hexdigest()!=digest:raise ValueError('Fixture changed: '+name)
if hashlib.sha256((checkpoint/'model.safetensors.index.json').read_bytes()).hexdigest()!=manifest['source_index_sha256']:raise ValueError('Different checkpoint index')
m,k,n=manifest['m'],manifest['k'],manifest['logical_projection'][0]
def bf16(v):return struct.unpack('<f',struct.pack('<I',v<<16))[0]
values=[bf16(v[0]) for v in struct.iter_unpack('<H',(fixture/'input.bf16').read_bytes())]
expected=[v[0] for v in struct.iter_unpack('<f',(fixture/'expected.f32').read_bytes())]
symbol=tensor.input('ct_int4_projection',shape=[m,k],dtype=tensor.float32)
@T.fusion(name='actual_ct_int4_projection',boundary='required')
def forward(hidden):return hidden*T.unary_op('qwen35_checkpoint_projection',weight_name=manifest['prefix']+'.weight')
graph=garnet.compiled_graph();io=garnet.cuda_tensor_io();checks=[]
try:
 graph.build(forward(symbol),[symbol],str(checkpoint/'model.safetensors.index.json'))
 info=json.loads(graph.stats_json())
 assert info['checkpoint_weight_precision']=='ct_int4_g128' and info['native_int4_g128_projection_weights']==1
 assert info['native_nvfp4_projection_weights']==0 and info['native_fp8_projection_weights']==0 and info['external_bf16_weight_inputs']==0 and not info['engine_refittable']
 inputs=io.from_host(values,[m,k]);baseline=None
 for run,enabled in enumerate([False,True,True,True,True]):
  graph.enable_cuda_graph(enabled);actual=garnet.tensor_to_cpu(graph.run_device([inputs])).tolist()
  assert len(actual)==m*n and all(math.isfinite(v) for v in actual)
  selected=[actual[row*n+col] for row in range(m) for col in range(manifest['n'])]
  errors=[abs(a-b) for a,b in zip(selected,expected)]
  assert all(e<=max(1e-6,abs(v)*.0078125) for e,v in zip(errors,expected))
  raw=struct.pack('<'+'f'*len(actual),*actual)
  if baseline is None:baseline=raw
  else:assert raw==baseline,'Projection output changed under graph execution'
  path=output.with_name(output.stem+'-actual'+str(run)+'.f32');path.write_bytes(raw)
  checks.append({'run':run,'graph_enabled':enabled,'checked_oracle_outputs':len(selected),'max_abs':max(errors),'all_projection_output_sha256':hashlib.sha256(raw).hexdigest()})
 info=json.loads(graph.stats_json());assert info['graph_replays']>=2 and info['graph_capture_failures']==0
 output.write_text(json.dumps({'checks':checks,'stats':info,'scope':'Full5120-row actual projection compiled; CPU oracle checks first128 output rows only. All outputs deterministic through replay. Not decoder/full-model quality or benchmark.'},indent=2))
 print('CT_INT4_GARNET_CAPTURE_LOWERING_REAL_PROJECTION_PASS',flush=True)
finally:graph.close();io.close()
