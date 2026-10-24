"""Preflight the existing one-shot TRT profiler on a real bounded normalization graph."""
import hashlib,json,math,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);report_path=out/'report.json'
if any(out.iterdir()):raise FileExistsError('Preserve prior profiler preflight')
flag='GARNET_PROFILE_DECODE_LAYERS';profile=out/'layers.tsv';checkpoint=root/'models/Qwen3.8-27B-NVFP4'
report={'status':'starting','scope':'Instrumentation-only normalization graph. Existing IProfiler hook samples one eager enqueue per engine and bypasses CUDA Graph for that enqueue; not representative captured-replay/model latency or isolated GPU-active timing.'}
io=garnet.cuda_tensor_io();graph=None
try:
 if os.environ.get(flag):raise ValueError('Preflight requires an unset inherited layer-profiler flag')
 io.open(str(checkpoint));config=json.loads((checkpoint/'config.json').read_text())
 symbol=tensor.input('profiler_preflight',shape=[1,4,5120],dtype=tensor.float32)
 @T.fusion(name='profiler_preflight_normalization',boundary='required')
 def forward(value):return value*T.unary_op('qwen35_rms_norm_packed',weight_name='model.language_model.norm.weight',eps=config['text_config']['rms_norm_eps'])
 graph=garnet.compiled_graph();graph.build(forward(symbol),[symbol],str(checkpoint));graph.enable_cuda_graph(True)
 value=io.from_host([((i%29)-14)/32 for i in range(4*5120)],[1,4,5120]);reference=None
 for repeat in range(3):
  raw=bytes(io.finite_f32_bytes(graph.run_device([value])))
  if reference is None:reference=raw
  else:assert raw==reference
 os.environ[flag]=str(profile)
 try:profiled=bytes(io.finite_f32_bytes(graph.run_device([value])))
 finally:os.environ.pop(flag,None)
 assert profiled==reference
 assert bytes(io.finite_f32_bytes(graph.run_device([value])))==reference
 if not profile.exists():raise RuntimeError('Native runtime did not emit the dynamically enabled profile')
 rows=[]
 for line in profile.read_text().splitlines():
  fields=line.split('\t')
  if fields==['engine','layer','milliseconds']:continue
  if len(fields)!=3:raise ValueError('Malformed native layer profile row')
  milliseconds=float(fields[2]);assert math.isfinite(milliseconds) and milliseconds>=0
  rows.append({'engine':fields[0],'layer':fields[1],'milliseconds':milliseconds})
 assert rows,'No native layer samples emitted'
 stats=json.loads(graph.stats_json());assert stats['graph_capture_failures']==0 and stats['graph_captures']==1
 report.update(status='passed',layers=rows,output_sha256=hashlib.sha256(reference).hexdigest(),output_bytes=len(reference),graph=stats,
  library_sha256=hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest())
 print('NATIVE_TRT_LAYER_PROFILER_PREFLIGHT_PASS',flush=True)
except Exception as error:report.update(status='failed',error=repr(error));raise
finally:
 os.environ.pop(flag,None);report_path.write_text(json.dumps(report,indent=2))
 if graph is not None:graph.close()
 io.close()
