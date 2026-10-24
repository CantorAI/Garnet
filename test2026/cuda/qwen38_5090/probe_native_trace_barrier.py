"""Qualify a bounded native stream observation before expensive model tracing."""
import hashlib,json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
root,out=map(Path,sys.argv[1:3])
if out.exists():raise FileExistsError('Preserve previous trace barrier qualification')
io=garnet.cuda_tensor_io();report={'status':'starting','scope':'Native four-byte retained-view readback after GPU updates; validates instrumentation API only. No model, CUDA kernel timing, throughput or speedup.'}
try:
 io.open(str(root/'models/Qwen3.8-27B-NVFP4'));source=io.zeros([1,5120]);view=io.storage_view(source,[1,1]);rows=[]
 for value in (0.0,1.0,-0.25,2.5):
  io.update(source,[value]*5120);raw=bytes(io.finite_f32_bytes(view))
  assert len(raw)==4 and raw==struct.pack('<f',value)
  rows.append({'value':value,'readback_hex':raw.hex()})
 report.update(status='passed',cases=rows,readback_bytes=4,library_sha256=hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest())
 print('NATIVE_FOUR_BYTE_STREAM_OBSERVATION_PASS',flush=True)
except Exception as error:report.update(status='failed',error=repr(error));raise
finally:out.write_text(json.dumps(report,indent=2));io.close()
