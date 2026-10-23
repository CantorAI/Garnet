"""Trace request index8 of the unchanged real-image cohort; diagnostic only."""
import ctypes,os,runpy,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
repo=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,repo)
from runtime.cuda_serving import Model
runtime=ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
original=Model.infer_batch
request_index=0
def traced(self,*args,**kwargs):
    global request_index
    index=request_index;request_index+=1;capture=index==8
    if capture:assert runtime.cudaProfilerStart()==0,'cudaProfilerStart failed'
    try:result=original(self,*args,**kwargs)
    finally:
        if capture:
            try:assert runtime.cudaDeviceSynchronize()==0,'capture tail synchronization failed'
            finally:assert runtime.cudaProfilerStop()==0,'cudaProfilerStop failed'
    result['gpu_trace_diagnostic']={'request_index':index,'capture':capture,
        'capture_tail_synchronized':capture,
        'scope':'One warmed matched request under node-level Nsight tracing; queued tail retired before stopping capture. Initialization and first two rounds excluded from trace. Profiler overhead and historical control prevent speed or causality claims.'}
    return result
Model.infer_batch=traced
benchmark=os.path.join(os.path.dirname(os.path.abspath(__file__)),'benchmark_qwen35_garnet.py')
sys.argv[0]=benchmark
runpy.run_path(benchmark,run_name='__main__')
assert request_index==28,'Expected full28 request cohort'
