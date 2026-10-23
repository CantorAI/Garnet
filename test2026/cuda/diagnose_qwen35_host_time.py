"""Coarse inclusive wall/thread-CPU clocks; diagnostic, never primary timing.

Accepts the unchanged benchmark_qwen35_garnet.py arguments. Wrappers preserve
arguments, return values, and exceptions. Nested categories overlap and must
not be summed. No CUDA synchronization or model-result cache is introduced.
"""
import functools,os,runpy,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
repo=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,repo)
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram
from xModel.qwen35.vl_0_8b.cuda_verify import CUDATargetVerifier
from xModel.qwen35.vl_0_8b.cuda_packed_replay import CUDAPackedDeltaReplayVerifier

current=None
def instrument(cls,name,label):
    original=getattr(cls,name)
    @functools.wraps(original)
    def timed(self,*args,**kwargs):
        if current is None:return original(self,*args,**kwargs)
        key=label(self);wall=time.perf_counter_ns();cpu=time.thread_time_ns()
        try:return original(self,*args,**kwargs)
        finally:
            elapsed_cpu=time.thread_time_ns()-cpu;elapsed=time.perf_counter_ns()-wall
            row=current.setdefault(key,{'calls':0,'inclusive_wall_ns':0,'inclusive_thread_cpu_ns':0})
            row['calls']+=1;row['inclusive_wall_ns']+=elapsed;row['inclusive_thread_cpu_ns']+=elapsed_cpu
    setattr(cls,name,timed)

instrument(CUDAPackedDecoderProgram,'run',lambda s:'target_program_c'+str(s.size))
instrument(CUDAMTPProgram,'run',lambda s:'mtp_program')
instrument(CUDATargetVerifier,'_read',lambda s:'target_read_reused' if s._reuse_accepted_head else 'target_read_fresh')
instrument(CUDAPackedDeltaReplayVerifier,'verify',lambda s:'target_verify_including_program_and_head')
instrument(CUDAPackedDeltaReplayVerifier,'commit_prefix',lambda s:'target_commit_including_read')
instrument(Model,'_prepare',lambda s:'prompt_and_vision_prepare')
original_request=Model.infer_batch
@functools.wraps(original_request)
def request(self,*args,**kwargs):
    global current
    if current is not None:raise RuntimeError('Diagnostic requires sequential non-nested requests')
    current={};wall=time.perf_counter_ns();cpu=time.thread_time_ns()
    try:
        result=original_request(self,*args,**kwargs)
        result['host_time_diagnostic']={
            'request_wall_ns':time.perf_counter_ns()-wall,
            'request_thread_cpu_ns':time.thread_time_ns()-cpu,
            'inclusive_categories':current,
            'scope':'Instrumented calling-thread CPU and wall time. Categories overlap; do not sum. CPU includes Python/native/driver work on this thread, excludes worker threads. Wall minus CPU is not GPU time or a causal bottleneck. Diagnostic clocks add overhead; not primary timing.'}
        return result
    finally:current=None
Model.infer_batch=request
benchmark=os.path.join(os.path.dirname(os.path.abspath(__file__)),'benchmark_qwen35_garnet.py')
sys.argv[0]=benchmark
runpy.run_path(benchmark,run_name='__main__')
