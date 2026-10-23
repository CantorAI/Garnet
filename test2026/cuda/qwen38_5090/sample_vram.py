"""Adapter-wide NVML memory sampling; sampled maxima are not instantaneous peaks."""
import ctypes,os,threading,time,json,statistics
from pathlib import Path

class Memory(ctypes.Structure):
    _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]

class Sampler:
    def __init__(self,path,index=0,interval=.02):
        self.path=Path(path);self.interval=interval;self.phase='initial';self.samples=[];self.errors=[]
        self.stop=threading.Event();self.lock=threading.Lock();self.handle=ctypes.c_void_p()
        self.lib=ctypes.WinDLL('nvml.dll') if os.name=='nt' else ctypes.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlInit_v2.argtypes=[];self.lib.nvmlShutdown.argtypes=[]
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory)]
        for name in ('nvmlDeviceGetName','nvmlDeviceGetUUID'):
            getattr(self.lib,name).argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint]
        self.check(self.lib.nvmlInit_v2())
        try:
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index,ctypes.byref(self.handle)))
            def text(name):
                buffer=ctypes.create_string_buffer(256);self.check(getattr(self.lib,name)(self.handle,buffer,len(buffer)))
                return buffer.value.decode()
            self.device={'index':index,'name':text('nvmlDeviceGetName'),'uuid':text('nvmlDeviceGetUUID')}
        except Exception:self.lib.nvmlShutdown();raise
        self.file=self.path.open('x',encoding='utf-8');self.thread=threading.Thread(target=self.loop,daemon=True)
    @staticmethod
    def check(code):
        if code:raise RuntimeError('NVML error '+str(code))
    def sample(self):
        with self.lock:
            memory=Memory();self.check(self.lib.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(memory)))
            row={'monotonic':time.perf_counter(),'phase':self.phase,'used_bytes':memory.used,'free_bytes':memory.free,'total_bytes':memory.total}
            self.samples.append(row);self.file.write(json.dumps(row)+'\n');self.file.flush()
            return row
    def loop(self):
        while not self.stop.wait(self.interval):
            try:self.sample()
            except Exception as error:self.errors.append(repr(error));self.stop.set()
    def start(self):self.sample();self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread.ident is not None:self.thread.join()
        try:self.sample()
        finally:self.file.close();self.lib.nvmlShutdown()
        if self.errors:raise RuntimeError('Memory sampling failed: '+str(self.errors))
    def peak(self,phase):
        rows=[r for r in self.samples if r['phase']==phase]
        if not rows:raise ValueError('No samples for cohort')
        gaps=[b['monotonic']-a['monotonic'] for a,b in zip(rows,rows[1:])]
        return {'sampled_peak_adapter_used_bytes':max(r['used_bytes'] for r in rows),'samples':len(rows),'requested_interval_seconds':self.interval,
            'observed_sample_gap_median_seconds':statistics.median(gaps) if gaps else None,
            'observed_sample_gap_max_seconds':max(gaps) if gaps else None,
            'scope':'NVML adapter-wide used memory, including background allocations; sampled maximum, not instantaneous or process-attributed peak'}
