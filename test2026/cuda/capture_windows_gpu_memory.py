"""Diagnostic WDDM counters, mapped to an NVIDIA ordinal by PCI ID + CUDA LUID.

Never run alongside a primary latency cohort. Formatted counters are sampled
observations, not true lifetime peaks. Preserve raw per-process/physical-node
rows and do not silently turn missing counters into zero memory.
"""
import argparse,csv,ctypes,json,re,struct,subprocess,time
from ctypes import wintypes
from pathlib import Path

def check(code):
    if code:raise RuntimeError('CUDA driver query failed: '+str(code))

def pci_key(value):
    domain,bus,rest=value.strip().split(':');device,function=rest.split('.')
    return tuple(int(x,16) for x in (domain,bus,device,function))

def adapter(device):
    fields=next(csv.reader(subprocess.check_output(['nvidia-smi','--id='+str(device),
        '--query-gpu=pci.bus_id,uuid,name','--format=csv,noheader'],text=True).splitlines()))
    pci,uuid,name=[s.strip() for s in fields]
    driver=ctypes.WinDLL('nvcuda.dll');check(driver.cuInit(0))
    count=ctypes.c_int();check(driver.cuDeviceGetCount(ctypes.byref(count)))
    for ordinal in range(count.value):
        dev=ctypes.c_int();check(driver.cuDeviceGet(ctypes.byref(dev),ordinal))
        bus=ctypes.create_string_buffer(32);check(driver.cuDeviceGetPCIBusId(bus,len(bus),dev))
        if pci_key(bus.value.decode())!=pci_key(pci):continue
        luid=ctypes.create_string_buffer(8);mask=ctypes.c_uint()
        check(driver.cuDeviceGetLuid(luid,ctypes.byref(mask),dev))
        low,high=struct.unpack('<II',luid.raw)
        assert mask.value and (low or high),'CUDA returned no usable Windows LUID'
        return {'nvidia_smi_device':device,'uuid':uuid,'name':name,'pci_bus_id':pci,
            'cuda_driver_ordinal':ordinal,'luid_high':high,'luid_low':low,'node_mask':mask.value,
            'counter_luid_tag':f'luid_0x{high:08x}_0x{low:08x}',
            'mapping':'NVIDIA PCI bus ID matched to CUDA driver device, then cuDeviceGetLuid'}
    raise RuntimeError('NVIDIA device PCI ID not found in CUDA driver enumeration')

def counters(mapping,target):
    # Tag comes only from validated numeric CUDA driver data, never shell input.
    tag=mapping['counter_luid_tag']
    command="$ErrorActionPreference='Stop'; @(Get-CimInstance -Namespace root/cimv2 -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {$_.Name -like '*"+tag+"*'} | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted) | ConvertTo-Json -Depth 3 -Compress"
    started=time.time()
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',command],text=True,
        creationflags=subprocess.CREATE_NO_WINDOW).strip()
    rows=json.loads(raw) if raw else []
    if isinstance(rows,dict):rows=[rows]
    for row in rows:
        match=re.fullmatch(r'pid_(\d+)_'+re.escape(tag)+r'_phys_(\d+)(?:#(\d+))?',row['Name'],re.I)
        if not match:raise RuntimeError('Unexpected WDDM adapter counter name: '+row['Name'])
        row['process_id']=int(match[1]);row['physical_node']=int(match[2])
        row['duplicate_instance_suffix']=int(match[3]) if match[3] else 0
        row['cuda_node_selected']=bool(mapping['node_mask']&(1<<row['physical_node']))
    selected=[r for r in rows if r['process_id']==target and r['cuda_node_selected']] if target else []
    return {'event':'sample','query_begin_unix':started,'query_end_unix':time.time(),
        'adapter_rows':rows,'target_process_id':target,'target_rows':selected,
        'target_counter_present':bool(selected) if target else None,
        'scope':'WDDM formatted per-process memory on PCI/LUID-matched adapter; sampled values, not true peaks. Missing target rows are unobserved, not a proven zero.'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output');parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--watch-process',type=int);parser.add_argument('--duration',type=float,default=0)
    parser.add_argument('--target-process',type=int)
    parser.add_argument('--post-exit-seconds',type=float,default=0)
    parser.add_argument('--interval',type=float,default=1)
    args=parser.parse_args();assert args.interval>=.2 and args.duration>=0 and 0<=args.post_exit_seconds<=60
    mapping=adapter(args.device);kernel=ctypes.WinDLL('kernel32.dll',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000|0x00100000,False,args.watch_process) if args.watch_process else None
    if args.watch_process and not handle:raise ctypes.WinError(ctypes.get_last_error())
    output=Path(args.output);started=time.monotonic();samples=0;exited_at=None
    with output.open('w') as stream:
        def emit(value):stream.write(json.dumps(value)+'\n');stream.flush()
        emit({'event':'mapping','unix_seconds':time.time(),'adapter':mapping,
            'watch_process_id':args.watch_process,'target_process_id':args.target_process if args.target_process is not None else args.watch_process})
        try:
            while True:
                emit(counters(mapping,args.target_process if args.target_process is not None else args.watch_process));samples+=1
                if handle:
                    status=kernel.WaitForSingleObject(handle,0)
                    if status==0:
                        if exited_at is None:
                            exited_at=time.monotonic();emit({'event':'watched_process_exited','unix_seconds':time.time()})
                        if time.monotonic()-exited_at>=args.post_exit_seconds:break
                    elif status!=258:raise ctypes.WinError(ctypes.get_last_error())
                elif time.monotonic()-started>=args.duration:break
                time.sleep(args.interval)
            emit({'event':'complete','unix_seconds':time.time(),'samples':samples,
                'reason':'watched_process_exited' if handle else 'duration_elapsed'})
        finally:
            if handle:kernel.CloseHandle(handle)
    print('QWEN35_WINDOWS_MEMORY_CAPTURE_COMPLETE',samples,flush=True)

if __name__=='__main__':main()
