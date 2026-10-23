"""Signal only a verified study-owned Linux reference parent; caller checks exit."""
import argparse,hashlib,json,os,signal
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['vllm','sglang']);p.add_argument('--pid',type=int,required=True);a=p.parse_args()
    if a.pid<=1:raise ValueError('Invalid parent PID')
    raw=Path('/proc/'+str(a.pid)+'/cmdline').read_bytes();argv=raw.decode().strip('\0').split('\0')
    prefix='/opt/cantorai5090/'+a.engine+'/bin/';checkpoint='/mnt/d/CantorAI/5090-study-20261023/models/Qwen3.8-27B-NVFP4'
    port='18081' if a.engine=='sglang' else '18080'
    if not argv[0].startswith(prefix) or checkpoint not in argv or '--port' not in argv or argv[argv.index('--port')+1]!=port:
        raise ValueError('PID is not the expected study-owned reference server')
    print(json.dumps({'engine':a.engine,'pid':a.pid,'signal':'SIGTERM','cmdline_sha256':hashlib.sha256(raw).hexdigest(),
        'scope':'Signal delivery only; process exit still requires observation'}),flush=True)
    os.kill(a.pid,signal.SIGTERM)
