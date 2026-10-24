from pathlib import Path
import datetime,json,os
r=Path('/mnt/d/CantorAI/5090-study-20261023');pid=171;p=Path('/proc')/str(pid);args=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0');assert '/mnt/d/CantorAI/5090-study-20261023/models/Qwen3.8-27B-INT4' in args and 'serve' in args and '18080' in args
out=r/'results/vllm-int4-cache-reset-startup-process.json';assert not out.exists()
def read(path):
 try:return path.read_text()
 except OSError as error:return repr(error)
fds={}
for fd in (p/'fd').iterdir():
 try:fds[fd.name]=os.readlink(fd)
 except OSError:pass
inodes={v[8:-1] for v in fds.values() if v.startswith('socket:[')}
net={name:[line for line in read(p/'net'/name).splitlines() if any(inode in line.split() for inode in inodes)] for name in ('tcp','tcp6','udp','udp6')}
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':pid,'argv':args,'status':read(p/'status'),'wchan':read(p/'wchan'),'syscall':read(p/'syscall'),'stack':read(p/'stack'),'fds':fds,'network_socket_rows':net,'thread_wait_channels':{t.name:read(t/'wchan') for t in (p/'task').iterdir()},'scope':'Read-only startup observation of the identity-checked owned vLLM parent; no performance measurement or process mutation.'}
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
