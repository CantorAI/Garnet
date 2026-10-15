"""Read-only role diagnosis; never emit unrelated argv, constants or library paths."""
import ast,hashlib,json,subprocess
from pathlib import Path
script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
launch=script.with_name('launch.py');raw=script.read_bytes();text=raw.decode(errors='replace').lower()
budget="'garnet_trt_weight_streaming_budget_mib'"
tree=ast.parse(raw.decode());imports=[];calls=set()
for node in ast.walk(tree):
 if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
 if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
 if isinstance(node,ast.Call):
  n=node.func;names=[]
  while isinstance(n,ast.Attribute):names.append(n.attr);n=n.value
  if isinstance(n,ast.Name):names.append(n.id);calls.add('.'.join(reversed(names)))
result=dict(script_sha256=hashlib.sha256(raw).hexdigest(),launch_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),
    inference_marker_counts={key:text.count(key) for key in ('garnet','mlx','ollama','vllm','torch','qwen','metal')},
    markers_after_exact_budget_removal={key:text.replace(budget,'').count(key) for key in ('garnet','mlx','ollama','vllm','torch','qwen','metal')},
    imports=sorted(set(imports)),call_symbols=sorted(calls),processes=[])
for pid in (2897,2985):
 entry=dict(pid=pid);current=pid;identified=False;identity=''
 for _ in range(3):
  p=subprocess.run(['/bin/ps','-p',str(current),'-o','ppid=,args='],capture_output=True,text=True);fields=p.stdout.strip().split(None,1)
  if p.returncode or len(fields)!=2:break
  if current==pid:identity=fields[1]
  if str(script) in fields[1]:identified=True;break
  current=int(fields[0])
  if current<=1:break
 prefix=str(script.parents[2]/'xlang3')+' -B -u -c '
 entry.update(live=bool(identity),known_supervisor_ancestor=identified,known_inline_prefix=identity.startswith(prefix),inline_sha256=hashlib.sha256(identity[len(prefix):].encode()).hexdigest() if identity.startswith(prefix) else None)
 libraries=subprocess.run(['/usr/sbin/lsof','-p',str(pid),'-Fn'],capture_output=True,text=True)
 entry['library_audit_exit']=libraries.returncode
 entry['inference_library_markers']={key:key in libraries.stdout.lower() for key in ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino','coreml.framework','appleneuralengine','ane.framework','bnns')}
 result['processes'].append(entry)
print(json.dumps(result,indent=2))
