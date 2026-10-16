"""Read-only diagnosis of the latest guard actor; no argv or constants emitted."""
import ast,hashlib,json,re,shlex,subprocess,sys
from pathlib import Path
folder=Path(sys.argv[1]);waits=json.loads((folder/'quiet-waits.json').read_text());assert waits
matches=re.findall(r'"(\d+)\s',waits[-1]['reason']);assert matches
pid=int(matches[0]);r=subprocess.run(['/bin/ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
result=dict(pid=pid,live=r.returncode==0 and bool(r.stdout.strip()),raw_arguments_emitted=False)
if result['live']:
 args=shlex.split(r.stdout.strip());result['argv_sha256']=hashlib.sha256(r.stdout.encode()).hexdigest();source=None
 if '-c' in args:
  source=args[args.index('-c')+1].encode();result['source_kind']='inline'
 else:
  paths=[Path(v) for v in args[1:] if v.endswith(('.py','.x')) and Path(v).is_file()]
  if len(paths)==1:source=paths[0].read_bytes();result['source_kind']='file'
 if source is not None:
  result['source_sha256']=hashlib.sha256(source).hexdigest();lower=source.decode(errors='replace').lower()
  result['inference_marker_counts']={k:lower.count(k) for k in ('garnet','mlx','ollama','vllm','torch','qwen','metal','openvino','inference')}
  try:
   tree=ast.parse(source.decode());imports=[];calls=set()
   for n in ast.walk(tree):
    if isinstance(n,ast.Import):imports.extend(v.name for v in n.names)
    if isinstance(n,ast.ImportFrom):imports.append(n.module or '')
    if isinstance(n,ast.Call):
     f=n.func;names=[]
     while isinstance(f,ast.Attribute):names.append(f.attr);f=f.value
     if isinstance(f,ast.Name):names.append(f.id);calls.add('.'.join(reversed(names)))
   result.update(imports=sorted(set(imports)),call_symbols=sorted(calls),parse_succeeded=True)
  except Exception as error:result.update(parse_succeeded=False,parse_error_type=type(error).__name__)
 libraries=subprocess.run(['/usr/sbin/lsof','-p',str(pid),'-Fn'],capture_output=True,text=True)
 result['library_audit_exit']=libraries.returncode;result['inference_library_markers']={k:k in libraries.stdout.lower() for k in ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino','coreml.framework','appleneuralengine','ane.framework','bnns')}
result['guard_exclusion_authorized']=False
print(json.dumps(result,indent=2))
