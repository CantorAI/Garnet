"""Read only diagnostics for exact guard-rejected PIDs; never print argv/env."""
import ast,hashlib,json,shlex,subprocess,sys
from pathlib import Path

allowed=Path('/Users/zsdx/cantorai-release-tests/current/bin').resolve()
inference=('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework',
           'libopenvino','coreml.framework','appleneuralengine','ane.framework','bnns')
records=[]
for value in sys.argv[1:]:
    pid=int(value);chain=[];current=pid
    for _ in range(3):
        result=subprocess.run(['/bin/ps','-p',str(current),'-o','ppid=,args='],capture_output=True,text=True)
        fields=result.stdout.strip().split(None,1)
        if result.returncode or len(fields)!=2:break
        entry=dict(pid=current,parent=int(fields[0]),known_scripts=[])
        try:tokens=shlex.split(fields[1])
        except ValueError:tokens=[]
        if ' -c ' in fields[1]:
            payload=fields[1].split(' -c ',1)[1]
            entry['inline_payload_sha256']=hashlib.sha256(payload.encode()).hexdigest()
            entry['inline_mentions']={key:(key in payload.lower()) for key in ('worldsense','garnet','metal','mlx','vllm','torch','qwen')}
        for token in tokens:
            path=Path(token)
            if not path.is_absolute() or path.suffix!='.py':continue
            path=path.resolve()
            if not path.is_relative_to(allowed) or not path.is_file():continue
            raw=path.read_bytes();text=raw.decode(errors='replace').lower()
            try:
                tree=ast.parse(raw.decode())
                imports=sorted({node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}|
                               {item.name for node in ast.walk(tree) if isinstance(node,ast.Import) for item in node.names})
            except (SyntaxError,UnicodeDecodeError):imports=None
            normalized=text.replace("'garnet_trt_weight_streaming_budget_mib'",'')
            entry['known_scripts'].append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                marker_counts={key:text.count(key) for key in ('garnet','metal','mlx','vllm','torch','qwen')},
                inference_markers_after_exact_budget_literal_removed=any(key in normalized for key in ('garnet','metal','mlx','vllm','torch','qwen')),
                imported_modules=imports))
        chain.append(entry);current=int(fields[0])
        if current<=1:break
    loaded=subprocess.run(['/usr/sbin/lsof','-p',str(pid),'-Fn'],capture_output=True,text=True)
    hits=sorted({Path(line[1:]).name for line in loaded.stdout.splitlines()
                 if line.startswith('n') and any(key in line.lower() for key in inference)})
    records.append(dict(pid=pid,ancestor_chain=chain,lsof_exit_code=loaded.returncode,
                        inference_or_gpu_library_names=hits,
                        scope='Diagnostic only; does not exempt, stop or modify this process'))
print(json.dumps(dict(processes=records),indent=2))
