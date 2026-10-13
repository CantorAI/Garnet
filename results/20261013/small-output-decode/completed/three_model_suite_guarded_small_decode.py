"""Run isolated validation/benchmarks and preserve each actual command and exit."""
import json,subprocess,sys
from pathlib import Path
runtime,build,source,phase,output=sys.argv[1:];source=Path(source);output=Path(output);root=source.parent
import ast,hashlib,os,time
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
output.mkdir(parents=True,exist_ok=True)
nodes=[n for n in ast.parse((source/'test2026/metal/compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())

scripts=source/'test2026/metal';models=root/'models';tasks=[]
if phase in ('checks','projection-checks','handoff-checks'):
    extra=['bf16'] if phase in ('projection-checks','handoff-checks') else []
    tasks.append(('qwen35-layers',[sys.executable,str(scripts/'test_chunk_graph.py'),runtime,build,str(source),str(models/'Qwen3.5-0.8B')]+extra))
    for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        tasks.append((name+'-layers',[sys.executable,str(scripts/'test_dense_reference.py'),runtime,build,str(source),str(models/name)]+extra))
    tasks.append(('serving',[sys.executable,str(scripts/'test_decoder_model.py'),runtime,build,str(source)]))
    tasks.append(('mixed-image-pipeline',[sys.executable,str(scripts/'test_image_pipeline.py'),runtime,build,str(source)]))
    for name in ('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        tasks.append((name+'-head',[sys.executable,str(scripts/'test_head_bf16.py'),runtime,build,str(source),str(models/name)]))
    tasks.append(('vision-small',[sys.executable,str(scripts/'test_vision_graph.py'),runtime,build,str(source)]))
    tasks.append(('vision-pretrained',[sys.executable,str(scripts/'test_pretrained_vision_graph.py'),runtime,build,str(source),str(models/'Qwen3.5-0.8B')]))
    tasks.append(('ctest',['/opt/homebrew/bin/ctest','--test-dir',build,'--output-on-failure']))
    if phase=='handoff-checks':
        early=('ctest','serving','mixed-image-pipeline')
        tasks=[next(task for task in tasks if task[0]==name) for name in early]+[task for task in tasks if task[0] not in early]
elif phase=='index-checks':
    # The address-width variant retains the same arithmetic and GEMM path.
    # Cover aligned/odd and padded matrix cases, then the real large heads.
    tasks.append(('ctest',['/opt/homebrew/bin/ctest','--test-dir',build,'--output-on-failure']))
    for name in ('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        tasks.append((name+'-head',[sys.executable,str(scripts/'test_head_bf16.py'),runtime,build,str(source),str(models/name)]))
elif phase in ('benchmarks','projection-benchmarks'):
    for batch in (1,2):
        for name in ('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
            for mode in ((2,3,3) if phase=='projection-benchmarks' else (0,3,3)):
                suffix='control' if mode==(2 if phase=='projection-benchmarks' else 0) else 'optimized' if not tasks or not tasks[-1][0].endswith('optimized') else 'repeat'
                if name=='Qwen3.5-0.8B':
                    command=[sys.executable,str(scripts/'benchmark_image.py'),runtime,build,str(source),str(models/name),str(root/'images'/('RealWorld-04.batch'+str(batch)+'.json')),'32','16',str(mode)]
                else:
                    prompt='What is the capital of France? Answer briefly.' if name=='Qwen3-0.6B' else 'Write a Python function add(a, b) that returns their sum. Output only code.'
                    command=[sys.executable,str(scripts/'benchmark_dense_text.py'),runtime,build,str(source),str(models/name),str(batch),'64',prompt,'16',str(mode)]
                tasks.append((name+'-b'+str(batch)+'-'+suffix,command))
elif phase=='extended':
    for name,prompt in [('Qwen3-0.6B','Give five numbered, practical tips for writing readable Python. Keep each tip to one sentence.'),
        ('Qwen2.5-Coder-0.5B-Instruct','Write a Python function factorial(n) using a loop. Output only code.')]:
        for mode in (0,3):
            command=[sys.executable,str(scripts/'benchmark_dense_text.py'),runtime,build,str(source),str(models/name),'1','128',prompt,'16',str(mode)]
            tasks.append((name+'-extended-'+str(mode),command))
elif phase=='resident':
    tasks.append(('all-three-resident',[sys.executable,str(scripts/'benchmark_three_resident.py'),runtime,build,str(source)]))
else:raise ValueError('Unknown suite phase')
statuses=[]
for name,command in tasks:
    guard()
    folder=output/name;folder.mkdir(parents=True,exist_ok=True)
    print('Starting isolated job',name,flush=True)
    with (folder/'output.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status=dict(name=name,command=command,exit_code=result.returncode);statuses.append(status)
    (folder/'exit.json').write_text(json.dumps(status,indent=2)+'\n');(output/'job_statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    print('Completed',name,'exit',result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
