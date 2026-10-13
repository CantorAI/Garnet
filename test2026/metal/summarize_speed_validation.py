"""Validate extended generation and three-model residency from saved native logs."""
import ast, hashlib, json, re, sys
from pathlib import Path

root=Path(sys.argv[1])
comparison=json.loads((root/'target_comparison.json').read_text(encoding='utf-8'))

def read(folder):
    status=json.loads((folder/'exit.json').read_text(encoding='utf-8'))
    assert status['exit_code']==0,status
    results=[];process=None
    for line in (folder/'output.log').read_text(encoding='utf-8').splitlines():
        marker,_,payload=line.partition(' ')
        if marker in ('metal_text_result','metal_three_resident_result'):
            results.append(json.loads(payload))
        elif marker in ('metal_text_process','metal_three_resident_process'):
            process=json.loads(payload)
    if process is not None:assert process['exit_code']==0
    return results,process

extended=[];factorial=None
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    old,_=read(root/'double-extended-20261012/jobs'/(name+'-extended-0'))
    new,process=read(root/'double-extended-20261012/jobs'/(name+'-extended-3'))
    assert len(old)==len(new)==2
    for a,b in zip(old,new):
        assert a['token_ids']==b['token_ids'] and a['finished']==b['finished']
        assert all(b['finished']) and all(t>0 for t in b['decode_step_seconds'])
    extended.append(dict(model=name,requests=2,exact_tokens=True,finished=new[1]['finished'],
        output_tokens=new[1]['output_tokens'],prompt=new[1]['prompt'],text=new[1]['text'],
        ttft_seconds=new[1]['ttft_loaded_seconds'],tokens_per_second=new[1]['decode_steps_per_second'],
        process=process))
    if name.startswith('Qwen2.5'):factorial=new[1]['text'][0]

resident,process=read(root/'double-resident-20261012/jobs/all-three-resident')
assert len(resident)==6 and process is not None
for item in resident:
    control=next(v for v in comparison['comparisons'] if v['model']==item['model'] and v['batch']==1)['control_samples'][0]
    assert item['token_ids']==control['token_ids'] and item['finished']==control['finished']
    assert item['models_loaded']==3 and item['decode_steps_per_second']>0

smoke,_=read(root/'double-qwen35-text-bf16-20261012')
assert len(smoke)==2
for item in smoke:
    assert item['bf16_projections'] and item['batch']==2 and all(item['finished'])
    assert item['token_ids']==[[11751,248046],[11751,248046]]

add=next(v for v in comparison['comparisons'] if v['model'].startswith('Qwen2.5') and v['batch']==1)['optimized_samples'][0]['text'][0]
code_tests=[]
allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.BinOp,
    ast.Add,ast.Name,ast.Load,ast.Store,ast.Assign,ast.Constant,ast.For,ast.Call,ast.AugAssign,ast.Mult)
for name,text,cases in (
    ('add',add,[([2,3],5),([-4,9],5),([0,0],0),([1.5,2.5],4.0)]),
    ('factorial',factorial,[([0],1),([1],1),([5],120),([7],5040)])):
    source=re.search(r'```python\n(.*?)```',text,re.S).group(1)
    tree=ast.parse(source)
    assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name==name
    assert all(isinstance(node,allowed) for node in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert isinstance(node.func,ast.Name) and node.func.id=='range' and not node.keywords
    namespace={'__builtins__':{'range':range}}
    exec(compile(tree,'<validated native generated code>','exec'),namespace)
    tests=[]
    for inputs,expected in cases:
        actual=namespace[name](*inputs);assert actual==expected
        tests.append(dict(inputs=inputs,expected=expected,actual=actual))
    code_tests.append(dict(function=name,source=source,sha256=hashlib.sha256(source.encode()).hexdigest(),
        ast_restricted=True,tests=tests))

output=dict(extended=extended,resident=dict(exact_tokens=True,requests=6,results=resident,process=process),
    qwen35_text_batch2=dict(exact_tokens=True,requests=2,results=smoke),generated_code=code_tests)
(root/'post_validation.json').write_bytes((json.dumps(output,indent=2)+'\n').encode('utf-8'))
print(json.dumps(dict(extended_exact_tokens=True,resident_exact_tokens=True,
    resident_peak_rss_gib=process['peak_rss_bytes']/2**30,qwen35_text_batch2=True,generated_function_cases=8)))
