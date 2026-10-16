"""Check the exact generated route guards against 36 frozen retained-control requests.

The new candidate's counters remain unproven until actual execution. This
preflight cannot establish candidate outputs, correctness or performance.
"""
from pathlib import Path
import ast,hashlib,json,runpy,sys,tempfile
baseline=Path(sys.argv[1]).resolve();admission_path=Path(sys.argv[2]).resolve();output=Path(sys.argv[3]).resolve()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(admission_path)=='f9505c75246b7ac43fbf1c28987104dd1e5d4c7101ae265e984af20ba246df97'
admission=json.loads(admission_path.read_text())
assert json.loads((baseline/'archive-verification.json').read_text())['archive_sha256']=='7ed9d48e71a3723cf84e62fc95170bbee52cf614d90ef8fd7c93df338dc92c33'
proof=json.loads((baseline/'evidence/manifest.json').read_text())
with tempfile.TemporaryDirectory(prefix='garnet-selective-route-preflight-') as temporary:
    target=Path(temporary)/'generated';saved=sys.argv[:]
    try:
        sys.argv=['generate','/unused',str(target),'--generate-only']
        runpy.run_path(str(Path(__file__).with_name('compare_ndarray_selective_native.py')),run_name='__main__')
    finally:sys.argv=saved
    expanded=target.with_name(target.name+'.executed.py').read_text();tree=ast.parse(expanded)
expressions={}
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('expected_nd','expected_legacy'):
        assert node.targets[0].id not in expressions
        expressions[node.targets[0].id]=compile(ast.Expression(node.value),'exact-generated-route-expression','eval')
assert set(expressions)=={'expected_nd','expected_legacy'}
checked=[]
for model in admission['models']:
    name=model['model'];common_half_count=sum(c['global_half_ready'] for c in model['cases'])
    tensors=[c['tensor'] for c in model['cases'] if not c['exact_half_ready']]
    for batch in (1,2):
        label=name+'-b'+str(batch)
        for repeat in (0,1):
            rel='evidence/logs/ndarray-native-repaired-20261015/jobs/'+label+'-attempt0/'+label+'-control-'+str(repeat)+'/output.log'
            path=baseline/rel;info=proof['files'][rel]
            assert digest(path)==info['sha256'] and path.stat().st_size==info['bytes']
            prefix='matched_frame_result ' if name=='Qwen3.5-0.8B' else 'matched_result '
            rows=[json.loads(line.split(' ',1)[1]) for line in path.read_text().splitlines() if line.startswith(prefix)];assert len(rows)==3
            for value in rows:
                variables=dict(value=value,batch=batch,common_half_count=common_half_count,ndarray_tensors=tensors,policy='control')
                expected=eval(expressions['expected_legacy'],variables)
                assert value['prefill_profile']['global_half_alpha_dispatches']==expected
                assert eval(expressions['expected_nd'],variables)==0
                variables['policy']='candidate';new_expected=eval(expressions['expected_nd'],variables)
                assert new_expected==(1 if name=='Qwen3.5-0.8B' else 0)
                checked.append(dict(model=name,batch=batch,repeat=repeat,request=value['request'],
                    observed_control_common_half_calls=expected,unexecuted_selective_candidate_expected_calls=new_expected))
assert len(checked)==36
result=dict(observed_control_requests=36,expanded_comparator_sha256=hashlib.sha256(expanded.encode()).hexdigest(),
    checked=checked,scope=__doc__,full_goal_met=False)
output.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='checked'}))
