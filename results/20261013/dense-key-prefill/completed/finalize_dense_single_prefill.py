"""Only promote the candidate to performance testing after full numerical checks."""
import ast,hashlib,json,os,subprocess,tarfile,tempfile,time
from pathlib import Path

root=Path('/Users/zsdx/garnet-metal-investigation')
source=root/'source';scripts=source/'test2026/metal'
output=root/'logs/dense-single-finalize-20261013/jobs'
output.mkdir(parents=True,exist_ok=False)
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body
       if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
checks=root/'logs/split-key64-extended-dense-checks-20261013'
assert json.loads((checks/'exit.json').read_text())['exit_code']==0
records=[]
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    last=json.loads((root/'models'/name/'config.json').read_text())['num_hidden_layers']-1
    lines=(checks/'jobs'/(name+'.log')).read_text().splitlines()
    values=[json.loads(line.split('dense vector reference ',1)[1]) for line in lines if line.startswith('dense vector reference ')]
    assert len(values)==24
    assert {(v['layer'],v['batch'],v['chunk'],v['step']) for v in values}=={
        (layer,batch,chunk,step) for layer in (0,last) for batch in (1,2) for chunk in (32,64) for step in (0,1,2)}
    assert all(v['worst_ratio']<=1 and v['hidden_l2_worst_ratio']<=1 and v['value_l2_worst_ratio']<=1 for v in values)
    records.append(dict(model=name,checks=values))
(output/'extended-numerical-proof.json').write_text(json.dumps(records,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
proof=dict(library_sha256=digest(root/'build-split-key64-prefill/libgarnet.dylib'),
           selected_sha256=digest(root/'build-linear-chunk/libgarnet.dylib'),
           kernel_sha256=digest(source/'src/backends/metal/metal_context.mm'))
assert proof['selected_sha256']=='26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671'
(output/'before-feature-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
guard()
archive=root/'split-key64-numerical-evidence.tar.gz'
names=['split-key64-installed.json','source/src/backends/metal/metal_context.mm',
       'source/src/backends/metal/compiled_graph.cpp','source/src/backends/metal/metal_context.h',
       'source/cmake/GarnetMetal.cmake','source/test2026/metal/matmul_test.cpp',
       'source/test2026/metal/test_extended_dense_reference_numpy.py',
       'source/test2026/metal/run_dense_reference_capture.py','source/test2026/metal/validate_dense_capture.py',
       'logs/split-key64-build-20261013','logs/split-key64-checks-20261013',
       'logs/split-key64-extended-dense-checks-20261013']
with tarfile.open(archive,'w:gz') as t:
    for name in names:t.add(root/name,arcname=name)
(output/'numerical-archive.json').write_text(json.dumps(dict(sha256=digest(archive),bytes=archive.stat().st_size),indent=2)+'\n')
bundle=root/'dense-single-final-source.tar.gz'
assert digest(bundle)=='ca6d4660b2855012dd7a4cb42a21577cf68a3a65d960ddd248cec1eaebdff9ba'
with tempfile.TemporaryDirectory(dir=root) as folder:
    with tarfile.open(bundle) as t:t.extractall(folder,filter='data')
    extracted=Path(folder);manifest=json.loads((extracted/'manifest.json').read_text())
    for name,sha in manifest.items():assert digest(extracted/name)==sha
    for name in manifest:os.replace(extracted/name,source/name)
(output/'installed-feature-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
statuses=[]
def run(label,command):
    guard()
    print('Starting',label,flush=True)
    with (output/(label+'.log')).open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    statuses.append(dict(label=label,command=command,exit_code=result.returncode))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    assert result.returncode==0,(label,result.returncode)
    guard()
    print('Passed',label,flush=True)
py='/opt/homebrew/bin/python3'
run('build',[py,str(root/'build_dense_single_prefill.py')])
assert digest(source/'src/backends/metal/metal_context.mm')==proof['kernel_sha256']
run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(root/'build-dense-single-prefill'),'--output-on-failure'])
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
base=root/'logs/equivalent-f32-linear-chunk-text-20261013/jobs'
for batch in (1,2):
    for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
        label=name+'-b'+str(batch)
        run('old-profile-guard-'+label,[runtime,str(scripts/'run_dense_profile_guard.py'),str(root/'build-linear-chunk'),
            str(source),str(root/'models'/name),str(base/(label+'-case.json'))])
        actual=[json.loads(v.split(' ',1)[1]) for v in (output/('old-profile-guard-'+label+'.log')).read_text().splitlines() if v.startswith('matched_result ')]
        frozen=[json.loads(v.split(' ',1)[1]) for v in (base/(label+'-garnet-0/output.log')).read_text().splitlines() if v.startswith('matched_result ')]
        assert len(actual)==len(frozen)==3
        for a,b in zip(actual,frozen):
            for key in ('token_ids','finished','prompt_token_ids'):assert a[key]==b[key]
run('native-abba',[py,str(scripts/'benchmark_dense_single_prefill.py'),str(root),str(root/'logs/dense-single-native-abba-20261013/jobs')])
(output/'final-binary-proof.json').write_text(json.dumps(dict(candidate_sha256=digest(root/'build-dense-single-prefill/libgarnet.dylib'),
    selected_sha256=digest(root/'build-linear-chunk/libgarnet.dylib'),kernel_sha256=digest(source/'src/backends/metal/metal_context.mm')),indent=2)+'\n')
