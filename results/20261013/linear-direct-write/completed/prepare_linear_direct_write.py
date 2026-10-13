"""Preserve the measured dense candidate before installing the next experiment."""
import ast,hashlib,json,os,subprocess,tarfile,tempfile,time
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');source=root/'source';scripts=source/'test2026/metal'
output=root/'logs/linear-direct-write-prepare-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
guard()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
names=['logs/dense-single-finalize-20261013','logs/dense-single-native-abba-20261013','logs/dense-single-native-confirm-20261013',
       'logs/equivalent-f32-single-prefill-text-20261013','logs/equivalent-f32-single-prefill-summary-20261013',
       'compare_equivalent_f32_single_prefill.py','equivalent_f32_summarize_vllm_comparison.py',
       'finalize_dense_single_prefill.py','build_dense_single_prefill.py']
for name in names:
    if name.startswith('logs/'):
        if name=='logs/dense-single-native-abba-20261013':
            jobs=root/name/'jobs'
            assert len(json.loads((jobs/'summary.json').read_text())['groups'])==4
            assert all(v['exit_code']==0 for v in json.loads((jobs/'statuses.json').read_text()))
        else:assert json.loads((root/name/'exit.json').read_text())['exit_code']==0,name
for name in ('cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/backends/metal/metal_context.h',
             'src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/matmul_test.cpp',
             'test2026/metal/run_matched_garnet.py','xModel/qwen/metal_dense/decoder.py',
             'test2026/metal/benchmark_dense_single_prefill.py','test2026/metal/test_extended_dense_reference_numpy.py',
             'test2026/metal/run_dense_reference_capture.py','test2026/metal/validate_dense_capture.py'):
    names.append('source/'+name)
archive=root/'dense-single-completed-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for name in names:t.add(root/name,arcname=name)
(output/'preserved-dense-evidence.json').write_text(json.dumps(dict(archive_sha256=digest(archive),bytes=archive.stat().st_size,
    selected_sha256=digest(root/'build-linear-chunk/libgarnet.dylib'),dense_candidate_sha256=digest(root/'build-dense-single-prefill/libgarnet.dylib')),indent=2)+'\n')
bundle=root/'direct-write-source.tar.gz'
assert digest(bundle)=='6494fe1de58d5f72e02ef60e8e6507fac9d1c59842783b9c2a1c725c1d3994a5'
with tempfile.TemporaryDirectory(dir=root) as directory:
    with tarfile.open(bundle) as t:t.extractall(directory,filter='data')
    temporary=Path(directory);manifest=json.loads((temporary/'manifest.json').read_text())
    for name,sha in manifest.items():assert digest(temporary/name)==sha
    for name in manifest:os.replace(temporary/name,source/name)
(output/'installed-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
original=(scripts/'compare_vllm_metal.py').read_text()
needle="if image_mode:case['image_metadata']="
assert original.count(needle)==1
updated=original.replace(needle,"if image_mode:\n            case['chunk']=64 if batch==1 else 32\n            case['prefill_linear_chunk']=True\n        "+needle)
frame_driver=root/'compare_linear_direct_write_frames.py';frame_driver.write_text(updated)
py='/opt/homebrew/bin/python3';runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
def after(label,previous,command):
    subprocess.run([py,str(scripts/'background_comparison.py'),'launch',str(root/'logs'/label),py,str(scripts/'run_after_owned_job.py'),str(root/'logs'/previous),*command],check=True)
after('linear-direct-write-build-20261013','linear-direct-write-prepare-20261013',[py,str(root/'build_linear_direct_write.py')])
check_script=root/'check_linear_direct_write.py'
check_script.write_text('''import ast,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation');s=r/'source/test2026/metal';output=r/'logs/linear-direct-write-checks-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';py='/opt/homebrew/bin/python3';build=str(r/'build-linear-direct-write');statuses=[]
commands=[('ctest',['/opt/homebrew/bin/ctest','--test-dir',build,'--output-on-failure'])]
for label,script in [('serving','test_decoder_model.py'),('mixed-image','test_image_pipeline.py')]:commands.append((label,[py,str(s/script),runtime,build,str(r/'source')]))
commands.append(('linear-reference',[py,str(s/'test_direct_write_linear_reference.py'),runtime,build,str(r/'source'),str(r/'models/Qwen3.5-0.8B'),'bf16']))
for label,command in commands:
 guard();print('Starting',label,flush=True)
 with (output/(label+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
 statuses.append(dict(label=label,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
 assert result.returncode==0,(label,result.returncode)
 guard();print('Passed',label,flush=True)
''')
after('linear-direct-write-checks-20261013','linear-direct-write-build-20261013',[py,str(check_script)])
frame=root/'logs/linear-direct-write-frame-abba-20261013'
after(frame.name,'linear-direct-write-checks-20261013',[py,str(frame_driver),runtime,str(root/'build-linear-direct-write'),str(source),str(root/'comparators/vllm/venv/bin/python'),str(frame/'jobs'),'shared',str(root/'build-dense-single-prefill'),'vl'])
after('linear-direct-write-frame-summary-20261013',frame.name,[py,str(scripts/'summarize_native_frames.py'),str(frame/'jobs')])
