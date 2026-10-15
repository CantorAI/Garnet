"""Isolated source/build, unit gate and six full-output diagnostic ABBA profiles."""
from pathlib import Path
base=Path(__file__).with_name('run_joined_prefill_checks.py').read_text()
exec(compile(base[:base.index("assert not build.exists()")],__file__,'exec'))
import shutil
build=r/'build-counter-trace-file';copy=r/'counter-trace-file-source'
assert not build.exists() and not copy.exists()
assert shutil.disk_usage(r).free>=2*2**30
copy.mkdir();shutil.copy2(r/'source/CMakeLists.txt',copy/'CMakeLists.txt')
for name in ('src','cmake','third_party'):shutil.copytree(r/'source'/name,copy/name)
(copy/'test2026/metal').mkdir(parents=True)
shutil.copy2(s/'matmul_test.cpp',copy/'test2026/metal/matmul_test.cpp')
original=hashlib.sha256((copy/'src/backends/metal/metal_context.mm').read_bytes()).hexdigest()
assert original=='b0cf78db529b4837022a66ac06db879774e114a73285d43065582124b505f480'
run('generate',[sys.executable,str(s/'make_counter_trace_source.py'),str(r/'source/src/backends/metal/metal_context.mm'),str(copy/'src/backends/metal/metal_context.mm')])
identities={str(p.relative_to(copy)):hashlib.sha256(p.read_bytes()).hexdigest() for p in copy.rglob('*') if p.is_file()}
(output/'source-identities.json').write_text(json.dumps(dict(original_metal_context_sha256=original,diagnostic_source_sha256=identities),indent=2))
reference=json.loads((r/'logs/global-half-alpha-checks-20261014/jobs/metadata.json').read_text())
options=reference['options'];assert 'METAL_GLOBAL_HALF_ALPHA' in options and 'METAL_PREFILL_STREAM' not in options and 'METAL_DIRTY_PREFIX_RESET' not in options
os.environ['GARNET_METAL_PROFILE_OPS']='0';os.environ['GARNET_METAL_COUNTER_TRACE']='0'
command=['/opt/homebrew/bin/cmake','-S',str(copy),'-B',str(build)]+['-DGARNET_'+x+'=ON' for x in options]+['-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release']
run('configure',command);run('build',['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4']);run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])
(output/'metadata.json').write_text(json.dumps(dict(scope=__doc__,library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),options=options,source_sha256={name:hashlib.sha256((s/name).read_bytes()).hexdigest() for name in ('run_counter_trace_profiles.py','make_counter_trace_source.py','counter_trace_helpers.inc','run_matched_garnet.py','frame_worker_garnet.py','run_matched_frame_process.py')}),indent=2))
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';frontend=str(r/'comparators/vllm/venv/bin/python');profiles=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for batch in (1,2):
        label=model+'-b'+str(batch);vl=model=='Qwen3.5-0.8B'
        baseline=r/'logs'/('equivalent-f32-qwen35-extended-prefill-20261013' if vl else 'equivalent-f32-finer-prefill-20261013')/'jobs'
        case=json.loads((baseline/(label+'-case.json')).read_text());case['requests']=3
        if not vl:case['prefill_exact_tail']=model!='Qwen3-0.6B' or batch==2
        casefile=output/(label+'-case.json');casefile.write_text(json.dumps(case,indent=2))
        if vl:expected=json.loads((baseline/(label+'-garnet-0/protocol/command.json.event-0-frame_result.json')).read_text())
        else:expected=next(json.loads(line.split(' ',1)[1]) for line in (baseline/(label+'-garnet-0/output.log')).read_text().splitlines() if line.startswith('matched_result '))
        (output/(label+'-expected.json')).write_text(json.dumps(expected,indent=2))
        for tracing,repeat in ((False,0),(True,0),(True,1),(False,1)):
            name=label+'-'+('traced' if tracing else 'quiet')+'-'+str(repeat);folder=output/name;folder.mkdir()
            os.environ['GARNET_METAL_COUNTER_TRACE']='1' if tracing else '0'
            tracefile=folder/'gpu-counters.jsonl';os.environ['GARNET_METAL_COUNTER_TRACE_FILE']=str(tracefile)
            if vl:command=[frontend,str(s/'run_matched_frame_process.py'),runtime,str(build),str(r/'source'),str(r/'models'/model),str(casefile),'garnet',str(folder/'protocol')]
            else:command=[runtime,str(s/'run_matched_garnet.py'),str(build),str(r/'source'),str(r/'models'/model),str(casefile)]
            run(name,command)
            text=(output/(name+'.log')).read_text();prefix='matched_frame_result ' if vl else 'matched_result '
            records=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith(prefix)];assert len(records)==3
            for value in records:
                assert all(value[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished')) and value['case']==case
                if vl:assert value['vision_reencoded'] and value['patch_sha256']==case['image_metadata']['patch_sha256']
                assert value['global_half_alpha_enabled'] and not value['prefill_stream_enabled'] and not value['dirty_prefix_reset_enabled']
            if tracing:assert tracefile.is_file();raw=tracefile.read_text()
            else:assert not tracefile.exists();raw=''
            samples=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith('metal_counter_profile ')]
            commands=[json.loads(line.split(' ',1)[1]) for line in raw.splitlines() if line.startswith('metal_counter_command ')]
            assert bool(samples)==tracing and bool(commands)==tracing
            for c in commands:
                assert c['diagnostic'] and not c['serialized'] and c['partial_submissions']==0
                events=[e for e in samples if (e['context'],e['command'])==(c['context'],c['command'])]
                assert [e['index'] for e in events]==list(range(c['samples']))
                assert all(e['diagnostic'] and not e['serialized'] and 0<e['start_raw']<=e['end_raw'] for e in events)
            profiles.append(dict(model=model,batch=batch,tracing=tracing,repeat=repeat,requests=3,complete_outputs_match=True,records=records,counter_commands=len(commands),counter_samples=len(samples)))
            (output/'profiles.json').write_text(json.dumps(dict(scope='Raw compute-pass diagnostic, not matched serving speed; MPS internals are unmeasured and clock intervals may overlap',profiles=profiles),indent=2))
assert len(profiles)==24 and sum(p['requests'] for p in profiles)==72
print('Counter trace complete outputs and contiguous sample records PASS 72',flush=True)
