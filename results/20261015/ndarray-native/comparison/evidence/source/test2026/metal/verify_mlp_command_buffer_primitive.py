"""Independently audit frozen generation, compilation, 64 original safety cases."""
from pathlib import Path
import hashlib,json,runpy,sys

base=Path(sys.argv[1]).resolve();root=base/'evidence'
proof=json.loads((root/'manifest.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    path=(base/name).resolve();assert path.is_relative_to(base)
    assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256']
trial=sys.argv[4] if len(sys.argv)>4 else 'mlp-command-buffer-primitive-20261015'
assert trial in ('mlp-command-buffer-primitive-20261015','mlp-command-buffer-chained-20261015')
assert proof['jobs']=={trial:dict(exit_code=0,wall_seconds=proof['jobs'][trial]['wall_seconds'])}
out=root/'logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text())
snap=out/'source-snapshot'
for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
factory=runpy.run_path(str(snap/'test2026/metal/make_mlp_command_buffer_context.py'))
native=snap/'src/backends/metal'
for name,operation,source in (
    ('metal_context.mm','context',native/'metal_context.mm'),
    ('mlp_prefill_graph_impl.h','graph',native/'mlp_prefill_graph_impl.h'),
    ('mlp_command_buffer_primitive.cpp','primitive',snap/'test2026/metal/mlp_prefill_graph_test.cpp')):
    options=dict(chained=True) if trial=='mlp-command-buffer-chained-20261015' and operation=='primitive' else {}
    assert (out/name).read_bytes()==factory[operation](source.read_text(),**options).encode()
    assert digest(out/name)==meta['generated_sha256'][name]
assert (out/'metal_context.h').read_bytes()==(native/'metal_context.h').read_bytes()
assert digest(out/'metal_context.h')==meta['generated_sha256']['metal_context.h']
for name,sha in meta['sdk_headers'].items():assert digest(out/name)==sha
assert 'commitAndContinue' in (out/'MPSGraph.h').read_text()
assert 'rootCommandBuffer' in (out/'MPSCommandBuffer.h').read_text()
assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string']
assert int(meta['hardware']['hw.memsize'])==16*2**30
assert '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' in meta['defines']
assert '-DGARNET_METAL_QWEN35_WIDE_BATCH_PREFILL=1' in meta['defines']
stages=json.loads((out/'statuses.json').read_text())
assert [s['name'] for s in stages]==['compile','primitive']
assert all(s['exit_code']==0 for s in stages)
for stage in stages:
    monitor=json.loads((out/(stage['name']+'-monitor.json')).read_text())
    assert monitor and all(s['guard_passed'] for s in monitor)
command=stages[0]['command'];assert '-framework' in command and 'MetalPerformanceShadersGraph' in command
binary=Path(command[command.index('-o')+1]).name
assert proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']==meta['binary_sha256']
expected='MLP command-buffer graph producer/consumer, ownership, abort/recovery and original native bound PASS: 64 cases\n'
if trial=='mlp-command-buffer-chained-20261015':
    expected='MLP command-buffer dependent two-graph chains PASS: 64 cases\n'+expected
assert (out/'primitive.log').read_text()==expected
test=(out/'mlp_command_buffer_primitive.cpp').read_text()
assert 'size_t(160)' not in test and 'rows.push_back(160)' in test
assert '5e-6+5e-5*std::abs(double(expected[i]))' in test
assert test.count('compare(gpu.Download(consumer),expectedConsumer);')==2
assert 'LastPartialSubmissions()==0,"Command-buffer graph/native-flush counts wrong"' in test
graph=(out/'mlp_prefill_graph_impl.h').read_text()
assert 'encodeToCommandBuffer:buffer' in graph and 'descriptor.waitUntilCompleted=NO' in graph
assert 'runAsyncWithMTLCommandQueue:' not in graph and '    Flush();' not in graph
result=dict(verified_files=len(proof['files']),original_primitive_cases=64,
    dependent_two_graph_cases=64 if trial=='mlp-command-buffer-chained-20261015' else 0,
    native_consumers_per_case=True,explicit_native_flushes_per_graph=0,
    original_native_comparison_bound='5e-6+5e-5*abs(native)',
    scope='Private primitive numerical/order/ownership evidence only; no trained model or performance result',
    full_goal_met=False)
if len(sys.argv)>3:
    with Path(sys.argv[3]).open('rb') as stream:result['archive_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(result))
