"""Original64+64 primitive assertions, with exact BF16 storage/FP32-cast generation."""
from pathlib import Path
import sys
p=Path(__file__).with_name('verify_mlp_command_buffer_primitive.py');source=p.read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("trial=sys.argv[4] if len(sys.argv)>4 else 'mlp-command-buffer-primitive-20261015'","trial='mlp-bf16-feed-primitive-20261015'")
replace("assert trial in ('mlp-command-buffer-primitive-20261015','mlp-command-buffer-chained-20261015')","assert trial=='mlp-bf16-feed-primitive-20261015'")
replace("factory=runpy.run_path(str(snap/'test2026/metal/make_mlp_command_buffer_context.py'))",'''sys.path.insert(0,str(snap/'test2026/metal'))
for name in ('make_mlp_bf16_feed_graph','make_mlp_command_buffer_context'):sys.modules.pop(name,None)
factory=runpy.run_path(str(snap/'test2026/metal/make_mlp_bf16_feed_graph.py'))''')
source=source.replace("trial=='mlp-command-buffer-chained-20261015'",'True')
replace("result=dict(verified_files=len(proof['files']),original_primitive_cases=64,",'''assert 'MPSDataTypeBFloat16' in (out/'MPSCoreTypes.h').read_text()
assert meta['weight_storage']=='original BF16 buffers'
assert meta['arithmetic']=='FP32 after exact graph cast; no activation precision change'
assert 'graphFloat' not in graph
assert 'castTensor:tensor toType:MPSDataTypeFloat32' in graph
assert 'initWithMTLBuffer:w->buffer' in graph
assert 'if(tensor.dataType!=MPSDataTypeFloat32)' in graph
result=dict(verified_files=len(proof['files']),original_primitive_cases=64,''')
exec(compile(source,__file__,'exec'))
