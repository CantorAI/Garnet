"""Independent generated-kernel, exact-output and ABBA timing audit."""
from pathlib import Path
from make_fused_projection_group import generate
source=Path(__file__).with_name('verify_concurrent_bf16_groups.py').read_text()
start=source.index("    shader=('#include <metal_stdlib>")
end=source.index("    stages=json.loads",start)
source=source[:start]+'''    original=(out/'source-snapshot/test2026/metal/concurrent_bf16_group_gpu.mm').read_text()
    shader,harness=generate(native,original)
    assert (out/'fused-projections.metal').read_bytes()==shader.encode() and digest(out/'fused-projections.metal')==meta['native_shader_sha256']
    assert (out/'fused_projection_group_gpu.mm').read_bytes()==harness.encode() and digest(out/'fused_projection_group_gpu.mm')==meta['generated_harness_sha256']
    assert 'Projection or padding changed bits' in harness and 'Original FP64 envelope exceeded' in harness
    assert proof['omitted_binaries']['logs/'+trial+'/jobs/fused_projection_group_gpu']['sha256']==meta['binary_sha256']
'''+source[end:]
source=source.replace("c['validated_executions']>=252","c['validated_executions']>=136").replace('for mode in (1,2):','for mode in (1,):').replace("len(summary['profiles'])==32","len(summary['profiles'])==16")
source=source.replace('Independent projection component only, original shader bits; concurrent API request does not prove hardware overlap; no full inference or vLLM gain','Single-dispatch independent projection component only; original shader output bits, unchanged FP64 bounds; no full inference or vLLM gain')
exec(compile(source,__file__,'exec'))
