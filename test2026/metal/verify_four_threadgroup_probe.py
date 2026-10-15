"""Original per-matrix exact/numerical/timing assertions, same four-output shader."""
from pathlib import Path
p=Path(__file__).with_name('verify_two_matrix_probe.py');source=p.read_text()
old="source=ns['source'].replace('make_two_body','make_two_matrix_probe').replace('coalesced_two_body_gpu','two_matrix_probe_gpu')"
new="source=ns['source'].replace('make_two_body','make_four_threadgroup_probe').replace('coalesced_two_body_gpu','four_threadgroup_gpu').replace('metal_context_coalesced_two','metal_context_four_threadgroup').replace('coalesced_two_body_shader.h','retained_four_shader.h')"
assert source.count(old)==1;source=source.replace(old,new)
old="('make_prefill_tile_shader','make_two_matrix_probe','make_two_body','make_coalesced_eight_body')"
assert source.count(old)==1
source=source.replace(old,"('make_prefill_tile_shader','make_four_threadgroup_probe','make_two_matrix_probe','make_two_body','make_coalesced_eight_body')")
source=source.replace("    summary=json.loads((out/'summary.json').read_text())",'''    summary=json.loads((out/'summary.json').read_text())
    assert meta['new_shader_compiled'] is False
    generated=(out/'metal_context_four_threadgroup.mm').read_text()
    original=(out/'source-snapshot/src/backends/metal/metal_context.mm').read_text()
    assert generated.count('bf16BodyCoalescedBlock;')==original.count('bf16BodyCoalescedBlock;')
    assert 'bf16BodyCoalescedTwo' not in generated and 'bf16BodyCoalescedEight' not in generated
    assert (out/'retained_four_shader.h').read_bytes()==(out/'source-snapshot/src/backends/metal/coalesced_body_block_shader.h').read_bytes()''')
source=source.replace('Per-selected-matrix independent repeated-resident probes;', 'Same original four-output shader/pipeline at64/128/256 threads.64-thread candidate is a null comparison. Per-selected-matrix repeated-resident probes;')
exec(compile(source,__file__,'exec'))
