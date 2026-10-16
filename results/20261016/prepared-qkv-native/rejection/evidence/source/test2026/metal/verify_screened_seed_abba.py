"""Audit both same-process seed policies against original full GPU logits."""
from pathlib import Path
source=Path(__file__).with_name('verify_screened_head.py').read_text()
start=source.index('    screen=(snap/');end=source.index('    for marker in ',start)
source=source[:start]+'''    assert bits==8 and strict and meta['control_seed_rows']==8192 and meta['candidate_seed_rows']==1024
    import sys,importlib
    sys.path.insert(0,str(snap/'test2026/metal'))
    for module in ('make_screened_seed_abba','make_screened_seed_probe','make_guarded_screened_head'):sys.modules.pop(module,None)
    generator=importlib.import_module('make_screened_seed_abba')
    screen=generator.generate_shader((snap/'test2026/metal/screened_head.metal').read_text())
    harness=generator.generate_harness((snap/'test2026/metal/pruned_head_gpu.mm').read_text())
    sys.path.pop(0)
    expected='#include <metal_stdlib>\\nusing namespace metal;\\n'+raw+argmax+screen
    assert (out/'screened-head.metal').read_bytes()==expected.encode() and digest(out/'screened-head.metal')==meta['shader_sha256']
    assert (out/'screened_head_gpu.mm').read_bytes()==harness.encode() and digest(out/'screened_head_gpu.mm')==meta['harness_sha256']
    for marker in ('strictOptions.fastMathEnabled=NO','timingControl=true;run(false)','if(pruned||timingControl)','pruned?seed:controlSeed','pruned?preparePipeline:controlPrepare','pruned?prunePipeline:controlScreen','Control retained bits changed','Control rejection can beat seed','Control greedy mismatch'):assert marker in harness
'''+source[end:]
needle="        if strict:\n            fallbacks=";assert source.count(needle)==1
source=source.replace(needle,"        controls=[v for v in records if v['kind']=='control_check'];assert len(controls)==17 and {v['query'] for v in controls}=={q['name'] for q in assets['queries']}\n        assert all(v['all_values_checked'] and v['greedy_exact'] and v['batch']==next(q['batch'] for q in assets['queries'] if q['name']==v['query']) for v in controls)\n"+needle)
source=source.replace("scope='Captured-query greedy component only;","control_seed_rows=8192,candidate_seed_rows=1024,scope='Same-process exact-seed component only; timing control is screened8192, candidate screened1024; not the full raw head. Captured-query greedy component only;")
source=source.replace("if k!='profiles'","if k not in ('profiles','unsafe_input_fallback_checks')")
exec(compile(source,__file__,'exec'))
