"""Freeze a standalone selective auditor, preserving all prior numerical/timing checks."""
from pathlib import Path
import ast,sys
source=Path(__file__).with_name('verify_ndarray_prefill.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("proof=json.loads((root/'evidence/manifest.json').read_text())", """proof=json.loads((root/'evidence/manifest.json').read_text())
admission_path=Path(sys.argv[3]).resolve()
assert hashlib.sha256(admission_path.read_bytes()).hexdigest()=='f9505c75246b7ac43fbf1c28987104dd1e5d4c7101ae265e984af20ba246df97'
admission=json.loads(admission_path.read_text())
assert len(admission['models'])==3 and not admission['full_goal_met']""")
replace("sys.modules.pop('make_prefill_tile_shader',None)", """for module in ('make_prefill_tile_shader','make_ndarray_prefill','make_ndarray_selective_prefill'):
        sys.modules.pop(module,None)""")
source=source.replace("snap/'test2026/metal/make_ndarray_prefill.py'", "snap/'test2026/metal/make_ndarray_selective_prefill.py'")
source=source.replace('metal_context_ndarray_prefill.mm','metal_context_ndarray_selective_prefill.mm').replace('ndarray_prefill_gpu.mm','ndarray_selective_prefill_gpu.mm')
replace("        assert half_eligible>0 and float_eligible>0", """        assert half_eligible>0 and float_eligible>0
        full=next(v for v in admission['models'] if v['model']==model)
        assert full['checkpoint_sha256']==assets['checkpoint_sha256']
        body={c['tensor']:c for c in full['cases']}
        no_half=0
        for c in assets['cases']:
            special=c['tensor'].startswith('model.layers.') and c['tensor'].endswith(suffix)
            if special:continue
            actual=body[c['tensor']]
            assert actual['raw_sha256']==c['sha256']['raw.bin'] and (actual['rows'],actual['inner'])==(c['rows'],c['inner'])
            no_half+=not actual['exact_half_ready']
        assert no_half==(1 if model=='Qwen3.5-0.8B' else 0)""")
replace('eligible={1:half_eligible,2:float_eligible,3:0}[policy]', 'eligible={1:float_eligible,2:no_half,3:0}[policy]')
replace('(policy!=3 or check[\'bitwise_equal\'])', '(eligible!=0 or check[\'bitwise_equal\'])')
replace("if policy==3:assert check['bitwise_equal'] and check['comparison_worst']==0", "if eligible==0:assert check['bitwise_equal'] and check['comparison_worst']==0")
source=source.replace('Mixed three-source alpha/zero addend or exact F32 weight expansion;', 'Full or no-exact-half-only two-source unit-alpha F32 weight expansion;')
source=source.replace('Private independent first-layer projections;', 'Private independent first-layer projections; zero-operation changes cannot establish gain;')
ast.parse(source)
Path(sys.argv[1]).write_bytes(source.encode())
