"""Audit the isolated generated context, original-sized rows and measured means."""
from pathlib import Path
source=Path(__file__).with_name('verify_padded_mps_body.py').read_text().replace('import hashlib,json,math,statistics,sys','import hashlib,json,math,statistics,sys,runpy')
source=source.replace('/jobs/padded_mps_body_gpu','/jobs/direct_mps_body_gpu').replace('for physical in (4,8,16):','for physical in (batch,):').replace('padded_rows_zero','no_extra_rows')
needle="    stages=json.loads";assert source.count(needle)==1
source=source.replace(needle,'''    module=runpy.run_path(str(snap/'test2026/metal/make_direct_mps_body.py'))
    native=(snap/'src/backends/metal/metal_context.mm').read_text();cpp=(snap/'test2026/metal/padded_mps_body_gpu.mm').read_text()
    assert (out/'metal_context_direct.mm').read_bytes()==module['context'](native).encode() and digest(out/'metal_context_direct.mm')==meta['context_generated_sha256']
    assert (out/'direct_mps_body_gpu.mm').read_bytes()==module['harness'](cpp).encode() and digest(out/'direct_mps_body_gpu.mm')==meta['harness_generated_sha256']
    assert '-DGARNET_METAL_EXACT_HALF_PREFILL=1' in meta['context_macro_defines'] and '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in meta['context_macro_defines']
    for name,sha in json.loads((root/'evidence/source-manifest.json').read_text()).items():assert digest(root/'evidence/source'/name)==sha
'''+needle)
source=source.replace('Resident individual native-context body components; input padding outside timing.','Resident individual context-copy body components at original B1/B2; no padding. Generated context changes only controlled MPS admission; original raw BF16 control remains.')
exec(compile(source,__file__,'exec'))
