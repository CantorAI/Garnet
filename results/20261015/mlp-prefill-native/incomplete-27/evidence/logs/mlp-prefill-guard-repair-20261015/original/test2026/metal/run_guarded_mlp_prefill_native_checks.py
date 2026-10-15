"""Run the original28 numerical/recovery stages and the new native graph primitive."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_guarded_coalesced_body_native_checks.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-coalesced-bf16-body','build-mlp-prefill-graph')
source=source.replace("'METAL_COALESCED_BF16_BODY'","'METAL_COALESCED_BF16_BODY','METAL_MLP_PREFILL_GRAPH'")
exec(compile(source,__file__,'exec'))
