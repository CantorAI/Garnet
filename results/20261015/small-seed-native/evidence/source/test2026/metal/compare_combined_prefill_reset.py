"""Fresh native ABBA for combined reset and ordered prefill; no new vLLM ratios."""
import ast
from pathlib import Path
ns={'__file__':__file__}
wrapper=Path(__file__).with_name('compare_prefill_stream.py').read_text()
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace('build-prefill-stream','build-combined-prefill-reset').replace('prefill-stream-checks-20261015/jobs/metadata.json','combined-prefill-reset-20261015/checks/metadata.json')
source=source.replace("assert not value['vision_global_alpha_enabled'] and not value['dirty_prefix_reset_enabled']", "assert not value['vision_global_alpha_enabled'] and value['dirty_prefix_reset_enabled']==(policy=='candidate')")
# Reuse the exact executed reset-count admission formula, not a weaker subset.
tree=ast.parse(Path(__file__).with_name('compare_dirty_prefix_reset.py').read_text())
blocks=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "if policy=='candidate':" in n.value and "dirty_prefix_reset_bytes" in n.value]
assert len(blocks)==1
needle="                assert not value['single_coder_head_enabled']"
assert source.count(needle)==1
source=source.replace(needle,blocks[0]+needle)
exec(compile(source,__file__,'exec'))
