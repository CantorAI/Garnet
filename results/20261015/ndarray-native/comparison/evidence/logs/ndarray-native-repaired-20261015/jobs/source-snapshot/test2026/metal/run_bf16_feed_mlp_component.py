"""Trained dependent MLP casts; first/setup separate and original gates retained."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_stage_rounded_mlp_graph.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
source=source.replace('make_stage_rounded_mlp_graph.py','make_bf16_feed_mlp_component.py')
source=source.replace('stage_rounded_mlp_graph_gpu','bf16_feed_mlp_graph_gpu')
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,needle+"s/'run_stage_rounded_mlp_graph.py',")
needle="    checks=[v for v in values if v['kind']=='check']";assert source.count(needle)==1
source=source.replace(needle,'''    casts=[v for v in values if v['kind']=='weight_cast']
    learned={c['tensor']:c['rows']*c['inner'] for c in info['cases'] if '.mlp.' in c['tensor']}
    assert len(casts)==len(learned) and {v['tensor'] for v in casts}==set(learned)
    assert all(v['bitwise_equal'] and v['values']==learned[v['tensor']] for v in casts)
'''+needle)
needle="(out/'metadata.json').write_bytes";at=source.index(needle)
source=source[:at]+"metadata.update(scope='Complete dependent MLP with original BF16 placeholder/constant storage, exact FP32 casts and unchanged FP32 arithmetic. Mandatory GPU roundtrip of every learned weight bit before timing.',limitations='Original stage-rounded reference and native comparison bounds; all failed profiles excluded without changed coefficients. One native submission or synchronous graph call per MLP; no native command-buffer integration, full-model TTFT, memory saving or vLLM result.')\n"+source[at:]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
