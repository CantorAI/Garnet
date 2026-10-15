"""Audit true MLP dependencies, full FP64 envelopes and original timing gates."""
from pathlib import Path

source=Path(__file__).with_name('verify_combined_graph_prefill.py').read_text()
source=source.replace("check['values']==rows*sum(c['rows'] for c in assets['cases']) and check['projections']==len(assets['cases'])", "check['values']==rows*sum(c['rows'] for c in assets['cases'] if c['tensor'].endswith('.mlp.down_proj.weight')) and check['projections']==sum('.mlp.' in c['tensor'] for c in assets['cases'])")
source=source.replace("assert check['control_fp64_worst']<=1 and check['passed']==(check['finite'] and check['fp64_worst']<=1 and check['comparison_worst']<=1)", "assert check['passed']==(check['finite'] and check['control_fp64_worst']<=1 and check['fp64_worst']<=1 and check['comparison_worst']<=1)")
needle="                    assert check['fp32_activation_weight_product_output']";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                    assert check['mlp_blocks']==(2 if model=='Qwen3.5-0.8B' else 1) and check['dependent_matmuls']==3*check['mlp_blocks']")
source=source.replace('All independent first-layer projections in one native submission or one graph call. Original FP32 tensor values;', 'Complete gate/up-SiLU-down MLP dependencies, four trained blocks. Full CPU FP64 trained-layer envelope and original native comparison; original FP32 tensor values;')
exec(compile(source,__file__,'exec'))
