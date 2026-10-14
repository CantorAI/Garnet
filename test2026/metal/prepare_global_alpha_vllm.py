"""Prepare a separate, pinned full-six comparator selection after native checks."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
repo = here.parents[1]
report = json.loads((repo/'results/20261014/global-half-alpha/native/selected-vl2.json').read_text())
assert report['full_output_requests']==12 and report['all_actual_global_alpha_counters_match']
trial = report['profiles'][0]
assert trial['ttft_fraction']<1 and trial['decode_ratio']>.98
selection = dict(
    scope='Proposed global-alpha bundle for fresh full-six matched-FP32 study; no general promotion',
    basis='Two complete native trials plus selected resident-KV VL-B2 comparison; warm latency improves, decode unchanged',
    models={m:{str(b):'global_alpha' for b in (1,2)} for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')},
    builds={'global_alpha':dict(directory='build-global-half-alpha',sha256='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f',attention_prefix_enabled=True,global_half_alpha_enabled=True)},
    overall_goal_status='Unmet; all six decode/latency targets remain required',
)
destination=repo/'results/20261014/global-half-alpha/vllm'
destination.mkdir(exist_ok=True)
(destination/'selection.json').write_bytes((json.dumps(selection,indent=2)+'\n').encode())

source=(here/'compare_selected_attention_vllm.py').read_text()
source=source.replace('work/attention-prefix-next-comparator-selection.json','work/global-alpha-next-comparator-selection.json')
needle="destination=Path(sys.argv[2]).with_suffix"
assert source.count(needle)==1
source=source.replace(needle,"source=source.replace(\"assert value['attention_prefix_enabled']==selected['attention_prefix_enabled']\",\"assert value['attention_prefix_enabled']==selected['attention_prefix_enabled']\\n            assert value['global_half_alpha_enabled'] and value['prefill_profile'].get('global_half_alpha_dispatches',0)>0\\n            assert not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']\")\n"+needle)
(here/'compare_global_alpha_vllm.py').write_bytes(source.encode())
source=(here/'run_selected_attention_vllm.py').read_text().replace('compare_selected_attention_vllm.py','compare_global_alpha_vllm.py')
(here/'run_global_alpha_vllm.py').write_bytes(source.encode())
source=(here/'archive_selected_attention_vllm.py').read_text().replace("glob('selected-attention-vllm*')","glob('global-alpha-vllm-20261014*')").replace('work/attention-prefix-next-comparator-selection.json','work/global-alpha-next-comparator-selection.json')
source=source.replace("for name in names:files[", "names=names+('compare_global_alpha_vllm.py','run_global_alpha_vllm.py','archive_global_alpha_vllm.py')\nfor name in names:files[")
(here/'archive_global_alpha_vllm.py').write_bytes(source.encode())
source=(here/'verify_selected_attention_vllm.py').read_text().replace("glob('selected-attention-vllm*')","glob('global-alpha-vllm-20261014*')")
source=source.replace("assert value['attention_prefix_enabled']==chosen['attention_prefix_enabled']", "assert value['attention_prefix_enabled']==chosen['attention_prefix_enabled']\n                assert value['global_half_alpha_enabled'] and value['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n                assert not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']")
source=source.replace('print(json.dumps(dict(verified_files=', 'verification=dict(verified_files=').rstrip()
assert source.endswith('))))')
source=source[:-2]+"\nif len(sys.argv)>2:Path(sys.argv[2]).write_bytes((json.dumps(verification,indent=2)+'\\n').encode())\nprint(json.dumps(verification))\n"
(here/'verify_global_alpha_vllm.py').write_bytes(source.encode())
source=(here/'summarize_selected_attention_vllm.py').read_text().replace('selected-attention-vllm-20261013','global-alpha-vllm-20261014')
(here/'summarize_global_alpha_vllm.py').write_bytes(source.encode())
print('Prepared six-profile selection and comparator/audit helpers')
