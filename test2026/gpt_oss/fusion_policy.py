"""CPU-only selection tests for the optional GPT-OSS prefill composite."""
import sys
from pathlib import Path

model_dir = Path(__file__).resolve().parents[2] / 'xModel' / 'gpt_oss' / '120b'
sys.path.insert(0, str(model_dir))
from fusion_policy import (BASE_MOE_OPERATOR, FUSED_PREFILL_OPERATOR,
                           select_moe_operator)


for rank in (0, 1):
    assert select_moe_operator(True, rank, 1, '0') == BASE_MOE_OPERATOR
    assert select_moe_operator(True, rank, 1, '1') == FUSED_PREFILL_OPERATOR
    assert select_moe_operator(False, rank, 1, '1') == BASE_MOE_OPERATOR

assert select_moe_operator(False, -1, 0, '0') == BASE_MOE_OPERATOR
for prefill, rank, shard, enabled in (
        (True, -1, 1, '1'), (True, 0, 0, '1'), (True, 2, 1, '1'),
        (False, -1, 0, 'invalid'), (True, 0, 1, 'true')):
    try:
        select_moe_operator(prefill, rank, shard, enabled)
    except ValueError:
        pass
    else:
        raise AssertionError((prefill, rank, shard, enabled))

print('GPT-OSS fused prefill selection/default/decode eligibility PASS')
