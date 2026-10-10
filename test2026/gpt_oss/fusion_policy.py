"""CPU-only selection tests for the optional GPT-OSS prefill composite."""
import sys
from pathlib import Path

model_dir = Path(__file__).resolve().parents[2] / 'xModel' / 'gpt_oss' / '120b'
sys.path.insert(0, str(model_dir))
from fusion_policy import (BASE_MOE_OPERATOR, FUSED_PREFILL_OPERATOR,
                           select_moe_operator)


for rank in (0, 1):
    assert select_moe_operator(True, rank, 1, '0', '0') == BASE_MOE_OPERATOR
    assert select_moe_operator(True, rank, 1, '1', '1') == FUSED_PREFILL_OPERATOR
    assert select_moe_operator(False, rank, 1, '1', '0') == BASE_MOE_OPERATOR
    assert select_moe_operator(1, rank, 1, '1', '1') == FUSED_PREFILL_OPERATOR
    assert select_moe_operator(0, rank, 1, '1', '0') == BASE_MOE_OPERATOR

assert select_moe_operator(False, -1, 0, '0', '0') == BASE_MOE_OPERATOR
assert select_moe_operator(True, 0, 1, '1', '1',
                           expert_weight_shard=0) == FUSED_PREFILL_OPERATOR
for prefill, rank, shard, enabled, peer_group in (
        (True, -1, 1, '1', '1'), (True, 0, 0, '1', '1'),
        (True, 2, 1, '1', '1'), (False, -1, 0, 'invalid', '0'),
        (True, 0, 1, 'true', '1'), (True, 0, 1, '1', '0'),
        ('0', 0, 1, '1', '1'), ('false', 0, 1, '1', '1'),
        (None, 0, 1, '1', '1'), (1.0, 0, 1, '1', '1'),
        (True, True, 1, '1', '1'), (True, 0, True, '1', '1'),
        (True, 0, 1.0, '1', '1')):
    try:
        select_moe_operator(prefill, rank, shard, enabled, peer_group)
    except ValueError:
        pass
    else:
        raise AssertionError((prefill, rank, shard, enabled))

for expert_weight_shard in (1, -1, False, True, 0.0, '0', None):
    try:
        select_moe_operator(True, 0, 1, '1', '1',
                            expert_weight_shard=expert_weight_shard)
    except ValueError:
        pass
    else:
        raise AssertionError(('expert_weight_shard', expert_weight_shard))

print('GPT-OSS fused prefill selection/default/decode eligibility PASS')
