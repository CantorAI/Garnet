"""Memory limits, unequal GPUs, stable cache identity and insufficient VRAM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from garnet_pipeline import plan_layers


def device(id, size):
    return dict(id=id, name='synthetic', total_bytes=size, free_bytes=size,
                compute_major=8, compute_minor=9, pci_bus_id=str(id), peer_access=[False, False])


devices = [device(0, 100), device(1, 300)]
plan = plan_layers(devices, [70, 70, 70, 70], reserve_bytes=0, memory_fraction=1)
assert [(s['start'], s['end']) for s in plan['stages']] == [(0, 1), (1, 4)]
assert all(s['estimated_bytes'] <= s['budget_bytes'] for s in plan['stages'])
devices[1]['free_bytes'] = 290
assert plan_layers(devices, [70] * 4, reserve_bytes=0, memory_fraction=1)['cache_key'] == plan['cache_key']
devices[1]['compute_minor'] = 6
assert plan_layers(devices, [70] * 4, reserve_bytes=0, memory_fraction=1)['cache_key'] != plan['cache_key']
for devices, layers in [([device(0, 100), device(1, 100)], [70] * 4),
                         ([device(0, 100), device(0, 100)], [1, 1]),
                         ([device(0, 100), device(1, 100)], [1])]:
    try:
        plan_layers(devices, layers, reserve_bytes=0, memory_fraction=1)
        raise AssertionError('invalid placement accepted')
    except ValueError:
        pass
print('placement-tests-passed')
