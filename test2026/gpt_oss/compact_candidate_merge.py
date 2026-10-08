"""Actual native CPU/CUDA compact greedy selections, exact IDs/ties and score bits."""
import json
import math
from pathlib import Path
import struct
import sys
import garnet as G

output = Path(sys.argv[1])
if output.exists() or output.with_suffix('.partial.json').exists():
    raise FileExistsError(output)
records, negatives = [], []

def bits(value):
    return struct.pack('<f', value).hex()

def save(partial=True):
    path = output.with_suffix('.partial.json') if partial else output
    path.write_text(json.dumps(dict(records=records, negatives=negatives,
        complete=not partial), indent=2))

def finite_oracle(row):
    # An order-statistic oracle independent of the production branch comparator.
    choices = [(row[0], int(row[1])), (row[2], int(row[3]))]
    return sorted(choices, key=lambda pair: (-pair[0], pair[1]))[0]

def check(tensor, flat, device, batch, changed):
    reply = G.merge_greedy_candidate_pairs(tensor)
    actual = dict(device=device, batch=batch, changed=changed, input=flat,
        ids=reply['token_ids'], values_bits=[bits(v) for v in reply['token_values']])
    records.append(actual)
    save()
    wanted = [finite_oracle(flat[i:i + 4]) for i in range(0, len(flat), 4)]
    assert reply['status'] == 'ok' and reply['token_id'] == wanted[0][1]
    assert actual['ids'] == [v[1] for v in wanted]
    assert actual['values_bits'] == [bits(v[0]) for v in wanted]
    # The native read-only observation must not modify either source scores/IDs.
    assert [bits(v) for v in G.tensor_to_cpu(tensor).tolist()] == [bits(v) for v in flat]

def rejected(tensor, label):
    caught = False
    try:
        G.merge_greedy_candidate_pairs(tensor)
    except Exception:
        caught = True
    negatives.append(dict(label=label, rejected=caught))
    save()
    assert caught, label

devices = json.loads(G.cuda_devices_json())[:2]
if len(devices) != 2:
    raise RuntimeError('Both rank GPUs required for native merge qualification')
for device in [None] + devices:
    previous = G.cuda_set_device(device['id']) if device is not None else None
    place = 'cpu' if device is None else 'cuda'
    label = 'cpu' if device is None else device['id']
    try:
        for batch in (1, 7, 224, 512, 4096):
            families = [[2., 17., 1., 4.], [1., 17., 2., 4.], [3., 17., 3., 4.],
                [-3., 0., -3., float((1 << 24) - 1)], [-0., 9., 0., 9.]]
            flat = [v for i in range(batch) for v in families[i % len(families)]]
            tensor = G.tensor_from_host(flat, dtype='float32', shape=[batch, 1, 4], device=place)
            check(tensor, flat, label, batch, False)
            changed = [v for i in range(batch) for v in families[(i + 1) % len(families)]]
            assert G.tensor_update_from_host(tensor, changed)
            check(tensor, changed, label, batch, True)
        for bad_id in (-1., .5, float(1 << 24), float('inf'), float('nan')):
            rejected(G.tensor_from_host([1., bad_id, 2., 0.], dtype='float32',
                shape=[1, 1, 4], device=place), str(label) + '-invalid-id-' + str(bad_id))
        rejected(G.tensor_from_host([0.] * 3, dtype='float32', shape=[3], device=place), str(label) + '-bad-count')
        rejected(G.tensor_from_host([], dtype='float32', shape=[0], device=place), str(label) + '-empty')
        rejected(G.tensor_from_host([0] * 4, dtype='int32', shape=[4], device=place), str(label) + '-bad-dtype')
        # Preserve old score comparator behavior for non-finite scores. NaN
        # does not outrank another candidate, and identical infinities tie by ID.
        special = [float('nan'), 17., 9., 4., 9., 17., float('nan'), 4.,
            float('inf'), 17., float('inf'), 4., -float('inf'), 17., -float('inf'), 4.]
        tensor = G.tensor_from_host(special, dtype='float32', shape=[4, 4], device=place)
        reply = G.merge_greedy_candidate_pairs(tensor)
        assert reply['token_ids'] == [17, 17, 4, 4]
        assert math.isnan(reply['token_values'][0]) and reply['token_values'][1] == 9.
        assert reply['token_values'][2] == float('inf') and reply['token_values'][3] == -float('inf')
        negatives.append(dict(label=str(label) + '-nonfinite-score-comparator', rejected=False, parity=True))
        save()
    finally:
        if previous is not None:
            G.cuda_set_device(previous)
rejected([1., 2., 3., 4.], 'not-a-tensor')
save(False)
print('Native compact merge exact selections/score bits, source immutability, changed inputs and negatives passed', flush=True)
