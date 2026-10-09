"""GPU contract for the generic GreedyBatchState XLang API."""
import math
import garnet as G

def gpu(values):
    return G.tensor_from_host(values,dtype='float32',shape=[4,4],device='cuda')

handles=[G.greedy_batch_state_create(4,3) for _ in range(8)]
try:
    for bad in ((0,3),(513,3),(1,0),(1,2049),(-1,3)):
        try:G.greedy_batch_state_create(*bad)
        except Exception:pass
        else:raise AssertionError('Invalid state shape was accepted')
    try:G.greedy_batch_state_create(1,3)
    except Exception:pass
    else:raise AssertionError('Ninth live state exceeded the bounded registry')
    state=handles[0]
    first=[1,4,2,3, 5,6,5,2, math.nan,5,7,4, 9,8,8,7]
    assert G.greedy_batch_state_step(state,gpu(first))==[3,2,5,8]
    second=[1,10,2,9, 6,1,6,2, 2,5,1,3, 4,7,9,12]
    assert G.greedy_batch_state_step(state,gpu(second))==[9,2,3,12]
    third=[1,14,1,13, 2,4,5,6, 9,7,8,4, 3,10,3,9]
    assert G.greedy_batch_state_step(state,gpu(third))==[13,4,4,9]
    expected=[[3,9,13],[2,2,4],[5,3,4],[8,12,9]]
    assert G.greedy_batch_state_history(state)==expected
    try:G.greedy_batch_state_step(state,gpu(first))
    except Exception:pass
    else:raise AssertionError('State accepted more than its output limit')
finally:
    for handle in handles:assert G.greedy_batch_state_release(handle) is True
assert G.greedy_batch_state_release(handles[0]) is False
try:G.greedy_batch_state_history(handles[0])
except Exception:pass
else:raise AssertionError('Released handle remained accessible')
print('GPU native greedy-state candidate, exact ties/NaN, bounded registry/history/release PASS')
