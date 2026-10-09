"""Scheduling contracts and real concurrent rank workers; no timing claim."""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'tools/gpt_oss'))
from resident_runtime_scheduling import parse_switch_us, RuntimeSwitchScope

for value in ('0','100','200','1000'):
    assert parse_switch_us(value) == int(value)
for value in ('', '01', '200.0', '-1', '5000', 'nan', None, 200):
    try:
        parse_switch_us(value)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed policy accepted')
with RuntimeSwitchScope(0, object()) as disabled:
    assert not disabled.metadata['applied']
    assert disabled.metadata['effective_seconds'] is None

original = sys.getswitchinterval()
for interval in (100,200,1000):
    with RuntimeSwitchScope(interval) as scoped:
        assert abs(sys.getswitchinterval()-interval/1000000.) < 1e-12
        # Both real worker threads must enter before either can finish.
        barrier = threading.Barrier(2)
        def rank(r):
            barrier.wait(timeout=5)
            assert abs(sys.getswitchinterval()-interval/1000000.) < 1e-12
            return r
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(rank,r) for r in (0,1)]
            assert [f.result(timeout=5) for f in futures] == [0,1]
        try:
            with RuntimeSwitchScope(interval):
                raise AssertionError('Nested scope accepted')
        except RuntimeError as error:
            assert 'cannot overlap' in str(error)
    assert scoped.metadata['default_seconds'] == scoped.metadata['restored_seconds'] == original
    assert sys.getswitchinterval() == original
    try:
        with RuntimeSwitchScope(interval):
            raise ValueError('request-failure')
    except ValueError as error:
        assert str(error) == 'request-failure'
    assert sys.getswitchinterval() == original

class Ineffective:
    def __init__(self): self.writes=[]
    def getswitchinterval(self): return .005
    def setswitchinterval(self,value): self.writes.append(value)
runtime=Ineffective()
try:
    with RuntimeSwitchScope(200,runtime):
        raise AssertionError('Effective mismatch accepted')
except RuntimeError as error:
    assert 'not applied' in str(error)
assert runtime.writes == [.0002,.005]
try:
    with RuntimeSwitchScope(200,object()):
        raise AssertionError('Missing API accepted')
except AttributeError:
    pass
with RuntimeSwitchScope(200):
    pass  # Prior entry failures must release scope ownership.
assert sys.getswitchinterval() == original

# Exercise the actual request wrapper: set/restore time belongs to full wall,
# and a failing request cannot leak a setting into the next request.
tree=ast.parse((repo/'tools/gpt_oss/run_resident_batch_tp2.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_request')
clock=[0.]
class Timer:
    def perf_counter(self):clock[0]+=1.;return clock[0]
def request(trial,ids,start):
    assert abs(sys.getswitchinterval()-.0002)<1e-12
    return dict(full_request_wall_seconds=.5,request_first_token_seconds=[.1]*2,
        decode_wall_seconds=.4,token_ids_by_request=[[7],[7]])
namespace=dict(time=Timer(),runtime_switch_us=200,RuntimeSwitchScope=RuntimeSwitchScope,
    run_request_body=request,batch=2,output_tokens=1)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-scheduling-wrapper','exec'),namespace)
result=namespace['run_request'](0,[1])
assert result['full_request_wall_seconds']==1. and result['full_request_output_tokens_per_second']==2.
assert result['request_completion_seconds']==[1.,1.] and result['request_first_token_seconds']==[.1,.1]
assert result['runtime_scheduling']['restored_seconds']==original and result['runtime_scheduling']['applied']
def failing(trial,ids,start):raise ValueError('request-failure')
namespace['run_request_body']=failing
# Rebind the actual wrapper after replacing its double. XLang3 exec retains
# resolved callable bindings; changing the input mapping alone is insufficient.
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-scheduling-failure-wrapper','exec'),namespace)
try:namespace['run_request'](0,[1])
except ValueError:pass
else:raise AssertionError('Request failure lost')
assert sys.getswitchinterval()==original
print('Resident runtime scheduling policy/concurrent workers/failure restoration/full timer PASS')
