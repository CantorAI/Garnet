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
    def perf_counter(self):clock[0]+=.5;return clock[0]
def request(trial,ids,start,state_handle=None):
    assert abs(sys.getswitchinterval()-.0002)<1e-12
    return dict(full_request_wall_seconds=.25,request_first_token_seconds=[.1]*2,
        decode_wall_seconds=.15,token_ids_by_request=[[7],[7]])
namespace=dict(time=Timer(),runtime_switch_us=200,RuntimeSwitchScope=RuntimeSwitchScope,
    run_request_body=request,batch=2,output_tokens=1,native_greedy_state=False)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-scheduling-wrapper','exec'),namespace)
result=namespace['run_request'](0,[1])
assert result['full_request_wall_seconds']==.5 and result['full_request_output_tokens_per_second']==4.
assert result['runtime_scheduling_finish_seconds']==.25 and result['native_greedy_state_finish_seconds']==0.
assert result['request_completion_seconds']==[.5,.5] and result['request_first_token_seconds']==[.1,.1]
assert result['runtime_scheduling']['restored_seconds']==original and result['runtime_scheduling']['applied']

# Execute the actual runner's warmup persistence expression. Compressed legacy
# records lost the scope evidence although measured trials retained it.
warmup_append=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and
    isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and
    isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='warmups' and
    n.value.func.attr=='append')
persisted=dict(measured=result,warmups=[])
exec(compile(ast.Module(body=[warmup_append],type_ignores=[]),
    'actual-warmup-persistence','exec'),persisted)
saved=persisted['warmups'][0]
assert saved['seconds']==result['full_request_wall_seconds']
assert {k:v for k,v in saved.items() if k!='seconds'}==result
assert saved['runtime_scheduling']['applied'] and saved['runtime_scheduling_finish_seconds']==.25

# The unmodified default avoids outer timing calls; the opt-in native state
# accounts for release and wrapper time as explicit parts of full latency.
def neutral_request(trial,ids,start,state_handle=None):
    return dict(full_request_wall_seconds=.5,request_first_token_seconds=[.1]*2,
        decode_wall_seconds=.4,token_ids_by_request=[[7],[7]])
default_ns=dict(time=Timer(),runtime_switch_us=0,RuntimeSwitchScope=RuntimeSwitchScope,
    run_request_body=neutral_request,batch=2,output_tokens=1,native_greedy_state=False)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-default-wrapper','exec'),default_ns)
default_result=default_ns['run_request'](0,[1])
assert default_result['full_request_wall_seconds']==.5
assert default_result['request_wrapper_finish_seconds']==0.
class StateAPI:
    def greedy_batch_state_create(self,batch,outputs):return 17
    def greedy_batch_state_release(self,handle):assert handle==17;return True
state_ns=dict(time=Timer(),runtime_switch_us=0,RuntimeSwitchScope=RuntimeSwitchScope,
    run_request_body=neutral_request,batch=2,output_tokens=1,native_greedy_state=True,G=StateAPI())
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-native-state-wrapper','exec'),state_ns)
state_result=state_ns['run_request'](0,[1])
assert state_result['native_greedy_state'] and state_result['native_greedy_state_finish_seconds']==.5
assert state_result['request_wrapper_finish_seconds']==.5
def failing(trial,ids,start,state_handle=None):raise ValueError('request-failure')
namespace['run_request_body']=failing
# Rebind the actual wrapper after replacing its double. XLang3 exec retains
# resolved callable bindings; changing the input mapping alone is insufficient.
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-scheduling-failure-wrapper','exec'),namespace)
try:namespace['run_request'](0,[1])
except ValueError:pass
else:raise AssertionError('Request failure lost')
assert sys.getswitchinterval()==original
from audit_resident_session import timed_trial, resident_host_candidates
import copy
assert not any(resident_host_candidates({}).values())
candidate_result={'optimization_environment':{'GARNET_RESIDENT_NATIVE_GREEDY_STATE':'1'},
    'resident_host_candidates':dict(reuse_output=False,native_candidate_merge=False,
        final_prefill_sample_only=False,pattern_updates=False,native_greedy_state=True)}
assert resident_host_candidates(candidate_result)['native_greedy_state']
for bad in (
    dict(candidate_result,resident_host_candidates=dict(candidate_result['resident_host_candidates'],native_greedy_state=False)),
    {'optimization_environment':{'GARNET_RESIDENT_NATIVE_GREEDY_STATE':'1'}},
    {'optimization_environment':{'GARNET_RESIDENT_NATIVE_GREEDY_STATE':'true'}}):
    try:resident_host_candidates(bad)
    except ValueError:pass
    else:raise AssertionError('Host candidate provenance was weakened')
trial=dict(full_request_wall_seconds=2.1,decode_wall_seconds=1.,prefill_seconds=1.,
    full_request_output_tokens_per_second=4/2.1,decode_aggregate_output_tokens_per_second=2.,
    request_first_token_seconds=[1.,1.],request_completion_seconds=[2.1,2.1],
    prefill_step_seconds=[.8],decode_step_seconds=[.8],runtime_scheduling_finish_seconds=.1,
    runtime_scheduling=dict(requested_microseconds=200,default_seconds=.005,
        effective_seconds=.0002,restored_seconds=.005,applied=True))
assert timed_trial(trial,2,2,3,4)==4/2.1
native_trial=dict(trial,native_greedy_state=True,native_greedy_state_finish_seconds=.05,
    request_wrapper_finish_seconds=.03,full_request_wall_seconds=2.18,
    full_request_output_tokens_per_second=4/2.18,request_completion_seconds=[2.18,2.18])
assert timed_trial(native_trial,2,2,3,4)==4/2.18
for key,value in [('native_greedy_state_finish_seconds',-.01),
        ('native_greedy_state_finish_seconds',float('nan')),('native_greedy_state','1'),
        ('request_wrapper_finish_seconds',-.01),('request_wrapper_finish_seconds',float('nan'))]:
    changed=copy.deepcopy(native_trial);changed[key]=value
    try:timed_trial(changed,2,2,3,4)
    except (ValueError,TypeError):pass
    else:raise AssertionError('Corrupted native state accounting accepted')
changed=copy.deepcopy(native_trial);changed['full_request_wall_seconds']=2.15
try:timed_trial(changed,2,2,3,4)
except ValueError:pass
else:raise AssertionError('Unaccounted state-release time accepted')
mutations=[('runtime_scheduling_finish_seconds',-.1),('runtime_scheduling_finish_seconds',float('nan')),
    ('runtime_scheduling_finish_seconds',.2),('runtime_scheduling',None)]
for key,value in mutations:
    changed=copy.deepcopy(trial);changed[key]=value
    try:timed_trial(changed,2,2,3,4)
    except (ValueError,TypeError):pass
    else:raise AssertionError('Corrupted scheduling accounting accepted')
for key,value in [('requested_microseconds',5000),('effective_seconds',.001),
        ('restored_seconds',.001),('default_seconds',float('nan')),('applied',False)]:
    changed=copy.deepcopy(trial);changed['runtime_scheduling'][key]=value
    try:timed_trial(changed,2,2,3,4)
    except (ValueError,TypeError):pass
    else:raise AssertionError('Corrupted scheduling metadata accepted')
changed=copy.deepcopy(trial);del changed['runtime_scheduling_finish_seconds']
try:timed_trial(changed,2,2,3,4)
except ValueError:pass
else:raise AssertionError('Missing finalization timing accepted')
print('Resident runtime scheduling policy/concurrent workers/failure restoration/full timer PASS')
