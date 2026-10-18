"""Native Garnet SDK scope contract, no model or inference."""
import sys,json
from pathlib import Path
mode=sys.argv[1];w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-thread-qos');s=Path('/Users/shawn/CantorAI/out/research/worldsense-thread-qos/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-thread-qos-'+mode);sys.path.insert(0,str(s));sys.path.insert(0,str(b));import garnet
rows=[]
def get():return json.loads(garnet.native_thread_qos_json())
def save():(w/('scope-guard-'+mode+'.json')).write_text(json.dumps(dict(records=rows),indent=2)+'\n')
initial=get();rows.append(dict(case='initial',value=initial));save();assert initial['active_scopes']==initial['end_errors']==0
assert ('native_text_worker_user_initiated_qos'in garnet.backend_info())==(mode!='off')
outer=garnet.native_thread_qos();info=json.loads(outer.info_json());rows.append(dict(case='outer_open',value=info));save();assert info['api_start_accepted']and not info['closed']and info['during']['active_scopes']==1
inner=garnet.native_thread_qos();inside=json.loads(inner.info_json());rows.append(dict(case='inner_open',value=inside));save();assert inside['during']['active_scopes']==2 and inner.close()and inner.close();assert get()['active_scopes']==1
assert outer.close()and outer.close();closed=json.loads(outer.info_json());rows.append(dict(case='explicit_idempotent_close',value=closed));save();assert closed['closed']and closed['api_end_status']==0 and closed['after']['active_scopes']==0
caught=False
try:
 lease=garnet.native_thread_qos()
 try:raise ValueError('expected_scope_unwind')
 finally:assert lease.close()
except ValueError as e:caught=str(e)=='expected_scope_unwind'
after=get();rows.append(dict(case='exception_finally',value=after));save();assert caught and after['active_scopes']==after['end_errors']==0 and after['starts']==after['ends']==3
assert initial['requested_class']==info['during']['requested_class']==closed['after']['requested_class']==after['requested_class']
(w/('scope-guard-'+mode+'.json')).write_text(json.dumps(dict(all_native_scope_getters_nested_idempotent_finally_contract_pass=True,records=rows,scope='No model/GPU/performance proof. Requested getters do not expose overrides.'),indent=2)+'\n');print('Native Garnet scope constructor/nested/idempotent/finally/requested-class contract PASS '+mode,flush=True)
