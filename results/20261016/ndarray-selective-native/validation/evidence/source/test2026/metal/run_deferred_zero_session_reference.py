"""Reuse full-logit/private-state/recovery checks with exact clear-count admission."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_reference.py').read_text()
source=source.replace("assert 'fused_head_argmax' in garnet.backend_info()","assert 'deferred_state_zero' in garnet.backend_info()")
source=source.replace('session.fused_head_argmax_dispatches==1','session.deferred_zero_states==2')
source=source.replace('session.fused_head_argmax_dispatches==0','session.deferred_zero_states==0')
source=source.replace('plain.fused_head_argmax_dispatches==0','plain.deferred_zero_states==2')
source=source.replace("run(session,'partial',-1);assert session.deferred_zero_states==0", "run(session,'partial',-1);assert session.deferred_zero_states==2")
source=source.replace("assert rejected and session.deferred_zero_states==2,'Nonfinite prediction did not execute/detect the fused head'", "assert rejected and session.deferred_zero_states==0,'A reused state was unexpectedly cleared during a failed head'")
needle="result=run(session,'partial');assert session.deferred_zero_states==2 and result==expected(session,head)"
assert source.count(needle)==2
source=source.replace(needle,needle+"\n    assert session.deferred_zero_bytes==2*batch*c['num_key_value_heads']*capacity*c['head_dim']*4")
source=source.replace('fused_head_session_reference PASS','deferred_zero_session_reference PASS').replace('actual_fused_head=True','actual_deferred_clear=True').replace('float_head_fallback=True','generic_float_head=True')
exec(compile(source,__file__,'exec'))
