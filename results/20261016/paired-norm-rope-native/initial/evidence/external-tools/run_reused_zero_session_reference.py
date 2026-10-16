"""Trained session parity, private resets, poison recovery, rebuild and close."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_reference.py').read_text()
source=source.replace("assert 'fused_head_argmax' in garnet.backend_info()", "assert 'reused_zero_source' in garnet.backend_info()")
source=source.replace('session.fused_head_argmax_dispatches==1','session.reused_zero_copy_states==2').replace('session.fused_head_argmax_dispatches==0','session.reused_zero_copy_states==2').replace('plain.fused_head_argmax_dispatches==0','plain.reused_zero_copy_states==2')
source=source.replace("'Nonfinite prediction did not execute/detect the fused head'", "'Nonfinite prediction was not rejected or reset counts changed'")
needle="    zero(session);zero(other);zero(plain)"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
    expected_source=batch*c['num_key_value_heads']*capacity*c['head_dim']*4
    def counts(s):
        assert s.reused_zero_source_bytes==expected_source
        assert s.reused_zero_copy_bytes==2*expected_source and s.reused_zero_copy_states==2
    counts(session);counts(other);counts(plain)
''')
needle="    print('fused_head_session_reference PASS'"
assert source.count(needle)==1
source=source.replace(needle,'''    counts(session);counts(other);counts(plain)
    session.build([graph],head);zero(session);counts(session)
    assert run(session,'partial')==expected(session,head)
    assert states(other)==saved
    session.close()
    assert session.reused_zero_source_bytes==session.reused_zero_copy_bytes==session.reused_zero_copy_states==0
    session.build([graph],head);zero(session);counts(session)
    assert run(session,'partial')==expected(session,head)
    print('reused_zero_session_reference PASS' ''')
source=source.replace('actual_fused_head=True','actual_reused_zero_source=True,rebuild_and_close=True').replace('float_head_fallback=True','generic_float_head=True')
exec(compile(source,__file__,'exec'))
