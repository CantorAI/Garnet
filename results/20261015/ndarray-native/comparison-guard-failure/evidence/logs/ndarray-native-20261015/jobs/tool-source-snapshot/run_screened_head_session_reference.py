"""Actual trained session greedy parity, generic logits, fallback and recovery."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_reference.py').read_text().replace('fused_head_argmax','screened_head').replace('fused_head_session_reference PASS','screened_head_session_reference PASS').replace('actual_fused_head','actual_screened_head')
needle="    print('screened_head_session_reference PASS'";assert source.count(needle)==1
source=source.replace(needle,'''    session.close();assert session.screened_head_dispatches==0
    rejected=False
    try:run(session,'partial')
    except Exception:rejected=True
    assert rejected,'Closed screened session executed'
    session.build([graph],head);zero(session);assert session.screened_head_dispatches==0
    result=run(session,'partial');assert session.screened_head_dispatches==1 and result==expected(session,head)
    assert states(other)==saved
'''+needle)
source=source.replace('private_sessions_isolated=True','private_sessions_isolated=True,rebuild_and_close=True')
exec(compile(source,__file__,'exec'))
