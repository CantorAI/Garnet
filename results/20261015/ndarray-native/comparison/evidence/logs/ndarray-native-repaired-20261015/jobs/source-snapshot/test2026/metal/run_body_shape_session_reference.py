"""Trained one-layer body route, original heads, private state and recovery."""
from pathlib import Path
source=Path(__file__).with_name('run_fused_head_session_reference.py').read_text()
source=source.replace("assert 'fused_head_argmax' in garnet.backend_info()","assert 'body_shape' in garnet.backend_info()")
source=source.replace('session.fused_head_argmax_dispatches==1','session.body_shape_dispatches==7').replace('session.fused_head_argmax_dispatches==0','session.body_shape_dispatches==0').replace('plain.fused_head_argmax_dispatches==0','plain.body_shape_dispatches==7')
source=source.replace("run(session,'partial',-1);assert session.body_shape_dispatches==0", "run(session,'partial',-1);assert session.body_shape_dispatches==7")
source=source.replace("'Nonfinite prediction did not execute/detect the fused head'", "'Failed head did not preserve the completed body-route count'")
needle='    zero(session);zero(other);zero(plain)';assert source.count(needle)==1
source=source.replace(needle,needle+'''
    compiled=session.body_shape_pipeline_count
    assert 0<compiled<=128 and session.body_shape_dispatches==session.body_shape_misses==0
''')
needle="    print('fused_head_session_reference PASS'";assert source.count(needle)==1
source=source.replace(needle,'''    assert session.body_shape_misses==other.body_shape_misses==plain.body_shape_misses==0
    assert session.body_shape_pipeline_count==compiled
    session.close();assert session.body_shape_pipeline_count==session.body_shape_dispatches==session.body_shape_misses==0
    session.build([graph],head);zero(session)
    assert session.body_shape_pipeline_count==compiled
    assert run(session,'partial')==expected(session,head) and session.body_shape_dispatches==7 and session.body_shape_misses==0
    print('body_shape_session_reference PASS' ''')
source=source.replace('actual_fused_head=True','actual_body_shape=True,eager_pipeline_cache=True,head_excluded=True,rebuild_and_close=True').replace('float_head_fallback=True','generic_float_head=True')
exec(compile(source,__file__,'exec'))
