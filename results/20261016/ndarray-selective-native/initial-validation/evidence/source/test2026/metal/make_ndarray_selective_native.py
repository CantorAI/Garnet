"""Native B1/B2 VL candidate: F32 NDArray only when original exact half is unavailable."""
from make_ndarray_selective_prefill import context as component_context,header
from make_ndarray_native import session,session_header

def context(source):
    source=component_context(source)
    source=source.replace('namespace Garnet {int gNDArrayPrefillPolicy=0;','namespace Garnet {')
    old='''if(a.rows>2&&!w->visionPrefill&&!w->floatPrefill&&
            (gNDArrayPrefillPolicy==1||(gNDArrayPrefillPolicy==2&&(!w->exactHalf||!w->halfScales))))'''
    assert source.count(old)==1
    source=source.replace(old,'if((a.rows==80||a.rows==160)&&!w->visionPrefill&&!w->floatPrefill&&(!w->exactHalf||!w->halfScales))')
    assert 'gNDArrayPrefillPolicy' not in source
    return source

def entry(source):
    from make_ndarray_native import entry as original
    return original(source).replace('ndarray_prefill_shape80','ndarray_prefill_selective_80_160')

def decoder(source):
    from make_ndarray_native import decoder as original
    return original(source).replace('ndarray_prefill_shape80','ndarray_prefill_selective_80_160')

def worker(source):
    from make_ndarray_native import worker as original
    return original(source).replace('ndarray_prefill_shape80','ndarray_prefill_selective_80_160')
