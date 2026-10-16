"""Private shape80 F32-only candidate from repeated trained component evidence."""
from make_ndarray_prefill import context as component_context,header

def context(source):
    source=component_context(source)
    source=source.replace('namespace Garnet {int gNDArrayPrefillPolicy=0;','namespace Garnet {')
    old='''if(a.rows>2&&!w->visionPrefill&&!w->floatPrefill&&
            ((gNDArrayPrefillPolicy==1&&w->globalHalf)||gNDArrayPrefillPolicy==2))'''
    assert source.count(old)==1
    source=source.replace(old,'if(a.rows==80&&!w->visionPrefill&&!w->floatPrefill)')
    assert source.count('const int policy=gNDArrayPrefillPolicy;')==1
    source=source.replace('const int policy=gNDArrayPrefillPolicy;','const int policy=2;')
    source=source.replace('if(ndFound==impl_->ndProducts.end()) {\n                    if(impl_->ndProducts.size()>=256)throw std::runtime_error("Private NDArray cache bound reached");',
        'if(ndFound==impl_->ndProducts.end()&&impl_->ndProducts.size()<256) {')
    source=source.replace('''                auto& item=ndFound->second;
                [item.product''','''                if(ndFound!=impl_->ndProducts.end()) {
                auto& item=ndFound->second;
                [item.product''')
    source=source.replace('''                ++impl_->ndarrayPrefillDispatches;return;
            }else throw std::runtime_error("NDArray buffer alias API requires macOS15");''','''                ++impl_->ndarrayPrefillDispatches;return;
                }
            } // Older OS and new views after the cache bound retain the original native fallback.''')
    assert 'gNDArrayPrefillPolicy' not in source and 'Private NDArray cache bound reached' not in source
    assert source.count('if(ndFound!=impl_->ndProducts.end())')==1
    return source

def session_header(source):
    from make_shape_selected_two_body import session_header as old
    return old(source).replace('twoBodyDispatches','ndarrayPrefillDispatches').replace('LastTwoBodyDispatches','LastNDArrayPrefillDispatches').replace('two_body_dispatches','ndarray_prefill_dispatches')

def session(source):
    from make_shape_selected_two_body import session as old
    return old(source).replace('twoBodyDispatches','ndarrayPrefillDispatches').replace('LastTwoBodyDispatches','LastNDArrayPrefillDispatches')

def entry(source):
    needle='info+="; coalesced_bf16_body";';assert source.count(needle)==1
    return source.replace(needle,needle+'info+="; ndarray_prefill_shape80";')

def decoder(source):
    needle="        if 'mlp_prefill_graph' in garnet.backend_info():self.prefill_profile['mlp_prefill_graph_dispatches']=0"
    assert source.count(needle)==1
    return source.replace(needle,needle+"\n        if 'ndarray_prefill_shape80' in garnet.backend_info():self.prefill_profile['ndarray_prefill_dispatches']=0")

def worker(source):
    from make_shape_selected_two_body import worker as old
    result=old(source).replace('two_body','ndarray_prefill').replace('shape_selected_ndarray_prefill','ndarray_prefill_shape80')
    needle='ndarray_prefill_counts=[decoder.session.ndarray_prefill_dispatches if ndarray_prefill_available else 0]'
    assert result.count(needle)==1
    return result.replace(needle,"ndarray_prefill_counts=[decoder.prefill_profile.get('ndarray_prefill_dispatches',0) if ndarray_prefill_available else 0]")
