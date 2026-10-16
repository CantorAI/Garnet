"""Compare full F32 NDArray with only the original no-exact-half fallback.

Policies 1/2 use identical two-source unit-alpha F32 products. Policy 2
retains every admitted common/row-wise exact-half route. Policy 3 is native.
No source weights, arithmetic precision or numerical bounds are changed.
"""
from make_ndarray_prefill import context as original_context, header, harness as original_harness


def context(source):
    source = original_context(source)
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    replace('((gNDArrayPrefillPolicy==1&&w->globalHalf)||gNDArrayPrefillPolicy==2)',
            '(gNDArrayPrefillPolicy==1||(gNDArrayPrefillPolicy==2&&(!w->exactHalf||!w->halfScales)))')
    replace('const int policy=gNDArrayPrefillPolicy;', 'const int policy=2;')
    replace('''if(ndFound==impl_->ndProducts.end()) {
                    if(impl_->ndProducts.size()>=256)throw std::runtime_error("Private NDArray cache bound reached");''',
            'if(ndFound==impl_->ndProducts.end()&&impl_->ndProducts.size()<256) {')
    replace('''                auto& item=ndFound->second;
                [item.product''', '''                if(ndFound!=impl_->ndProducts.end()) {
                auto& item=ndFound->second;
                [item.product''')
    replace('''                ++impl_->ndarrayPrefillDispatches;return;
            }else throw std::runtime_error("NDArray buffer alias API requires macOS15");''',
            '''                ++impl_->ndarrayPrefillDispatches;return;
                }
            } // Older OS and new views after the cache cap keep the original native fallback.''')
    replace('''                            for(size_t i=0;i<values.size();++i){uint16_t bits;std::memcpy(&bits,static_cast<const char*>(w->buffer.contents)+i*2,2);''',
            '''                            const char* original=static_cast<const char*>(w->buffer.contents);
                            for(size_t i=0;i<values.size();++i){uint16_t bits;std::memcpy(&bits,original+i*2,2);''')
    return source


def harness(source):
    source = original_harness(source)
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    replace('p.selected=p.weight.GlobalHalfReady();', 'p.selected=p.weight.ExactHalfReady();')
    replace('expected+=policy==1?p.selected:policy==2?!p.floatKey:0;',
            'expected+=policy==1?!p.floatKey:policy==2?(!p.selected&&!p.floatKey):0;')
    replace('selected+=policy==1?p.selected:policy==2?!p.floatKey:0;',
            'selected+=policy==1?!p.floatKey:policy==2?(!p.selected&&!p.floatKey):0;')
    replace('(policy!=3||bitwise)', '(selected!=0||bitwise)')
    replace('need(selected>0||policy==3,"No NDArray projection selected");',
            'need(selected>0||policy==2||policy==3,"No NDArray projection selected");')
    return source
