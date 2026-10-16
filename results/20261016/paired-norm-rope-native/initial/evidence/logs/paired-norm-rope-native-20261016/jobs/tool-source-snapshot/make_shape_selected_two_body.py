"""Confined native integration of three data-selected two-output geometries."""
from make_two_body import context as two_context,header as two_header,shader

TABLE=((1,4096,1024,64),(2,896,4864,64),(2,6144,1024,128))

def context(source):
    result=two_context(source)
    declaration='namespace Garnet {bool gCoalescedBlockEnabled=false;size_t gCoalescedBlockThreads=64;'
    assert result.count(declaration)==1
    result=result.replace(declaration,'namespace Garnet {')
    start=result.index('        if(coalesced&&gCoalescedBlockEnabled){')
    end=result.index('\n        }',start)+len('\n        }')
    cases='\n'.join(f'            if(a.rows=={b}&&o.columns=={n}&&a.columns=={k})selectedThreads={t};' for b,n,k,t in TABLE)
    result=result[:start]+'''        if(coalesced){
            size_t selectedThreads=0;
'''+cases+'''
            if(selectedThreads){pipeline=impl_->kernels->bf16BodyCoalescedTwo;threads=selectedThreads;decodeColumns=threads/32*2;++impl_->twoBodyDispatches;}
        }'''+result[end:]
    assert 'gCoalescedBlock' not in result
    return result

def header(source):return two_header(source)

def session_header(source):
    source=source.replace('coalescedBlockDispatches_=0,','coalescedBlockDispatches_=0,twoBodyDispatches_=0,')
    needle='    size_t LastCoalescedBlockDispatches() const {return coalescedBlockDispatches_;}'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'\n    size_t LastTwoBodyDispatches() const {return twoBodyDispatches_;}')
    needle='        APISET().AddProp("coalesced_block_dispatches",&MetalDecoderSession::LastCoalescedBlockDispatches);'
    assert source.count(needle)==1
    return source.replace(needle,needle+'\n        APISET().AddProp("two_body_dispatches",&MetalDecoderSession::LastTwoBodyDispatches);')

def session(source):
    assert source.count('coalescedBlockDispatches_=0;')==4
    source=source.replace('coalescedBlockDispatches_=0;','coalescedBlockDispatches_=0;twoBodyDispatches_=0;')
    needle='coalescedBlockDispatches_=gpu_->LastCoalescedBlockDispatches();'
    assert source.count(needle)==1
    return source.replace(needle,needle+'twoBodyDispatches_=gpu_->LastTwoBodyDispatches();')

def worker(source):
    start=source.index('coalesced_block_enabled=');end=source.index('\n',start)
    indent=source[source.rfind('\n',0,start)+1:start]
    # The private library exports this property. Control builds do not.
    source=source[:end]+"\n"+indent+"two_body_available='shape_selected_two_body' in garnet.backend_info()"+source[end:]
    needle='        coalesced_block_counts=[';start=source.index(needle);end=source.index('\n',start)
    source=source[:end]+"\n        two_body_counts=[decoder.session.two_body_dispatches if two_body_available else 0]"+source[end:]
    needle='            coalesced_block_counts.append(';start=source.index(needle);end=source.index('\n',start)
    source=source[:end]+"\n            two_body_counts.append(decoder.session.two_body_dispatches if two_body_available else 0)"+source[end:]
    needle='            coalesced_block_enabled=coalesced_block_enabled,coalesced_block_counts=coalesced_block_counts,'
    assert source.count(needle)==1
    return source.replace(needle,needle+'two_body_available=two_body_available,two_body_counts=two_body_counts,')

def entry(source):
    needle='info+="; coalesced_bf16_body";'
    assert source.count(needle)==1
    return source.replace(needle,needle+'info+="; shape_selected_two_body";')

def primitive(source):
    needle='{1024,1152}};';assert source.count(needle)==1
    source=source.replace(needle,'{1024,1152},{4096,1024},{896,4864},{6144,1024}};')
    condition='||'.join(f'(batch=={b}&&n=={n}&&k=={k})' for b,n,k,t in TABLE)
    needle='            need(gpu.LastCoalescedBlockDispatches()==size_t(selected),"Coalesced body route count mismatch");'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'\n            need(gpu.LastTwoBodyDispatches()==size_t('+condition+'),"Selected two body route count mismatch");')
    needle='need(gpu.LastCoalescedBlockDispatches()==0,"Abort retained coalesced body count");'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'need(gpu.LastTwoBodyDispatches()==0,"Abort retained selected two count");')
    needle='need(gpu.LastCoalescedBlockDispatches()==0,"Prefill used decode-only private layout");'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'need(gpu.LastTwoBodyDispatches()==0,"Prefill used selected two decode");')
    return source.replace('checks==80&&admitted==4','checks==104&&admitted==7')
