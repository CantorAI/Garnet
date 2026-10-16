"""Isolate unit-alpha versus explicit-addend alpha behavior; retain original gates."""
from pathlib import Path
import sys
def generate(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    replace('need(argc==2,"Expected mixed or f32 mode");','need(argc==3,"Expected storage and alpha path");')
    needle='need(mode=="mixed"||mode=="f32","Unknown mode");'
    replace(needle,needle+'\n    std::string variant=argv[2];need(variant=="unit"||variant=="three","Unknown alpha path");int scale=variant=="unit"?0:8;NSUInteger sources=variant=="three"?3:2;')
    replace('raw[c*k+j],8,half[c*k+j]','raw[c*k+j],scale,half[c*k+j]')
    replace('std::ldexp(expand(raw[c*k+j]),8)','std::ldexp(expand(raw[c*k+j]),scale)')
    replace('std::ldexp(float(restored),-8)','std::ldexp(float(restored),-scale)')
    assert source.count('std::ldexp(1.,-8)')==2
    source=source.replace('std::ldexp(1.,-8)','std::ldexp(1.,-scale)')
    replace('initWithDevice:device sourceCount:2','initWithDevice:device sourceCount:sources')
    replace('@"scale":@8,','@"scale":@(scale),@"variant":[NSString stringWithUTF8String:variant.c_str()],@"source_count":@(sources),@"alpha_requested":@(std::ldexp(1.,-scale)),@"alpha_reported":@(product.alpha),@"beta_reported":@(product.beta),')
    needle='auto lhs=array(left,m,k,MPSDataTypeFloat32),result=array(ndOut,m,n,MPSDataTypeFloat32);'
    replace(needle,needle+'''
            std::vector<float> zeros(m*n,0.f);auto addendStorage=buffer(device,zeros.data(),zeros.size()*4);
            auto addend=array(addendStorage,m,n,MPSDataTypeFloat32);NSArray* operands=sources==3?@[lhs,rhs,addend]:@[lhs,rhs];''')
    replace('sourceArrays:@[lhs,rhs] destinationArray:result','sourceArrays:operands destinationArray:result')
    needle='need(!std::memcmp(left.contents,a.data(),a.size()*4)&&!std::memcmp(original.contents,half.data(),half.size()*2),"Immutable input changed");'
    replace(needle,needle+'\n            need(!std::memcmp(addendStorage.contents,zeros.data(),zeros.size()*4),"Zero addend changed");')
    return source
if __name__=='__main__':Path(sys.argv[1]).write_bytes(generate(Path(__file__).with_name('ndarray_compat_gpu.mm').read_text()).encode())
