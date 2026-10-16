"""Test deferred transpose semantics separately, retaining all original numeric gates."""
from pathlib import Path
from make_ndarray_alpha_probe import generate as alpha_generate
import sys
def generate(source):
    source=alpha_generate(source)
    old='auto rhs=array(right,k,n,mode=="mixed"?MPSDataTypeFloat16:MPSDataTypeFloat32);'
    assert source.count(old)==1
    source=source.replace(old,'''std::vector<float> rowWide(n*k);for(size_t i=0;i<n*k;++i)rowWide[i]=std::ldexp(expand(raw[i]),scale);
        auto viewStorage=mode=="mixed"?original:buffer(device,rowWide.data(),rowWide.size()*4);
        auto viewBase=array(viewStorage,n,k,mode=="mixed"?MPSDataTypeFloat16:MPSDataTypeFloat32);
        MPSNDArray* rhs=nil;
        for(bool packed:{true,false}){
            auto desc=[viewBase descriptor];desc.preferPackedRows=packed;[desc transposeDimension:0 withDimension:1];
            auto v=[viewBase arrayViewWithDescriptor:desc];auto logical=v?[v descriptor]:nil;
            emit(@{@"kind":@"view",@"shape":@(shape),@"packed_requested":@(packed),@"available":@(v!=nil),
                @"dtype":@(v?v.dataType:0),@"user_buffer_alias":@(v&&v.userBuffer==viewStorage),
                @"physical_dim0":@(v?[v lengthOfDimension:0]:0),@"physical_dim1":@(v?[v lengthOfDimension:1]:0),
                @"logical_shape":logical?[logical getShape]:@[]});
            if(!packed)rhs=v;
        }
        need(rhs!=nil,"Deferred transpose view unavailable");auto logical=[rhs descriptor];
        need([logical lengthOfDimension:0]==n&&[logical lengthOfDimension:1]==k,"Deferred logical shape mismatch");''')
    old='need(!std::memcmp(right.contents,mode=="mixed"?static_cast<const void*>(transposed.data()):static_cast<const void*>(wide.data()),mode=="mixed"?transposed.size()*2:wide.size()*4),"NDArray weights changed");'
    assert source.count(old)==1
    source=source.replace(old,old+'\n            need(!std::memcmp(viewStorage.contents,mode=="mixed"?static_cast<const void*>(half.data()):static_cast<const void*>(rowWide.data()),mode=="mixed"?half.size()*2:rowWide.size()*4),"View storage changed");')
    return source
if __name__=='__main__':Path(sys.argv[1]).write_bytes(generate(Path(__file__).with_name('ndarray_compat_gpu.mm').read_text()).encode())
