"""Original BF16 storage feeds, exact FP32 casts, unchanged FP32 MLP arithmetic."""
from make_mlp_command_buffer_context import context,primitive,graph as command_graph

def graph(source):
    source=command_graph(source)
    needle='if(@available(macOS 13.0,*))';assert source.count(needle)==1
    source=source.replace(needle,'if(@available(macOS 14.0,*))')
    start=source.index('        for(auto w:{g,u,d}) {')
    end=source.index('        Impl::MlpGraph item;',start)
    # Buffer ownership remains retained by item.gate/up/downOwner. The original
    # tightly packed immutable BF16 buffer is authoritative, with no FP32 cache.
    source=source[:start]+'''        for(auto w:{g,u,d})
            if(!w->buffer||w->rows>impl_->device.maxBufferLength/2/w->columns||
                w->buffer.length<w->rows*w->columns*2)return false;
'''+source[end:]
    old='''            auto tensor=[graph placeholderWithShape:@[@(w->rows),@(w->columns)] dataType:MPSDataTypeFloat32 name:nil];
            item.feeds[tensor]=[[MPSGraphTensorData alloc] initWithMTLBuffer:w->graphFloat shape:@[@(w->rows),@(w->columns)] dataType:MPSDataTypeFloat32];
            weights[index++]=tensor;'''
    new='''            auto tensor=[graph placeholderWithShape:@[@(w->rows),@(w->columns)] dataType:MPSDataTypeBFloat16 name:nil];
            if(tensor.dataType!=MPSDataTypeBFloat16)throw std::runtime_error("Original BF16 feed dtype changed");
            item.feeds[tensor]=[[MPSGraphTensorData alloc] initWithMTLBuffer:w->buffer shape:@[@(w->rows),@(w->columns)] dataType:MPSDataTypeBFloat16];
            weights[index++]=[graph castTensor:tensor toType:MPSDataTypeFloat32 name:nil];'''
    assert source.count(old)==1
    source=source.replace(old,new)
    assert 'graphFloat' not in source
    assert 'if(tensor.dataType!=MPSDataTypeFloat32)' in source
    assert 'descriptor.waitUntilCompleted=NO' in source and '    Flush();' not in source
    return source
