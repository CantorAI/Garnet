"""Train-weight BF16 casts, complete MLP dependencies, unchanged stage reference."""
from pathlib import Path
import sys

def generate(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    replace('std::vector<double> activated(p.intermediate);','std::vector<float> activated(p.intermediate);')
    replace('activated[j]=(g/(1.+std::exp(-g)))*u;',
            'float gate=static_cast<float>(g),up=static_cast<float>(u);activated[j]=(gate/(1.f+std::exp(-gate)))*up;')
    replace('p.reference[r*p.width+c]=sum;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(sum);',
            'float rounded=static_cast<float>(sum);p.reference[r*p.width+c]=rounded;p.bounds[r*p.width+c]=5e-6+5e-5*std::abs(double(rounded));')
    replace('std::vector<float> values;', 'std::vector<float> values;std::vector<uint16_t> originalBits;')
    replace('weight.graph=[device newBufferWithBytes:weight.values.data() length:weight.values.size()*4 options:MTLResourceStorageModeShared];need(weight.graph!=nil,"Weight buffer missing");',
            '''weight.originalBits=bits;
        weight.graph=[device newBufferWithBytes:bits.data() length:bits.size()*2 options:MTLResourceStorageModeShared];need(weight.graph!=nil,"Weight buffer missing");
        @autoreleasepool {
            auto castGraph=[MPSGraph new];auto input=[castGraph placeholderWithShape:@[@(weight.n),@(weight.k)] dataType:MPSDataTypeBFloat16 name:nil];
            auto result=[castGraph castTensor:input toType:MPSDataTypeFloat32 name:nil];
            need(input.dataType==MPSDataTypeBFloat16&&result.dataType==MPSDataTypeFloat32,"Cast dtype changed");
            auto actual=[device newBufferWithLength:weight.values.size()*4 options:MTLResourceStorageModeShared];need(actual!=nil,"Cast output missing");
            auto feed=[[MPSGraphTensorData alloc] initWithMTLBuffer:weight.graph shape:@[@(weight.n),@(weight.k)] dataType:MPSDataTypeBFloat16];
            auto target=[[MPSGraphTensorData alloc] initWithMTLBuffer:actual shape:@[@(weight.n),@(weight.k)] dataType:MPSDataTypeFloat32];
            [castGraph runWithMTLCommandQueue:queue feeds:@{input:feed} targetOperations:nil resultsDictionary:@{result:target}];
            need(!std::memcmp(actual.contents,weight.values.data(),weight.values.size()*4),"GPU BF16 cast changed original learned bits");
            emit(@{@"kind":@"weight_cast",@"tensor":item[@"tensor"],@"values":@(weight.values.size()),@"bitwise_equal":@YES});
        }''')
    replace('''                    if(policy==0){weights[i]=[graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];p.feeds[policy][weights[i]]=[[MPSGraphTensorData alloc] initWithMTLBuffer:w.graph shape:shape dataType:MPSDataTypeFloat32];}
                    else weights[i]=[graph constantWithData:[NSData dataWithBytes:w.values.data() length:w.values.size()*4] shape:shape dataType:MPSDataTypeFloat32];''',
            '''                    MPSGraphTensor* stored;
                    if(policy==0){stored=[graph placeholderWithShape:shape dataType:MPSDataTypeBFloat16 name:nil];p.feeds[policy][stored]=[[MPSGraphTensorData alloc] initWithMTLBuffer:w.graph shape:shape dataType:MPSDataTypeBFloat16];}
                    else stored=[graph constantWithData:[NSData dataWithBytes:w.originalBits.data() length:w.originalBits.size()*2] shape:shape dataType:MPSDataTypeBFloat16];
                    need(stored.dataType==MPSDataTypeBFloat16,"Stored weight dtype changed");
                    weights[i]=[graph castTensor:stored toType:MPSDataTypeFloat32 name:nil];''')
    assert 'need(weights[i].dataType==MPSDataTypeFloat32' in source
    return source

if __name__=='__main__':Path(sys.argv[1]).write_bytes(generate(Path(__file__).with_name('dependent_mlp_graph_gpu.mm').read_text()).encode())
