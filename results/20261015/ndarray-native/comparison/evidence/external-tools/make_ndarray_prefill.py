"""Private native-context NDArray prefill; original fallback and FP32 gates."""
def header(source):
    needle='    size_t LastGlobalHalfAlphaDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastNDArrayPrefillDispatches() const;')

def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    replace('namespace Garnet {','namespace Garnet {int gNDArrayPrefillPolicy=0;')
    replace('    MPSMatrix* exactHalf;','''    std::mutex ndarrayMutex;
    MPSNDArray* ndHalfBase=nil;MPSNDArray* ndHalfTranspose=nil;
    MPSNDArray* ndFloatBase=nil;MPSNDArray* ndFloatTranspose=nil;
    id<MTLBuffer> ndFloatBuffer=nil;
    MPSMatrix* exactHalf;''')
    replace('    size_t globalHalfAlphaDispatches=0;','''    size_t globalHalfAlphaDispatches=0,ndarrayPrefillDispatches=0;
    struct NDProduct {
        MPSNDArrayMatrixMultiplication* product=nil;
        MPSNDArray* left=nil;MPSNDArray* right=nil;MPSNDArray* output=nil;MPSNDArray* zero=nil;
        id<MTLBuffer> zeroBuffer=nil;NSArray* operands=nil;
        MetalMatrix inputOwner,outputOwner;MetalBF16Weight weightOwner;
    };
    std::map<std::array<uintptr_t,9>,NDProduct> ndProducts;''')
    needle='size_t MetalContext::LastGlobalHalfAlphaDispatches() const {return impl_->globalHalfAlphaDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastNDArrayPrefillDispatches() const {return impl_->ndarrayPrefillDispatches;}')
    needle='#ifdef GARNET_METAL_EXACT_HALF_PREFILL\n        if((a.rows>2||w->visionPrefill)&&w->exactHalf&&w->halfScales) {'
    replace(needle,'''        if(a.rows>2&&!w->visionPrefill&&!w->floatPrefill&&
            ((gNDArrayPrefillPolicy==1&&w->globalHalf)||gNDArrayPrefillPolicy==2)) {
            if(@available(macOS 15.0,*)) {
                if(a.dataType!=MPSDataTypeFloat32||o.dataType!=MPSDataTypeFloat32||a.rowBytes!=a.columns*4||o.rowBytes!=o.columns*4)
                    throw std::runtime_error("NDArray prototype requires packed FP32 matrices");
                const int policy=gNDArrayPrefillPolicy;
                const std::array<uintptr_t,9> ndKey={reinterpret_cast<uintptr_t>((__bridge void*)a.data),a.offset,
                    reinterpret_cast<uintptr_t>((__bridge void*)o.data),o.offset,reinterpret_cast<uintptr_t>(w.get()),a.rows,a.columns,o.columns,uintptr_t(policy)};
                auto ndFound=impl_->ndProducts.find(ndKey);
                if(ndFound==impl_->ndProducts.end()) {
                    if(impl_->ndProducts.size()>=256)throw std::runtime_error("Private NDArray cache bound reached");
                    auto makeArray=[&](id<MTLBuffer> buffer,size_t offset,size_t rows,size_t columns,MPSDataType type){
                        auto desc=[MPSNDArrayDescriptor descriptorWithDataType:type shape:@[@1,@1,@(rows),@(columns)]];desc.preferPackedRows=YES;
                        auto value=[[MPSNDArray alloc] initWithBuffer:buffer offset:offset descriptor:desc];
                        if(!value||value.dataType!=type||value.userBuffer!=buffer)throw std::runtime_error("NDArray dtype/alias failed");return value;};
                    Impl::NDProduct item;item.inputOwner=input;item.outputOwner=output;item.weightOwner=weight;
                    item.left=makeArray(a.data,a.offset,a.rows,a.columns,MPSDataTypeFloat32);
                    item.output=makeArray(o.data,o.offset,o.rows,o.columns,MPSDataTypeFloat32);
                    {
                        std::lock_guard<std::mutex> lock(w->ndarrayMutex);
                        if(policy==1&&!w->ndHalfTranspose) {
                            auto h=w->exactHalf;
                            if(!h||h.dataType!=MPSDataTypeFloat16||h.rowBytes!=w->columns*2)throw std::runtime_error("NDArray half view unsupported");
                            w->ndHalfBase=makeArray(h.data,h.offset,w->rows,w->columns,MPSDataTypeFloat16);
                            auto desc=[w->ndHalfBase descriptor];[desc transposeDimension:0 withDimension:1];
                            w->ndHalfTranspose=[w->ndHalfBase arrayViewWithDescriptor:desc];
                        }
                        if(policy==2&&!w->ndFloatTranspose) {
                            if(w->rows>impl_->device.maxBufferLength/4/w->columns)throw std::runtime_error("NDArray float cache too large");
                            std::vector<float> values(w->rows*w->columns);
                            for(size_t i=0;i<values.size();++i){uint16_t bits;std::memcpy(&bits,static_cast<const char*>(w->buffer.contents)+i*2,2);
                                uint32_t expanded=uint32_t(bits)<<16;std::memcpy(&values[i],&expanded,4);}
                            w->ndFloatBuffer=[impl_->device newBufferWithBytes:values.data() length:values.size()*4 options:MTLResourceStorageModeShared];
                            if(!w->ndFloatBuffer)throw std::runtime_error("NDArray float cache allocation failed");
                            w->ndFloatBase=makeArray(w->ndFloatBuffer,0,w->rows,w->columns,MPSDataTypeFloat32);
                            auto desc=[w->ndFloatBase descriptor];[desc transposeDimension:0 withDimension:1];
                            w->ndFloatTranspose=[w->ndFloatBase arrayViewWithDescriptor:desc];
                        }
                        item.right=policy==1?w->ndHalfTranspose:w->ndFloatTranspose;
                    }
                    // Deferred views retain physical lengths; descriptor reports the logical transposed shape.
                    auto logical=item.right?[item.right descriptor]:nil;
                    if(!item.right||item.right.dataType!=(policy==1?MPSDataTypeFloat16:MPSDataTypeFloat32)||
                        [logical lengthOfDimension:0]!=w->rows||[logical lengthOfDimension:1]!=w->columns)
                        throw std::runtime_error("NDArray transposed view failed");
                    item.product=[[MPSNDArrayMatrixMultiplication alloc] initWithDevice:impl_->device sourceCount:policy==1?3:2];
                    if(!item.product)throw std::runtime_error("NDArray product unavailable");
                    item.product.alpha=policy==1?std::ldexp(1.0,-w->globalHalfScale):1.0;item.product.beta=0;
                    if(policy==1){item.zeroBuffer=[impl_->device newBufferWithLength:o.rows*o.columns*4 options:MTLResourceStorageModeShared];
                        if(!item.zeroBuffer)throw std::runtime_error("NDArray zero addend allocation failed");
                        std::memset(item.zeroBuffer.contents,0,o.rows*o.columns*4);item.zero=makeArray(item.zeroBuffer,0,o.rows,o.columns,MPSDataTypeFloat32);}
                    item.operands=policy==1?@[item.left,item.right,item.zero]:@[item.left,item.right];
                    ndFound=impl_->ndProducts.emplace(ndKey,std::move(item)).first;
                }
                auto& item=ndFound->second;
                [item.product encodeToCommandBuffer:impl_->command sourceArrays:item.operands destinationArray:item.output];
                output.impl_->ready=false;output.impl_->pending=true;
                if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
                ++impl_->ndarrayPrefillDispatches;return;
            }else throw std::runtime_error("NDArray buffer alias API requires macOS15");
        }
'''+needle)
    assert source.count('impl_->globalHalfAlphaDispatches=0;')==2
    return source.replace('impl_->globalHalfAlphaDispatches=0;','impl_->globalHalfAlphaDispatches=0;impl_->ndarrayPrefillDispatches=0;')

def harness(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    source=source.replace('gPackedBodyPrefillPolicy','gNDArrayPrefillPolicy').replace('LastPackedBodyPrefillDispatches','LastNDArrayPrefillDispatches')
    source=source.replace('original FP32 prefill versus packed BF16 loads','original native prefill versus mixed/F32 NDArray products')
    source=source.replace('Packed prefill actual route mismatch','NDArray prefill actual route mismatch').replace('packed_body_prefill_failed','ndarray_prefill_failed')
    replace('p.selected=p.weight.InterleavedBodyReady();','p.selected=p.weight.GlobalHalfReady();')
    needle='        auto inputs=[&](int pattern,bool reference)'
    replace(needle,'        for(auto& p:projections){p.input=gpu.Allocate(rows,p.k);p.output=gpu.Allocate(rows,p.n);}\n'+needle)
    replace('p.input=gpu.Upload(x,rows,p.k);p.output=gpu.Allocate(rows,p.n);','gpu.Update(p.input,x.data(),x.size());')
    replace('expected+=policy&&p.selected;','expected+=policy==1?p.selected:policy==2?!p.floatKey:0;')
    replace('bool finite=true;for(auto& p:projections){selected+=p.selected;',
            'bool finite=true,bitwise=true;for(auto& p:projections){selected+=policy==1?p.selected:policy==2?!p.floatKey:0;')
    replace('auto actual=gpu.Download(p.output);values+=actual.size();','auto actual=gpu.Download(p.output);bitwise=bitwise&&!std::memcmp(actual.data(),p.control.data(),actual.size()*4);values+=actual.size();')
    replace('bool passed=finite&&worst<=1&&comparison<=1;','bool passed=finite&&worst<=1&&comparison<=1&&(policy!=3||bitwise);')
    replace('need(selected>0,"No packed projection selected");','need(selected>0||policy==3,"No NDArray projection selected");')
    replace('@"finite":@(finite),@"passed":@(passed)', '@"finite":@(finite),@"bitwise_equal":@(bitwise),@"actual_ndarray_dispatches":@(gpu.LastNDArrayPrefillDispatches()),@"passed":@(passed)')
    return source
