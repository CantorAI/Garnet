"""Private graph encoding into the native MPSCommandBuffer; no serving edit.

The SDK permits graph encoding to commitAndContinue internally. Refresh the
underlying command buffer, retain the prior root, check graph completion at
final submission, and drain submitted GPU work before abort releases owners.
"""
import re

def context(source):
    needle='    struct MlpGraph {'
    assert source.count(needle)==1
    source=source.replace(needle,'''    struct MlpCompletion {
        std::mutex mutex;std::condition_variable condition;
        bool complete=false;std::string error;
    };
    std::vector<std::shared_ptr<MlpCompletion>> mlpCompletions;
'''+needle)
    needle='    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}'
    assert source.count(needle)==1
    source=source.replace(needle,'''#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
    for(auto& completion:impl_->mlpCompletions) {
        std::unique_lock<std::mutex> lock(completion->mutex);
        if(!completion->condition.wait_for(lock,std::chrono::seconds(5),[&](){return completion->complete;})) {
            success=false;failure=@"Command-buffer graph completion callback missing";
        } else if(!completion->error.empty()) {
            success=false;failure=[NSString stringWithUTF8String:completion->error.c_str()];
        }
    }
    impl_->mlpCompletions.clear();
#endif
'''+needle)
    needle='void MetalContext::Abort() {'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'''
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
    // An MPS graph may have committed internal buffers. The final continuation
    // is discarded; a queue barrier drains every already submitted producer.
    if(!impl_->mlpCompletions.empty()) {
        @autoreleasepool {
            auto barrier=[impl_->queue commandBuffer];
            if(barrier){[barrier commit];[barrier waitUntilCompleted];}
        }
        impl_->mlpCompletions.clear();
    }
#endif
''')
    return source

def graph(source):
    start=source.index('    struct Completion {')
    end=source.index('    ++impl_->mlpGraphDispatches;return true;',start)
    replacement='''    auto completion=std::make_shared<Impl::MlpCompletion>();
    auto descriptor=[MPSGraphExecutionDescriptor new];descriptor.waitUntilCompleted=NO;
    descriptor.completionHandler=^(MPSGraphTensorDataDictionary*,NSError* error){
        std::lock_guard<std::mutex> lock(completion->mutex);
        if(error){const char* message=error.localizedDescription.UTF8String;completion->error=message?message:"MLP command-buffer graph failed";}
        completion->complete=true;completion->condition.notify_all();
    };
    output.impl_->ready=false;output.impl_->pending=true;
    if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    auto buffer=[MPSCommandBuffer commandBufferWithCommandBuffer:impl_->command];
    if(!buffer)throw std::runtime_error("Cannot wrap native command buffer for MLP graph");
    auto original=buffer.rootCommandBuffer;
    impl_->mlpCompletions.push_back(completion);
    @try {
        auto& item=found->second;
        [item.graph encodeToCommandBuffer:buffer feeds:item.feeds targetOperations:nil resultsDictionary:item.results executionDescriptor:descriptor];
    } @catch(NSException* exception) {
        impl_->command=buffer.rootCommandBuffer;
        auto barrier=[impl_->queue commandBuffer];
        if(barrier){[barrier commit];[barrier waitUntilCompleted];}
        const char* message=exception.reason.UTF8String;throw std::runtime_error(message?message:"MLP command-buffer graph exception");
    }
    // Graph encoding is allowed to commitAndContinue. Every subsequent native
    // encoder must use its current root, never the possibly committed original.
    impl_->command=buffer.rootCommandBuffer;
    if(original!=impl_->command) {
        if(original.status<MTLCommandBufferStatusCommitted)
            throw std::runtime_error("MPS graph replaced an uncommitted root");
        if(!impl_->submitted)impl_->submitted=[NSMutableArray new];
        [impl_->submitted addObject:original];
    }
'''
    result=source[:start]+replacement+source[end:]
    assert 'runAsyncWithMTLCommandQueue:' not in result and '    Flush();' not in result
    return result

def primitive(source,*,chained=False):
    # Keep the original 64 cases and bounds; add a native consumer after each
    # graph so success also proves output ordering within the continuation.
    needle='control=gpu.Allocate(count,width),output=gpu.Allocate(count,width);'
    assert source.count(needle)==1
    source=source.replace(needle,'control=gpu.Allocate(count,width),output=gpu.Allocate(count,width),consumer=gpu.Allocate(count,width),consumerControl=gpu.Allocate(count,width);')
    needle='gpu.EncodeBF16MatVec(activated,downWeight,control);gpu.SubmitAndWait();auto expected=gpu.Download(control);'
    assert source.count(needle)==1
    source=source.replace(needle,'gpu.EncodeBF16MatVec(activated,downWeight,control);gpu.EncodeAdd(control,input,consumerControl);gpu.SubmitAndWait();auto expected=gpu.Download(control);auto expectedConsumer=gpu.Download(consumerControl);')
    needle='need(gpu.LastMlpGraphDispatches()==1&&gpu.LastPartialSubmissions()==1,"Actual graph/producer flush counts wrong");'
    assert source.count(needle)==1
    source=source.replace(needle,'need(gpu.LastMlpGraphDispatches()==1&&gpu.LastPartialSubmissions()==0,"Command-buffer graph/native-flush counts wrong");gpu.EncodeAdd(output,input,consumer);')
    needle='compare(gpu.Download(output),expected);'
    assert source.count(needle)==2
    source=source.replace(needle,needle+'compare(gpu.Download(consumer),expectedConsumer);')
    if chained:
        needle='            auto uninitialized=gpu.Allocate(count,width);'
        assert source.count(needle)==1
        source=source.replace(needle,'''            {
                auto chainInput=gpu.Allocate(count,width),chainOutput=gpu.Allocate(count,width);
                auto chainControl=gpu.Allocate(count,width),chainConsumer=gpu.Allocate(count,width),chainConsumerControl=gpu.Allocate(count,width);
                gpu.Begin();gpu.EncodeBF16MatVec(consumerControl,gateWeight,gate);gpu.EncodeBF16MatVec(consumerControl,upWeight,up);
                gpu.EncodeSiluMul(gate,up,activated);gpu.EncodeBF16MatVec(activated,downWeight,chainControl);
                gpu.EncodeAdd(chainControl,input,chainConsumerControl);gpu.SubmitAndWait();
                auto expectedChain=gpu.Download(chainControl),expectedChainConsumer=gpu.Download(chainConsumerControl);
                auto runChain=[&](){
                    gpu.Begin();gpu.EncodeAdd(input,input,prepared);
                    need(gpu.EncodeMlpPrefillGraph(prepared,gateWeight,upWeight,downWeight,output),"First dependent graph fell back");
                    gpu.EncodeAdd(output,input,chainInput);
                    need(gpu.EncodeMlpPrefillGraph(chainInput,gateWeight,upWeight,downWeight,chainOutput),"Second dependent graph fell back");
                    gpu.EncodeAdd(chainOutput,input,chainConsumer);
                    need(gpu.LastMlpGraphDispatches()==2&&gpu.LastPartialSubmissions()==0,"Dependent graph/flush count mismatch");
                };
                runChain();rejected=false;try{gpu.Download(chainConsumer);}catch(const std::logic_error&){rejected=true;}
                need(rejected,"Pending dependent graph consumer readable");gpu.SubmitAndWait();
                compare(gpu.Download(chainOutput),expectedChain);compare(gpu.Download(chainConsumer),expectedChainConsumer);
                runChain();gpu.Abort();need(gpu.LastMlpGraphDispatches()==0,"Dependent graph abort count survived");
                rejected=false;try{gpu.Download(chainOutput);}catch(const std::logic_error&){rejected=true;}
                need(rejected,"Aborted dependent graph output readable");
                runChain();gpu.SubmitAndWait();compare(gpu.Download(chainOutput),expectedChain);compare(gpu.Download(chainConsumer),expectedChainConsumer);
            }
'''+needle)
        needle='std::printf("MLP graph pending producer, ownership, abort/recovery and native bound PASS: %zu cases\\n",checked);'
        assert source.count(needle)==1
        source=source.replace(needle,'std::printf("MLP command-buffer dependent two-graph chains PASS: %zu cases\\n",checked);'+needle)
    return source.replace('MLP graph pending producer, ownership, abort/recovery and native bound PASS:',
        'MLP command-buffer graph producer/consumer, ownership, abort/recovery and original native bound PASS:')
