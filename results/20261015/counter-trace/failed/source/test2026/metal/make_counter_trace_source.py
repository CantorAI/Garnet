"""Generate one diagnostic source; production and selected libraries stay intact."""
import hashlib,json,sys
from pathlib import Path
native=Path(sys.argv[1]);destination=Path(sys.argv[2]);source=native.read_text()
source=source.replace('#include <cstring>','#include <cstring>\n#include <atomic>\n#include <cstdio>\n#include <cstdlib>')
needle='    double gpuMs=0;';assert source.count(needle)==1
source=source.replace(needle,needle+'''
    id<MTLCounterSampleBuffer> traceCounters=nil;
    bool traceActive=false;
    uint64_t traceContext=0,traceSequence=0;
    std::vector<std::string> traceNames;
''')
needle='static size_t Elements(';assert source.count(needle)==1
source=source.replace(needle,Path(__file__).with_name('counter_trace_helpers.inc').read_text()+'\n'+needle)
assert source.count('[impl_->command computeCommandEncoder]')==12
source=source.replace('[impl_->command computeCommandEncoder]','TraceEncoder(*impl_,__func__)')
for begin,end,tag in [
    ('void MetalContext::EncodeBF16MatVec(', 'void MetalContext::Update(', 'std::string("bf16:")+std::to_string(a.rows)+":"+std::to_string(a.columns)+":"+std::to_string(o.columns)'),
    ('void MetalContext::EncodePointwise(', 'void MetalContext::EncodeRMSNorm(', 'std::string("pointwise:")+std::to_string(operation)+":"+std::to_string(a.rows)+":"+std::to_string(a.columns)'),
    ('void MetalContext::EncodeStateKernel(', 'void MetalContext::EncodeLayout(', 'std::string("state:")+std::to_string(operation)+":"+std::to_string(c.rows)+":"+std::to_string(c.columns)')]:
    start=source.index(begin);stop=source.index(end,start)
    part=source[start:stop];assert 'TraceEncoder(*impl_,__func__)' in part
    source=source[:start]+part.replace('TraceEncoder(*impl_,__func__)','TraceEncoder(*impl_,'+tag+')')+source[stop:]
needle='    impl_->partialSubmissions=0;\n    impl_->joinedPrefillPairs=0;'
assert source.count(needle)==1
source=source.replace(needle,'    impl_->partialSubmissions=0;\n    TraceBegin(*impl_);\n    impl_->joinedPrefillPairs=0;')
needle='    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}'
assert source.count(needle)==1
source=source.replace(needle,'    if(success)TraceEmit(*impl_);\n'+needle)
destination.write_bytes(source.encode())
print(json.dumps(dict(original_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),diagnostic_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),instrumented_encoder_sites=12)))
