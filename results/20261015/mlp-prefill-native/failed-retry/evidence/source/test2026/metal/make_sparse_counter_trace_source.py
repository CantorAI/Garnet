"""Rotate one-in-31 encoder sampling and always capture vocabulary heads."""
import sys
from pathlib import Path
ns={'__file__':__file__}
wrapper=Path(__file__).with_name('make_counter_trace_source.py').read_text()
exec(compile(wrapper[:wrapper.index('destination.write_bytes')],__file__,'exec'),ns)
source=ns['source']
source=source.replace('    std::vector<std::string> traceNames;', '    std::vector<std::string> traceNames;\n    std::vector<unsigned> traceWeights;\n    std::vector<size_t> traceOrdinals;\n    size_t traceEncoderOrdinal=0;')
source=source.replace('    state.traceNames.clear();','    state.traceNames.clear();state.traceWeights.clear();state.traceOrdinals.clear();state.traceEncoderOrdinal=0;')
needle='    const size_t index=state.traceNames.size()*2;';assert source.count(needle)==1
source=source.replace(needle,'''    const size_t ordinal=state.traceEncoderOrdinal++;
    const bool head=name.rfind("bf16:",0)==0&&std::stoull(name.substr(name.rfind(':')+1))>100000;
    if(!head&&(ordinal+state.traceSequence)%31!=0)return [state.command computeCommandEncoder];
'''+needle)
source=source.replace('state.traceNames.push_back(name);return encoder;', 'state.traceNames.push_back(name);state.traceWeights.push_back(head?1u:31u);state.traceOrdinals.push_back(ordinal);return encoder;')
needle=r'\"op\":\"%s\",';assert source.count(needle)==1
source=source.replace(needle,needle+r'\"sampling_weight\":%u,\"encoder_ordinal\":%zu,')
source=source.replace('state.traceNames[i].c_str(),(unsigned long long)start','state.traceNames[i].c_str(),state.traceWeights[i],state.traceOrdinals[i],(unsigned long long)start')
needle=r'\"samples\":%zu,';assert source.count(needle)==1
source=source.replace(needle,needle+r'\"encoded_total\":%zu,\"sampling_stride\":31,')
source=source.replace('state.traceSequence,count,state.gpuMs','state.traceSequence,count,state.traceEncoderOrdinal,state.gpuMs')
Path(sys.argv[2]).write_bytes(source.encode())
print(__import__('json').dumps(dict(original_sha256=ns['hashlib'].sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),diagnostic_sha256=ns['hashlib'].sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),sampling_stride=31,heads_always_sampled=True,shader_payloads_unchanged=True)))
