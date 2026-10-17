from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;seed=Path('/Users/shawn/CantorAI/out/research/worldsense-lazy-defaults/source');source=Path('/Users/shawn/CantorAI/out/research/worldsense-storage-scopes/source');assert not source.exists();shutil.copytree(seed,source,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
protocol=dict(experiment='WS-M5-16',declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='Retain WS15 actual lazy retirement, replace frequent getter pools with graph BuildInternal/ordinary Run/resident Run lifetime pools. Hold native owners through completed GPU work; catch exceptions inside lexical pool then rethrow after drain. No attribution of WS15 text regression or assumed speed gain.',gates=dict(sampled_metal_allocation_min_reduction=.10,both_models_ttft_total_decode_max_regression=.02,native_skill_wall_max_regression=.02,exact_outputs_states_inputs_prompts_budgets_counters=True),validation='Native owner/alias/exception/nested scope tests, OFF/ON references, real32VLM/28text states/tokens/prefix/overflow recovery, mixed generic/resident/late second session/closed rebuilt/malformed controls, separate partial allocation fault recovery; then fresh WorldSense separate car controls, two independently passing fresh full three-frame/query C/A and A/C confirmation pairs.',tradeoffs='One pool per native graph/session boundary, std::function/capture/exception bridge cost belongs to candidate; same full context, original math/model/images/CUDA. Allocation snapshots not peaks/free RAM; fixed-frame timings not live decode/persistence/retrieval. Preserve failed gates, first calls and all artifacts. No unchanged favorable retry.')
(w/'protocol.json').write_text(json.dumps(protocol,indent=2))
p=source/'src/backends/metal/metal_context.mm';s=p.read_text();s=s.replace('#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n    @autoreleasepool {\n#endif\n','').replace('\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n    }\n#endif\n','\n');assert 'GARNET_METAL_LAZY_RESIDENT_DEFAULTS'not in s
s+='''\nnamespace Garnet {
void MetalContext::StorageScope(const std::function<void()>& callback) {
    std::exception_ptr error;
    @autoreleasepool {try {callback();}catch(...){error=std::current_exception();}}
    if(error)std::rethrow_exception(error);
}
}\n''';s=s.replace('#include <mutex>','#include <mutex>\n#include <exception>');p.write_text(s)
p=source/'src/backends/metal/metal_context.h';s=p.read_text().replace('#include <cstdint>','#include <cstdint>\n#include <functional>');s=s.replace('class MetalContext {','class MetalContext {');idx=s.index('public:',s.index('class MetalContext {'));s=s[:idx]+s[idx:].replace('public:', 'public:\n    static void StorageScope(const std::function<void()>&);',1);p.write_text(s)
p=source/'src/backends/metal/compiled_graph.h';s=p.read_text();needle='    bool BuildInternal(X::Value,X::Value,const std::string&,std::unordered_map<std::string,SharedWeight>,bool bf16Projections=false);';assert needle in s;s=s.replace(needle,needle+'\n    bool BuildInternalUnscoped(X::Value,X::Value,const std::string&,std::unordered_map<std::string,SharedWeight>,bool);\n    X::Value RunUnscoped(X::Value);');p.write_text(s)
p=source/'src/backends/metal/compiled_graph.cpp';s=p.read_text().replace('bool MetalCompiledGraph::BuildInternal(X::Value graph,','bool MetalCompiledGraph::BuildInternalUnscoped(X::Value graph,',1).replace('X::Value MetalCompiledGraph::Run(X::Value inputs) {','X::Value MetalCompiledGraph::RunUnscoped(X::Value inputs) {',1)
s+='''\nnamespace Garnet {
bool MetalCompiledGraph::BuildInternal(X::Value graph,X::Value symbols,const std::string& checkpoint,std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections) {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    bool result=false;MetalContext::StorageScope([&]{result=BuildInternalUnscoped(std::move(graph),std::move(symbols),checkpoint,std::move(shared),bf16Projections);});return result;
#else
    return BuildInternalUnscoped(std::move(graph),std::move(symbols),checkpoint,std::move(shared),bf16Projections);
#endif
}
X::Value MetalCompiledGraph::Run(X::Value inputs) {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    X::Value result;MetalContext::StorageScope([&]{result=RunUnscoped(std::move(inputs));});return result;
#else
    return RunUnscoped(std::move(inputs));
#endif
}
}\n''';p.write_text(s)
p=source/'src/backends/metal/decoder_session.h';s=p.read_text();needle='    bool poisoned_=false;';assert needle in s;s=s.replace(needle,'    X::Value RunUnscoped(X::Value,X::Value,X::Value,X::Value,X::Value,int64_t);\n'+needle);p.write_text(s)
p=source/'src/backends/metal/decoder_session.cpp';s=p.read_text().replace('X::Value MetalDecoderSession::Run(X::Value graphs,','X::Value MetalDecoderSession::RunUnscoped(X::Value graphs,',1);s+='''\nnamespace Garnet {
X::Value MetalDecoderSession::Run(X::Value graphs,X::Value hidden,X::Value positions,X::Value slots,X::Value active,int64_t predictionToken) {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    X::Value result;MetalContext::StorageScope([&]{result=RunUnscoped(std::move(graphs),std::move(hidden),std::move(positions),std::move(slots),std::move(active),predictionToken);});return result;
#else
    return RunUnscoped(std::move(graphs),std::move(hidden),std::move(positions),std::move(slots),std::move(active),predictionToken);
#endif
}
}\n''';p.write_text(s)
p=source/'src/entry/metal_main.cpp';s=p.read_text().replace('info+="; lazy_resident_defaults";','info+="; lazy_resident_defaults; storage_lifetime_scopes";');p.write_text(s)
shutil.copy2(w.parent/'worldsense-lazy-defaults/before-libgarnet.dylib',w/'before-libgarnet.dylib')
print('WS16 source prepared; no performance samples',flush=True)
