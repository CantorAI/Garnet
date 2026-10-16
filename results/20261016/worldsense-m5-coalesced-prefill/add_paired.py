from pathlib import Path
s=Path('/Users/shawn/CantorAI/out/research/worldsense-coalesced-prefill/source')
p=s/'src/backends/metal/decoder_session.h';x=p.read_text().replace('    double LastInferenceMs() const','    bool SetCoalescedPrefill(bool enabled);\n    double LastInferenceMs() const').replace('        APISET().AddFunc<0>("reset",','        APISET().AddFunc<1>("set_coalesced_prefill",&MetalDecoderSession::SetCoalescedPrefill);\n        APISET().AddFunc<0>("reset",');p.write_text(x)
p=s/'src/backends/metal/decoder_session.cpp';x=p.read_text().replace('void MetalDecoderSession::Reset() {','bool MetalDecoderSession::SetCoalescedPrefill(bool enabled) {std::lock_guard<std::mutex> lock(mutex_);if(!gpu_)throw std::logic_error("Metal session is closed");return gpu_->SetCoalescedPrefill(enabled);}\nvoid MetalDecoderSession::Reset() {');p.write_text(x)
p=s/'src/entry/metal_main.cpp';x=p.read_text().replace('    std::string ServeModel(','''    std::string BenchmarkPrefillPolicy(X::Value enabled) {
        auto module=Serving(enabled.host());
        return module["set_benchmark_prefill_policy"](enabled).ToString();
    }
    std::string ServeModel(''');x=x.replace('        APISET().AddFunc<4>("infer_json",','        APISET().AddFunc<1>("set_benchmark_prefill_policy",&GarnetMetalAPI::BenchmarkPrefillPolicy);\n        APISET().AddFunc<4>("infer_json",');p.write_text(x)
p=s/'runtime/metal_serving.py';x=p.read_text();x+='''\n\ndef set_benchmark_prefill_policy(enabled):
    """Synchronous benchmark ablation; no model state or weights are changed."""
    with _lock:
        if 'coalesced_prefill' not in garnet.backend_info() or not _models:
            return json.dumps(dict(supported=False))
        for model in _models.values():
            if not model.decoder.session.set_coalesced_prefill(bool(enabled)):
                raise ValueError('Native prefill policy is unavailable')
        return json.dumps(dict(supported=True, enabled=bool(enabled)))
''';p.write_text(x)
p=Path('/Users/shawn/CantorAI/Galaxy/garnetfilter/GarnetVLMFilter.cpp');x=p.read_text();idx=x.index('X::Value GarnetVLMFilter::BenchmarkCurrentFrame(')
helper='''// Benchmark-only ablation. The live default is restored on every return or
// exception; callers still execute the production Skill/search implementation.
class ScopedBenchmarkPrefillPolicy {
    X::Value garnet_;
    bool active_ = false;
public:
    ScopedBenchmarkPrefillPolicy(X::Value garnet, X::Value json, const X::Value& request) : garnet_(garnet) {
        if (!request.IsDict() || !Has(request,"benchmark") || !request.Get("benchmark").ToLongLong() ||
            !Has(request,"benchmark_prefill_policy")) return;
        const bool enabled=request.Get("benchmark_prefill_policy").ToLongLong()!=0;
        auto response=CallValue(garnet_["set_benchmark_prefill_policy"],{X::Value(enabled)});
        auto status=json["loads"](response.ToString());
        if (!status.IsDict() || !Has(status,"supported") || !status.Get("supported").ToLongLong())
            throw X::Error("This Garnet build does not support paired prefill ablation");
        active_=true;
    }
    ~ScopedBenchmarkPrefillPolicy() {
        if(active_)try {CallValue(garnet_["set_benchmark_prefill_policy"],{X::Value(true)});}
        catch(const std::exception& error) {std::cerr<<"Benchmark prefill restore failed: "<<error.what()<<'\\n';}
    }
};

'''
x=x[:idx]+helper+x[idx:];start=x.index('X::Value GarnetVLMFilter::BenchmarkCurrentFrame(');end=x.index('X::Value GarnetVLMFilter::PlanSearch(',start);part=x[start:end].replace('    m_framePerceptions.clear();\n    Request request','    ScopedBenchmarkPrefillPolicy policy(m_garnet,m_json,candidate);\n    m_framePerceptions.clear();\n    Request request');x=x[:start]+part+x[end:];start=x.index('X::Value GarnetVLMFilter::PlanSearch(');end=x.index('X::Value GarnetVLMFilter::RankSearch(',start);part=x[start:end].replace('    std::string kind;','    ScopedBenchmarkPrefillPolicy policy(m_garnet,m_json,imageSource);\n    std::string kind;');x=x[:start]+part+x[end:];p.write_text(x)
