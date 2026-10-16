from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-decode256/source')
shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source',s,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=s/'src/backends/metal/body_shape_shader.h';t=p.read_text();t=t.replace('uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n    ulong column=ulong(group)*16+simd*4;', 'uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint3 groupThreads [[threads_per_threadgroup]]) {\n    ulong column=ulong(group)*(ulong(groupThreads.x)/32*4)+simd*4;',1);p.write_text(t)
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('    size_t BodyShapePipelineCount() const;', '    static void BenchmarkDecode256(bool enabled);\n    static uint64_t BenchmarkDecode256Dispatches();\n    size_t BodyShapePipelineCount() const;');assert 'static void BenchmarkDecode256' in t;p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text().replace('#include <cstring>','#include <cstring>\n#include <atomic>');needle='bool MetalContext::PrepareBF16BodyShape';idx=t.index(needle);t=t[:idx]+'''namespace {
std::atomic<bool> benchmarkDecode256{false};std::atomic<uint64_t> benchmarkDecode256Count{0};
}
void MetalContext::BenchmarkDecode256(bool enabled) {
#ifdef GARNET_METAL_BENCHMARK_DECODE256
    benchmarkDecode256.store(enabled,std::memory_order_relaxed);
#else
    (void)enabled;throw std::logic_error("Decode256 experiment is not enabled");
#endif
}
uint64_t MetalContext::BenchmarkDecode256Dispatches(){return benchmarkDecode256Count.load(std::memory_order_relaxed);}
'''+t[idx:];t=t.replace('pipeline=found->second;++impl_->bodyShapeDispatches;', '''pipeline=found->second;++impl_->bodyShapeDispatches;
#ifdef GARNET_METAL_BENCHMARK_DECODE256
                if(benchmarkDecode256.load(std::memory_order_relaxed)&&a.rows==1&&(a.columns==2560||a.columns==9216)&&o.columns>256) {
                    threads=256;decodeColumns=32;benchmarkDecode256Count.fetch_add(1,std::memory_order_relaxed);
                }
#endif''');p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text().replace('option(GARNET_METAL_QWEN35_4B_WIDE_PREFILL', '''option(GARNET_METAL_BENCHMARK_DECODE256 "Experiment 4B body eight SIMD groups" OFF)
if(GARNET_METAL_BENCHMARK_DECODE256)
    if(NOT GARNET_METAL_BODY_SHAPE)
        message(FATAL_ERROR "Decode256 requires body shape specialization")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_BENCHMARK_DECODE256=1)
endif()
option(GARNET_METAL_QWEN35_4B_WIDE_PREFILL''',1);p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text().replace('    X::Value PrepareImage(', '''    std::string BenchmarkDecodeThreads(bool enabled) {MetalContext::BenchmarkDecode256(enabled);return "{\\"supported\\":true}";}
    int64_t BenchmarkDecodeDispatches(){return int64_t(MetalContext::BenchmarkDecode256Dispatches());}
    X::Value PrepareImage(''',1);t=t.replace('#ifdef GARNET_METAL_BODY_SHAPE', '#ifdef GARNET_METAL_BENCHMARK_DECODE256\n        info+="; benchmark_decode256";\n#endif\n#ifdef GARNET_METAL_BODY_SHAPE',1);t=t.replace('        APISET().AddFunc<0>("shutdown"', '''        APISET().AddFunc<1>("benchmark_decode_threads",&GarnetMetalAPI::BenchmarkDecodeThreads);
        APISET().AddFunc<0>("benchmark_decode_dispatches",&GarnetMetalAPI::BenchmarkDecodeDispatches);
        APISET().AddFunc<0>("shutdown"''');p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text().replace('        start = time.monotonic()','        decode256_before = garnet.benchmark_decode_dispatches()\n        start = time.monotonic()',1);t=t.replace("        stages['serving_source'] = __file__", "        stages['decode256_dispatches'] = garnet.benchmark_decode_dispatches() - decode256_before\n        stages['serving_source'] = __file__");idx=t.index('def set_benchmark_prefill_policy');t=t[:idx]+'''def set_benchmark_prefill_policy(enabled):
    """Existing generic paired-policy hook: decode grouping only; profiles fixed."""
    with _lock:
        return garnet.benchmark_decode_threads(bool(enabled))
''';p.write_text(t)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-11','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'4B body projections with inner2560/9216 and output257..99999, physical row1, use eight SIMD groups/256threads/32outputs versus four groups/128threads/16outputs. Same native shader for both modes, identical per-lane products and reduction order, BF16 original values and FP32 arithmetic. Reasoner untouched; VLM64 and native text prefix cache retained in both.','tradeoffs':'Changed occupancy/threadgroup count/input rereads; no additional weights or scratch layout. Dynamic group-width shader expression is shared by both policies; baseline parity with previous confirmed implementation required. Native actual dispatch counter overhead retained.','prior_evidence':'M4 four-threadgroup-probe rejected short isolated smaller-model settings, recommended full-model scheduling evidence. This distinct M5 4B shape-restricted WorldSense experiment is not a favorable retry of those cohorts.','validation':'Native numerical/bounds reference, exact all32VLM state bits/tokens versus previous confirmed guard and between policies; native actual positive/zero route counters; text route zero. No timing before correctness passes.','screen':'Fixed-car WorldSense two ABBA blocks, four measured samples per model/policy after both warm; identical inputs/prompts/nonpolicy budgets/output tokens.','gates':{'4b_decode_min_gain':.05,'4b_total_min_gain':.03,'4b_ttft_max_regression':.02,'text_total_ttft_decode_max_regression':.02,'exact_outputs_states_and_native_routes':True},'confirmation':'Only on screen pass: three archived frames plus query, fresh ABBA and fresh BAAB workers, each independently passes same gates; native Skill wall improves both. Preserve all failures and no unchanged favorable third worker.'},indent=2))
print('Prepared distinct native decode grouping experiment; no timing yet')
