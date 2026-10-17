from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-request-pool/source')
shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-memory/source',s,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('#include <cstdint>','#include <cstdint>\n#include <functional>');t=t.replace('    static MetalMemoryStats MemoryStats();','    static MetalMemoryStats MemoryStats();\n    static std::string RequestScope(const std::function<std::string()>&);\n    static uint64_t RequestScopeCount();');p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text().replace('#include <mach/mach.h>','#include <mach/mach.h>\n#include <atomic>\n#include <exception>');idx=t.index('MetalMemoryStats MetalContext::MemoryStats()');t=t[:idx]+'''namespace {std::atomic<uint64_t> requestScopeCount{0};}
std::string MetalContext::RequestScope(const std::function<std::string()>& callback) {
#ifdef GARNET_METAL_REQUEST_AUTORELEASE_POOL
    requestScopeCount.fetch_add(1,std::memory_order_relaxed);
    std::string result;std::exception_ptr error;
    // Catch C++ exceptions inside the lexical pool, then drain by normal exit.
    // Native owners/command buffers retain resources that must outlive this call.
    @autoreleasepool {try {result=callback();}catch(...){error=std::current_exception();}}
    if(error)std::rethrow_exception(error);
    return result;
#else
    return callback();
#endif
}
uint64_t MetalContext::RequestScopeCount(){return requestScopeCount.load(std::memory_order_relaxed);}
'''+t[idx:];p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text().replace('add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)','''add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)
option(GARNET_METAL_REQUEST_AUTORELEASE_POOL "Scope native serving load/infer autoreleases" OFF)
if(GARNET_METAL_REQUEST_AUTORELEASE_POOL)
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_REQUEST_AUTORELEASE_POOL=1)
endif()''',1);t=t.replace('enable_testing()','''enable_testing()
add_executable(garnet_request_scope_test test2026/metal/request_scope_test.mm)
target_link_libraries(garnet_request_scope_test PRIVATE garnet_metal)
add_test(NAME metal_request_scope_reference COMMAND garnet_request_scope_test)''',1);p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();t=t.replace('    std::string ServeModel(X::Value root,', '''    std::string ServeModel(X::Value root,X::Value xmodel,X::Value cache,X::Value profile,X::Value model) {
        const auto before=MetalContext::RequestScopeCount();
        auto result=MetalContext::RequestScope([&]{return ServeModelInner(root,xmodel,cache,profile,model);});
        auto parsed=nlohmann::json::parse(result);
        parsed["native_load_scope"]={{"invocations",MetalContext::RequestScopeCount()-before},
            {"memory_after_scope",nlohmann::json::parse(MemoryStatsJson())}};
        return parsed.dump();
    }
    std::string ServeModelInner(X::Value root,''',1)
t=t.replace('    std::string InferJson(X::Value prompt,', '''    std::string InferJson(X::Value prompt,X::Value image,X::Value limit,X::Value model) {
        const auto before=MetalContext::RequestScopeCount();
        const auto start=std::chrono::steady_clock::now();
        auto result=MetalContext::RequestScope([&]{return InferJsonInner(prompt,image,limit,model);});
        auto parsed=nlohmann::json::parse(result);
        parsed["native_request_scope"]={{"invocations",MetalContext::RequestScopeCount()-before},
            {"memory_after_scope",nlohmann::json::parse(MemoryStatsJson())},
            {"api_wall_seconds_before_final_dump",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
        return parsed.dump();
    }
    std::string InferJsonInner(X::Value prompt,''',1)
t=t.replace('#ifdef GARNET_METAL_DENSE_PREFIX_CACHE','#ifdef GARNET_METAL_REQUEST_AUTORELEASE_POOL\n        info+="; request_autorelease_pool";\n#endif\n#ifdef GARNET_METAL_DENSE_PREFIX_CACHE',1);p.write_text(t)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-13','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Bound native ServeModel/InferJson autorelease lifetime to each synchronous API call. Release retired resources while retained graph/session/command owners remain valid. No inference math, precision, prompts, model admission, prefill64/text16-prefix policies or CUDA changes.','tradeoffs':'Pool setup/drain CPU cost; releases occur at request/model-load boundaries. Pools do not free strongly owned cached graph buffers, so no assumed memory gain. C++ exceptions caught inside pool then rethrown after normal pool drain.','evidence':'WS12 actual resources20.409GiB vs17.760GiB recommended and native4MiB lifetime unit; actual request allocations plateau, not unbounded leak proof.','validation':'ON/OFF native allocation/alias/nesting/exception/recovery tests; real native serving entrypoint all32VLM/all28text state/token parity, capacity failure/recovery, surviving owners, correct per-API pool counters before timing.','design':'Separate fresh processes and native binaries, same source and telemetry, only pool option differs. Do not mix policies within a process, since older unpooled resources cannot be removed by a nested pool. First control/candidate car screen, four non-warm repeats each after one warm request. Only on pass run fresh full three-frame+query control/candidate and candidate/control confirmation pairs, independently passing.','gates':{'both_models_total_ttft_decode_max_regression':.02,'4b_total_min_gain':.03,'native_skill_wall_min_gain':.05,'sampled_post_request_metal_max_min_reduction':.10,'exact_full_outputs_states_inputs_prompts_configs_and_native_counters':True},'scope':'Timings include instrumentation; native API wall separately includes scope drain/snapshot/parse, model total excludes outer cleanup. Native Skill wall includes API cleanup. Memory snapshots not peaks; initialization/first and warm requests separate. No image/answer reuse, CPython image processing, camera or global-optimum claim.'},indent=2))
print('Prepared native request lifetime candidate; no speed or retention gain yet')
