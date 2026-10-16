from pathlib import Path
import shutil,json,datetime
r=Path('/Users/shawn/CantorAI/Garnet');w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-chunk64/source');s.mkdir(parents=True,exist_ok=False)
for rel in ['src','cmake','runtime','xModel','third_party/nlohmann']:
 shutil.copytree(r/rel,s/rel,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(r/'CMakeLists.txt',s/'CMakeLists.txt');(s/'test2026/metal').mkdir(parents=True)
for p in (r/'test2026/metal').iterdir():
 if p.is_file()and p.suffix in ['.cpp','.mm','.h','.py']:shutil.copy2(p,s/'test2026/metal'/p.name)
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('inline constexpr size_t kMetalPrefillPackedHiddenLimit=65536;','''#ifdef GARNET_METAL_QWEN35_4B_WIDE_PREFILL
inline constexpr size_t kMetalPrefillPackedHiddenLimit=64*2560;
#else
inline constexpr size_t kMetalPrefillPackedHiddenLimit=65536;
#endif''');p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text().replace('add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)','''add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)
option(GARNET_METAL_QWEN35_4B_WIDE_PREFILL "Validate 4B 64-token packed prefill" OFF)
if(GARNET_METAL_QWEN35_4B_WIDE_PREFILL)
    if(GARNET_METAL_QWEN35_EXTENDED_PREFILL)
        message(FATAL_ERROR "4B wide prefill and small-model extended prefill are separate experiments")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_QWEN35_4B_WIDE_PREFILL=1)
endif()''');p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();idx=t.index('#ifdef GARNET_METAL_BODY_SHAPE');t=t[:idx]+'''#ifdef GARNET_METAL_QWEN35_4B_WIDE_PREFILL
        info+="; qwen35_4b_wide_prefill";
#endif
'''+t[idx:]
# Reuse generic paired-policy bridge; production source has no benchmark setter.
old=(Path('/Users/shawn/CantorAI/out/research/worldsense-small-prefill/source/src/entry/metal_main.cpp')).read_text()
start=old.index('    X::Value SetBenchmarkPrefillPolicy(') if '    X::Value SetBenchmarkPrefillPolicy(' in old else -1
# Locate the exact generic function and registration by surrounding lines below.
p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text().replace("if self.visual and self.prefill_chunk * self.text_config['hidden_size'] > 65536:","packed_limit = 64 * 2560 if self.visual and self.xmodel == 'vl_4b' and 'qwen35_4b_wide_prefill' in garnet.backend_info() else 65536\n        if self.visual and self.prefill_chunk * self.text_config['hidden_size'] > packed_limit:")
t += '''\n\ndef set_benchmark_prefill_policy(enabled):
    """Paired 4B chunk ablation; graph keys already include physical chunk width."""
    with _lock:
        if 'qwen35_4b_wide_prefill' not in garnet.backend_info() or not _models:
            return json.dumps(dict(supported=False))
        changed = 0
        for model in _models.values():
            if model.visual and model.xmodel == 'vl_4b':
                if model.text_config['hidden_size'] != 2560 or not model.decoder.linear_chunk:
                    raise ValueError('Wide 4B prefill requires validated 4B parallel linear decoder')
                # Reuse chunk-keyed graphs and shared weights. Preserve both policies'
                # warm graphs; enable_prefill would otherwise close the old cache.
                model.decoder.chunk = 64 if enabled else 16
                model.prefill_chunk = model.decoder.chunk
                changed += 1
        return json.dumps(dict(supported=changed > 0, enabled=bool(enabled)))
''';p.write_text(t)
for name in ['controller.py','cases.json']:
 t=(w.parent/'worldsense-small-prefill'/name).read_text().replace('worldsense-small-prefill','worldsense-chunk64');(w/name).write_text(t)
shutil.copy2(w.parent/'worldsense-small-prefill/before-libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-07','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'4B physical prefill64 versus16; raise opt-in packed-width admission to64*2560 with native chunk limit64 unchanged. Same original BF16/FP32 kernels, capacity, pixels, prompt and decoder. Text reasoner stays16.','tradeoffs':'Larger activation/graph allocations, potential padding work and longer GPU batches; fewer repeated body-weight reads. Both chunk-keyed graph sets retained warm for same-process ablation, with shared weights. No answer/image cache. Logical extra allocations and RSS do not equal peak GPU memory.','stage':'Correctness native checks and32-layer state/token equivalence before single-car WorldSense ABBA screen, four samples per policy/model. Fresh full three-frame/search ABBA/BAAB confirmation required for promotion.','gates':{'exact_tokens_and_32_state_bits':True,'actual_chunk_plan':True,'4b_ttft_min_gain':.10,'4b_total_min_gain':.03,'decode_max_regression':.02,'text_total_ttft_max_regression':.02}},indent=2))
print('WS-M5-07 prepared, no measurements')
