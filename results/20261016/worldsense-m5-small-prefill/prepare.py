from pathlib import Path
import shutil,re,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-small-prefill/source');old=Path('/Users/shawn/CantorAI/out/research/worldsense-coalesced-nopad/source');shutil.copytree(old,s)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text()
# Remove all prototype shader/pipeline blocks; reuse established small kernel.
for needle in ['#include "coalesced_prefill_shader.h"','id<MTLComputePipelineState> coalescedPrefill16','source=[source stringByAppendingString:[NSString stringWithUTF8String:kCoalescedPrefillShader]]','impl_->kernels->coalescedPrefill16=','if(impl_->coalescedPrefillEnabled)widePipeline=']:
 blocks=list(re.finditer(r'#ifdef GARNET_METAL_COALESCED_PREFILL\n.*?#endif',t,re.S));found=[b for b in blocks if needle in b.group()];assert len(found)==1,needle
 b=found[0];t=t[:b.start()]+t[b.end():]
t=t.replace('if(wide&&impl_->coalescedPrefillEnabled)++impl_->coalescedPrefillDispatches;','if(wide&&a.rows==16&&impl_->coalescedPrefillEnabled){wide=false;++impl_->coalescedPrefillDispatches;}')
p.write_text(t)
(s/'src/backends/metal/coalesced_prefill_shader.h').unlink()
for p in s.rglob('*'):
 if p.is_file()and p.suffix in ['.cpp','.mm','.h','.py','.cmake','.txt']:
  try:t=p.read_text()
  except UnicodeDecodeError:continue
  t=t.replace('COALESCED_PREFILL','SMALL_PREFILL').replace('CoalescedPrefill','SmallPrefill').replace('coalescedPrefill','smallPrefill').replace('coalesced_prefill_nopad','small_prefill').replace('coalesced_prefill','small_prefill');p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text().replace('Coalesce original BF16 wide-tile loads with padded shared transpose','Use existing smaller BF16 prefill tile for physical rows 16');p.write_text(t)
p=s/'test2026/metal/coalesced_prefill_test.cpp';t=p.read_text().replace('size_t(b>=16&&n>=64)','size_t(b==16&&n>=64)').replace('Coalesced','Small-tile');p.write_text(t)
# CMake target keeps its old test filename but is distinctly archived and labeled.
for name in ['controller.py','state_guard.py','cases.json','analyze.py']:
 t=(w.parent/'worldsense-coalesced-nopad'/name).read_text().replace('worldsense-coalesced-nopad','worldsense-small-prefill').replace('coalesced_prefill','small_prefill').replace('CoalescedPrefill','SmallPrefill')
 (w/name).write_text(t)
shutil.copy2(w.parent/'worldsense-coalesced-nopad/before-libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-06','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Use existing 16x32x32 BF16 MMA tile instead of 16x64x64 for physical prefill rows 16. Same BF16 weights, FP32 outputs and decoder. Small tile has smaller shared allocation; benefit from occupancy is a hypothesis, not established hardware evidence.','tradeoffs':'More output tiles and repeated input loads; lower shared memory per threadgroup; no extra shader/weight storage. Candidate counters only when original wide route is replaced.','stage':'Single-car WorldSense current-event screen: both modes warm, two ABBA blocks, four measured samples per policy/model. Fresh full three-frame/search ABBA/BAAB confirmation required before promotion.','gates':{'exact_tokens_and_32_state_bits':True,'actual_routes':True,'4b_ttft_min_gain':.10,'4b_total_min_gain':.03,'decode_max_regression':.02,'text_total_ttft_max_regression':.02}},indent=2))
print('Prepared WS-M5-06, no timings')
