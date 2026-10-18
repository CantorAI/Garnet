"""Prospective native local-ownership experiment; no inference source edits."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI')
w=b/'work/worldsense-owned-storage-local';s=r/'out/research/worldsense-owned-storage-local/source'
seed=r/'out/research/worldsense-minimal-selective-defaults/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=load(b/'work/worldsense-next.json');assert prior['live_job'] is None and not prior['publication_pending']
assert prior['publication']['commit']=='16c46957f528392d44c96b68ff9d0a1d1b1e7b67'
assert not w.exists() and not s.exists()
assert not subprocess.check_output(['git','-C',str(r/'Garnet'),'status','--short'])
assert subprocess.check_output(['git','-C',str(r/'Garnet'),'rev-parse','HEAD'],text=True).strip()==prior['publication']['commit']
m=load(b/'work/worldsense-minimal-selective-defaults/correctness-source-manifest.json')
assert len(m)==1721 and all(sha(seed/k)==v for k,v in m.items())
rel='src/backends/metal/metal_context.mm';t=(seed/rel).read_text();h=(seed/'src/backends/metal/metal_context.h').read_text()
block='''        id<MTLBuffer> storage=w->buffer;bool primary=false;
#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        primary=w->primaryCharged;if(primary)storage=w->primaryPayload;
#endif
'''
assert t.count(block)==2
assignments=[];accesses=[]
for i,line in enumerate(t.splitlines(),1):
 if re.search(r'->(?:buffer|primaryPayload|primaryMetadata)\b',line):accesses.append(dict(line=i,text=line))
 if re.search(r'->(?:buffer|primaryPayload|primaryMetadata)\s*=(?!=)',line):assignments.append(dict(line=i,text=line))
assert len(assignments)==3 and all('value->' in x['text'] and 'newBufferWithBytes' in x['text'] for x in assignments)
assert 'std::shared_ptr<Impl> impl_;' in h and 'friend class MetalContext;' in h
assert t.count('std::make_shared<MetalBF16Weight::Impl>()')==1
review=load(b/'work/worldsense-host-ownership-review-after76/actual-host-object-review.json')
contracts=load(b/'work/worldsense-host-ownership-review-after76/official-ownership-contracts.json')
availability=load(b/'work/worldsense-host-ownership-review-after76/compiler-attribute-availability.json')
assert availability['output'].strip()=='ATTRIBUTE_SUPPORTED'
w.mkdir();(w/'previous76-published-checkpoint.json').write_text(json.dumps(prior,indent=2)+'\n')
(w/'seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
audit=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed74_files=1721,source_sha256=sha(seed/rel),header_sha256=sha(seed/'src/backends/metal/metal_context.h'),direct_field_assignments=assignments,all_relevant_pointer_field_accesses=accesses,host_object_evidence=review,official_contracts=contracts,compiler_attribute_availability=availability,
 ownership_review='Only metal_context.mm defines private MetalBF16Weight::Impl. One make_shared constructs it; buffer/primaryPayload/primaryMetadata are strong ARC members. Three pointer assignments are construction-only before either return. Other reviewed uses validate/query/bind/read/describe or copy the shared_ptr; no writable member reference/address export or mutating callee for these three fields was found. WeightBufferInfo exposes integer identities, not Impl ownership mutation. The two changed functions copy shared_ptr w before selection, keep it lexically alive through final storage use, and neither resets/moves it. No callback or foreign writable field reference is introduced. Graph Build/Run/Close use existing mutexes. Existing unsupported concurrent mutation of the same weight handle/context remains outside the API contract; this review does not establish arbitrary whole-program data-race safety.',
 command_lifetime='Begin/Flush/readback use ordinary queue commandBuffer; Flush retains submitted buffers; SubmitAndWait drains continuations and prior submissions; Abort drains submitted work. No command factory, resource binding, retention setting, owner container, wait or exception path will change. Existing original queued-owner and Abort tests remain mandatory; new test extends alias/source release before preparation, precommit release, postFlush release, cross-context same-device use and invalid-path cleanup.',
 hypothesis='Extra local ARC operations survive current O3 host ARM64 compilation. Their request cost is unmeasured. This is not a cause claim for prior GPU spans or postdecode residuals; removing them may be too small to satisfy original speed gates.',
 prior_scope='Bounded previous review searched research log and result READMEs for ARC/local ownership. Historical WS23 latency trace did not isolate ARC cost; inventories are not ARC counts. This method changes two selected-buffer local declarations, not matrix metadata caching, QoS, command encoder reuse, or whole unretained command buffers. No global never-attempted claim.')
(w/'read-only-owner-prior-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
design='''WS-M5-77: externally retained selected-buffer locals, prospective design
Written BEFORE inference source, builds, native/model/GPU execution or timing.
Full goal ACTIVE. WS76 is a published diagnostic, WS74/75 remain rejected.
Keep deployed validated WS10 until every original promotion gate passes.

Measured evidence and hypothesis
Existing uninstrumented current host ARM64 objects contain additional buffer
local objc_retain/objc_release operations in PrepareBF16BodyShape and
EncodeBF16MatVec. This is static compiler evidence, not runtime counts, cost,
GPU ISA or a demonstrated speed bottleneck. WS76 sparse GPU diagnostics do not
identify this as a cause. The proposal tests a distinct host overhead method.

Exact method and ownership contract
Seed the full unchanged 1721-file WS74 minimal FULL/vector/selective-defaults
closure, not the WS75 raw-wide-head profile or instrumented WS76 sources.
Add default-OFF Metal-only GARNET_METAL_OWNED_STORAGE_LOCAL. At only the two
existing selected-buffer blocks, ON first determines the same primary bool,
then initializes storage once to the same raw buffer/packed payload using
Clang objc_externally_retained. Unsupported compiler attribute fails compile.
The attribute implicitly makes the local const, omits local init/destruction
ownership operations and requires a separate owner; no unsafe_unretained,
weak local or lifetime extension is substituted. OFF keeps original block
byte-exact. Keep shared_ptr w lexical lifetime and all ARC strong Impl fields.
No owner reset, const_cast, field pointer mutation/export or owner-container
change. Default retained ordinary command buffers/resource bindings remain.
The private Impl is defined only in metal_context.mm. Its three relevant
buffer pointer assignments occur during one UploadBF16 construction before
return; other reviewed accesses only read/query/bind/describe. Existing graph
mutex serialization remains, without claiming arbitrary concurrent API safety.
Prepare early return/throw and Encode validation/unwind must retain old results.
Queued GPU lifetime relies on unchanged command retention after method return,
as it already does when callers drop weight handles. Test that contract.

Allowed source delta: metal_context.mm two guarded selection blocks plus an
attribute-availability compile guard; cmake/GarnetMetal.cmake option/definition
and test registration; src/entry/metal_main.cpp exact active-profile diagnostic;
one new test2026/metal/owned_storage_local_test.cpp. No shader/program literal,
codec, FP32 accumulation, weights/precision, matrix shape, allocation, packing,
graph/session source, image path, CUDA, XLang3 engine or original test edits.
All 201 packed matrices and original five GPU readers stay unchanged. No cache,
FP16 math, answer reuse, approximation or speculative decoding. No CPU inference.

Eight frozen build roles
OFF: original raw; PRIMARY: old primary/selective profile plus new flag ON;
ON/FAULT: old minimal FULL/vector/selective with new flag ON; COMPAT56 and
COMPAT56-FAULT: exact old compatibility flags with new flag OFF; CONTROL:
matched old minimal WS74 profile with new flag OFF; RAW-BORROW: OFF plus new
flag ON. Existing six-role route/program/complete-word proof remains. New
raw-borrow/control prove flag-on raw and matched packed behavior. No future
unproven feature-on compatibility deployment is authorized by feature-off tests.
PRIMARY new flag is tested against old original source/words/states/programs.
Actual CPU objects OFF versus RAW-BORROW and CONTROL versus ON must demonstrate
removal of the selected-local ownership calls, not just profile strings or
different whole-function counts. Retained shared_ptr lifetime must remain in
compiled host functions through last use. If compiler output is unchanged or
ownership cannot be established, stop before learned/timing; no no-op trial.

Native and learned correctness before actual timing
All original nine CTests per eight roles, original 186 complete FP32 streams,
20 core-isolation records/432804 exact words per role, original actual MAIN/
prefill/empty CORE program strings and route counters. No weakened tolerances.
Add per-role binary-fraction complete-output lifetime test for raw and packed:
alias after source release before Prepare/Encode; precommit last-owner release;
postFlush last-owner release; same-device other-context ownership; invalid
shape/active-Prepare throws; abort after flushed work/owner release and recovery;
all live-weight logical counters restored. Do not use retainCount instrumentation.
Inspect and independently recompute complete outputs and counters, not summaries.
Preserve original full learned 16-check seed74 suite, all 201 matrices through
five independent GPU readers EACH 4200857600 words, raw245 HOST1724383232,
direct546 HOST334609152, 32 VLM/28 text full layer states/tokens, generic versus
resident/shared routes, 64 private steps, numerical/lifecycle/fault/late/capacity/
vision/prefix/4GiB COMPAT and all original error cases. Extend exact raw/control
model states for flag isolation; freeze planned commands/inputs/helpers before
dispatch. No timing before native/full-learned independent audit succeeds.

Original performance gates IN ORDER, unchanged
Actual candidate-only functional4 (not speed eligible) -> resource8 with >=10%
sampled Metal allocation and >=5% sampled task-footprint reductions versus
fresh RAW -> deployed WS10 speed20 ALL-model TTFT/total/deriveddecode and whole
Skill within2% -> matched PRIMARY20 >=5%4B TTFT gain plus ALL2% model/wall/Metal/
footprint preservation -> RAW20 -> full mirrored48 EACH pair independently
passing -> canonical integration/rebuild/fullproof plus48actualWS10 and app
smoke. Original quantitative/golden/metric/threshold ASTs unchanged outside
strict new native-route admission, existing configuration rejection tests kept.
No pooled rescue/favorable subset/retry of unchanged rejected settings/memory
override. A valid failure stops later stages. Publish all first/warm/failure
methods, raw metrics, exact source/build/model/input/runtime/guard identities
and keep/reject decisions. First invocation is not claimed hardware-cold.

Operational settings and limits
Real WorldSense/Galaxy native Garnet C++/Metal/XLang3 pipeline only for speed.
Fixed recorded video-derived images, normal VLM64/text16 chunks, capacity2048,
image65536/shared vision/text prefix64tokens/2entries/64MiB; original stops,
limits and Skill thresholds. Proof512VLM/256text explicitly separate. Each
request fresh native image preparation/encoding. No camera/external CPython
image prep or inference. CPython orchestration/audits only. Never inspect/read/
copy/hash/archive protected work/worldsense-http.py; use runpy login/request
only. Four normal file sources restored in finally. No consumed source/helper/
model/library/guard mutations during live jobs. During timing only bounded
observations; no build/other GPU/hash sweep/archive copy. Same process handle
polled to authoritative terminal; timeout is not termination.
Derived decode=(generated-1)/(total-TTFT), includes host/stop/postloop/possible
EOS/telemetry, not pureGPU. GPU spans overlap; medians nonadditive. Memory
snapshots not peaks/free/unique/physical residency. No clock/thermal/ISA/register/
occupancy/causal/significance claim. Fixed-frame Skills exclude live decoding/
scheduling, persisted events/shared states/dedup, retrieval/ranking/navigation/
independent-camera fusion. Second-car source correlated. M5/24GiB/4B objective
distinct from unmet M4mini/small-model/vLLM handover. Full goal stays ACTIVE.
'''
(w/'native-design-before-implementation.txt').write_text(design)
(b/'outputs/Garnet-4B-WS77-Design.txt').write_text(design)
p=dict(experiment='WS-M5-77',utc=audit['utc'],state='DESIGN_BEFORE_SOURCE',source=str(s),seed_source=str(seed),seed_files=1721,seed74_commit='e15c3f4325682098bd4c7b4a17da62254e1e36c5',publication76=prior['publication']['commit'],design_sha256=sha(w/'native-design-before-implementation.txt'),audit_sha256=sha(w/'read-only-owner-prior-audit.json'),original_gates_unchanged=True,promotion=False)
(w/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
d=dict(utc=audit['utc'],experiment='WS-M5-77',goal='active',goal_complete=False,promotion=False,state='DESIGN_BEFORE_SOURCE',work=str(w),source=str(s),seed_source=str(seed),seed_files=1721,design_sha256=p['design_sha256'],source_implemented=False,live_job=None,normal_selected='WS10',normal_final=prior['normal_final'],previous_goal_turn='progress: WS76 publication closed; actual host ownership evidence and constructor-only field/command/alias review complete; distinct prospective77design frozen before source.',publication_pending=False,next_actions=['Implement two guarded externally-retained locals plus declared CMake/profile/new lifetime test; prove exact source inverse and untouched shaders/CUDA/engine/original tests.','Eight roles; inspect real host object ownership changes before native/fulllearned/actual WorldSense gates. Keep WS10 until ALL original gates pass.'])
(b/'work/worldsense-next.json').write_text(json.dumps(d,indent=2)+'\n')
print('WS77 design saved before source; seed1721 verified, owner review frozen, eight roles planned; no build/model dispatched.')
