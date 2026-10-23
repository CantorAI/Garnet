"""Render a qualified working report from completed saved evidence only."""
import argparse,datetime,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args()
root=Path(a.root);out=Path(a.output)
def read(name):
    path=root/name
    return json.loads(path.read_text()) if path.exists() else None
def link(name):return f'[{name}](../../../work/qwen35-cuda/{name})'
def median(values):return statistics.median(values) if values else None
def fmt(value):return f'{value:,.1f}' if value is not None else 'unavailable'
lines=['# Qwen3.5 CUDA development report — in progress','',
    f'Generated from saved artifacts at {datetime.datetime.now(datetime.timezone.utc).isoformat()}.', '',
    '**Incomplete: Garnet has not achieved the 2× latency target. Broad correctness, '
    'isolated peak/residency coverage, fresh local-source Metal build preservation, and the attention prototype remain open.**','',
    'The original screenshots came from another machine. They are context, not a quantitative baseline for this RTX4080 GPU0 run.', '',
    '## Protocol and precision','',
    '- Qwen3.5-2B and 4B pinned BF16 source weights; the shared embedding/head is counted once. Original Garnet cohorts exclude unused MTP weights and have speculation OFF. Experimental Garnet MTP cohorts include their MTP engines/buffers.',
    '- Correction (October23): saved LM Studio2B batches1/2/4/8 actually use MTP, with28/56/112/224 draft-acceptance events. The requested false flag did not establish execution. Saved LM4B logs have no drafting events, so their speculation status remains unresolved. Saved vLLM configuration disables speculation. Exact proposal settings are not matched. See '+link('saved-comparator-speculation-audit.json')+'. Earlier contrary descriptions are superseded; raw timing logs are unchanged.',
    '- Garnet uses BF16 projection input/output with FP32 accumulation and reduced-precision reductions disabled; residual, norm, RoPE, attention and recurrent state are FP32. Other engines have different intermediate arithmetic. Equal BF16 source weights do not imply bit-exact execution.',
    '- Current fixed workload: 320×192 real-image variants, 60 merged visual tokens, 85 prompt tokens, 32 output-token limit, reasoning off, greedy sampling. A unique8×8 corner tile is added to each invocation; source/JPEG/decoded-RGB hashes are saved. Four related image families are not a broad quality dataset.',
    '- Seven rounds of four groups: two warmup rounds, five measured rounds. One model/engine/profile at a time. Batch groups contain B simultaneous requests inside that engine.',
    '- Current partitioned LM cohorts verify B slots,512-token slot capacity and `kv_unified=false`. Earlier unified-pool LM cohorts are retained as provisional/historical. Their slot ceiling alone did not prove independent per-row capacity.',
    '- Garnet and vLLM image hashes and prompt IDs must match. LM raw prompt IDs are unavailable; input counts and full prompt-evaluation logs are checked. Unique images and full prompt evaluations evidence actual work, not a verified setting disabling every internal cache.',
    '- Equal-output groups below contain32 sampled tokens per row in every engine. Group completion latency is not aggregate throughput or individual streaming latency. API timing boundaries differ.', '',
    'Protocol: '+link('inputs/matched512-batches/manifest.json')+'. Configuration sources: '
    '[matched profile](../../test2026/cuda/qwen35_matched_benchmark_profiles.json), '
    '[development methods](qwen35-cuda-development-log.md).','',
    '## Completed capacity-matched comparisons','',
    '| Model | Batch | Equal-output groups | Garnet ms | LM Studio ms | vLLM ms | 2× fastest-baseline target ms |',
    '|---|---:|---:|---:|---:|---:|---:|']
completed=[]
for size in ('2b','4b'):
    for batch in (1,2,4,8):
        name=f'matched512-{size}-b{batch}-partitioned-summary.json';data=read(name)
        if not data or not data.get('all_three_engines_present') or not data.get('lmstudio_independent_capacity_verified'):
            lines.append(f'| {size.upper()} | {batch} | pending | — | — | — | — |');continue
        assert data['all_input_hashes_equal'] and data['garnet_vllm_prompt_ids_equal']
        walls=data['equal_work_median_group_completion_ms'];target=min(walls['vllm'],walls['lmstudio'])/2
        lines.append(f'| {size.upper()} | {batch} | {data["equal_32_sample_groups"]} | {walls["garnet"]:.1f} | {walls["lmstudio"]:.1f} | {walls["vllm"]:.1f} | {target:.1f} |')
        completed.append((size,batch,name,data))
lines+=['','Evidence: '+', '.join(link(name) for _,_,name,_ in completed)+'.', '',
    '## Aggregate measured sampled throughput','',
    'Total sampled tokens divided by total group wall time across20 measured groups, including unequal stop lengths. '
    'This metric does not establish a per-request latency speedup.','',
    '| Model | Batch | Garnet tok/s | LM Studio tok/s | vLLM tok/s |','|---|---:|---:|---:|---:|']
for size,batch,_,data in completed:
    t=data['all_measured_aggregate_sample_throughput']
    lines.append(f'| {size.upper()} | {batch} | {t["garnet"]:.1f} | {t["lmstudio"]:.1f} | {t["vllm"]:.1f} |')
lines+=['','## Experimental direct-KV attention candidate','',
    'Opt-in `fusedAttention` is not accepted or enabled by default. Baseline files above remain unchanged. '
    'These comparisons reuse saved competitor cohorts and select four-way equal32-token groups.','',
    '| Model | Batch | Candidate | Groups | Prior Garnet ms | Candidate ms | Own latency reduction | Same sequences /28 |',
    '|---|---:|---|---:|---:|---:|---:|---:|']
for size in ('2b','4b'):
    for batch in (1,2,4,8):
        for suffix in ('-fused-attention','-fused-attention-workspace16k','-fused-attention-chunk16'):
            candidate=read(f'matched512-{size}-b{batch}{suffix}-candidate.json')
            if not candidate:continue
            w=candidate['median_group_ms'];fraction=candidate['candidate_control_latency_fraction']
            lines.append(f'| {size.upper()} | {batch} | {suffix[1:]} | {candidate["equal_32_four_way_groups"]} | {w["control"]:.1f} | {w["candidate"]:.1f} | {(1-fraction)*100:.1f}% | {candidate["same_sequence_cases"]}/28 |')
lines+=['','The strict batch4 decoder fixture passed; batch8 still fails a padded-step hidden tolerance. '
    'Six changed2B cases reproduced, with192 independent teacher-forced decisions:185 deterministic top1 matches, '
    '189 reference-maximum-set choices and192 reconstructed-head matches. Three non-tied differences remain. '
    'The oracle consumes native image embeddings; this is not independent vision or broad quality acceptance.','',
    'One instrumented2B request has GPU activity span466.237ms, union423.850ms and kernel-duration sum417.377ms. '
    'BF16 projection families and the head account for about83.2% of summed kernel time. Actual GPU copy time '
    'is6.239ms; blocking API durations must not be interpreted as transfer time. These are profiling diagnostics, '
    'not matched latency or a projected speedup. See '+link('profile-fused-attention-2b-b1-summary.json')+'.']
lines+=['','## Batch1 phase diagnostics','',
    'These are engine-specific diagnostics, not the primary latency ranking. Garnet TTFT includes reset/JPEG/vision/prefill; '
    'vLLM request metrics start after caller image decode; LM reports a server TTFT. '
    'Garnet decode uses post-prefill steps/time, vLLM uses first-to-last token timestamps, and LM supplies its own rate.','',
    '| Model | Engine / scope | Median TTFT ms | Median decode tok/s |', '|---|---|---:|---:|']
for size,batch,_,_ in completed:
    if batch!=1:continue
    g=[r for r in read(f'matched512-garnet-{size}-b1.json') if not r['warmup']]
    v=[r for r in read(f'matched512-vllm-{size}-b1.json') if not r['warmup']]
    lm=[r for r in read(f'matched512-lmstudio-{size}-b1-partitioned.json') if not r['warmup']]
    vttft=[];vdecode=[]
    for r in v:
        m=r['records'][0]['metrics'] or {}
        if m.get('first_token_latency') is not None:vttft.append(m['first_token_latency']*1000)
        delta=m.get('last_token_ts',0)-m.get('first_token_ts',0)
        if delta>0:vdecode.append((m['num_generation_tokens']-1)/delta)
    stats=[r['records'][0]['result']['stats'] for r in lm]
    phase=[('Garnet / Model API',median([r['ttft_seconds']*1000 for r in g]),median([r['decode_steps']/r['decode_seconds'] for r in g])),
        ('vLLM / scheduler request',median(vttft),median(vdecode)),
        ('LM Studio / server-reported',median([s['time_to_first_token_seconds']*1000 for s in stats]),median([s['tokens_per_second'] for s in stats]))]
    for engine,ttft,decode in phase:lines.append(f'| {size.upper()} | {engine} | {fmt(ttft)} | {fmt(decode)} |')
lines+=['','## Memory evidence and scope','',
    'The excessive-memory diagnosis led to one BF16 text-weight set per layer serving prefill/decode, '
    'one shared BF16 embedding/head, GPU-resident state, and explicit allocator tracking. '
    'This applies to the new Qwen3.5 CUDA path; the legacy Qwen3-VL path is not thereby proven memory-optimized.','',
    '| Model | Batch | Stable retained ownership GB (decimal) |', '|---|---:|---:|']
for size in ('2b','4b'):
    data=read(f'matched512-{size}-allocation-growth.json')
    if data:
        for profile in data['profiles']:lines.append(f'| {size.upper()} | {profile["batch"]} | {profile["bytes"]["owned_model_bytes"]/1e9:.3f} |')
lines+=['','Retained ownership excludes some driver/external-library/graph-executable/transient allocations. '
    'Engine live counters include weights and runtime buffers, not just workspace. Per-engine peaks are never added as a process peak. '
    'These values are not isolated process VRAM or proof of physical residency.', '',
    'Historical batch1 WDDM process-counter diagnostics were close to LM Studio, but used Garnet384 versus LM512 context; '
    'they do not substitute for current-profile isolated peak/residency measurements. WSL host VM counters had ambiguous duplicate instances; '
    'vLLM worker allocator counters were reported separately. See '+link('ISOLATED_MEMORY.md')+'.','',
    '## Correctness and regression evidence','',
    '- Component/state/masking/CUDA-graph tests cover selected independent fixtures. The real-model decoder oracle uses native image embeddings, so it bypasses independent vision validation.',
    '- Changed output trajectories after batched full-attention projections reproduced;2B and4B each matched64/64 teacher-forced decoder decisions. This is narrow trajectory evidence.',
    '- The4B batch8 candidate decoder reference had750/768 exact IDs and754/768 maximum-set choices; fourteen non-tied rank2 differences remain unresolved. A synthetic batch8 hidden check fails on both archived and current builds; layer tracing localizes its first error after an exactly matching recurrent layer.',
    '- Legacy Qwen3-VL-2B current-DLL smoke passes two16-token requests with exact text/count replay. Its preprocessing produces308patches/77visual tokens for this input, not Qwen3.5’s60. Raw IDs are not exposed by that legacy API. See '+link('qwen3-vl-matched-grid-regression.json')+'.',
    '- Metal existing-binary pretrained layer0/3 reference tests passed batches1/2 and16 stateful steps. Original/current4B Python factories produced identical8-token outputs twice on one real image. Remote native Metal sources differ from the Windows base; fresh local-source native build preservation and broad quality remain open. See '+link('metal-factory-preservation-summary.json')+'.','',
    '## Failed experiments retained','',
    '- TensorRT conditional branch around a plugin was rejected by the builder.',
    '- 4Bbatch8 attention builds exceeded64MiB and128MiB workspace caps; a256MiB layer probe and subsequent384-context full profile passed. This is not proof every512-context profile fits.',
    '- Original unified LMbatch8 load passed slot-ceiling checks but failed KV allocation on the first group. Partitioned non-unified512 slots completed. Prior shared-pool measurements remain historical/provisional.',
    '- The first legacy regression used an incompatible240-patch compiled profile; the retry derives its308-patch profile from native preprocessing.',
    '- Harness startup/schema failures and native precision discrepancies remain in raw logs; successful retries do not erase them.','',
    '## Remaining acceptance work','',
    '- All eight baseline profiles completed; preserve these artifacts and compare accepted optimizations on the same workload. Capacity and full prompt work are audited; a setting disabling every internal cache remains unverified.',
    '- Resolve or rigorously bound numerical discrepancies and add independent vision/held-out quality coverage, admission/cache-boundary tests at official-model scope.',
    '- Resolve numerical and broader pretrained-model gates for the opt-in attention candidate. TensorRT/component passing tests and selected speed measurements do not establish full model acceptance.',
    '- Repeat relevant isolated steady/peak/residency memory diagnostics on the accepted implementation and larger profiles; do not use allocator ownership as peak VRAM.',
    '- Verify fresh native Metal source/build preservation beyond same-binary factory compatibility.',
    '- Achieve the requested2× full-request latency advantage over the fastest matched baseline with acceptable memory/correctness, then audit every requirement before marking the goal complete.','']
prototype_logs={kind:root/f'test-attention-core-racefix-{kind}.log' for kind in ('parity','racecheck','memcheck','synccheck')}
if all(path.exists() for path in prototype_logs.values()):
    contents={kind:path.read_text(errors='replace') for kind,path in prototype_logs.items()}
    passed=('QWEN35_ATTENTION_CORE_COMPLETE cases 7' in contents['parity'] and
        '0 hazards displayed (0 errors, 0 warnings)' in contents['racecheck'] and
        all('ERROR SUMMARY: 0 errors' in contents[k] for k in ('memcheck','synccheck')))
    lines+=['## Attention-core component status','',
        ('Standalone7-case parity, exact KV/input/guard/graph replay checks and memcheck/racecheck/synccheck pass. '
         'The original reduction raced despite matching outputs; its failure logs/binary are retained. '
         'Opt-in serving integration now has selected measurements above; component checks alone do not establish quality or promotion.' if passed else
         'Saved component logs do not establish a complete pass; inspect failures before integration.'),'']
memory=read('memory-attention-4b-b8-512-summary.json')
if memory:
    lines+=['## Attention candidate: isolated4B batch8 memory','',
        'Context512/chunk4, same attention binary as the candidate cohort. Timings excluded. '
        'PCI/CUDA-LUID/node resolved WDDM process counters; no ambiguous nonzero duplicate instances.','',
        '| Phase | Enclosed samples | Dedicated GB median | Shared GB median |','|---|---:|---:|---:|']
    for name in ('steady_before','steady_after','after_model_close'):
        phase=memory['phases'][name];values=phase['bytes']
        lines.append(f'| {name} | {phase["samples"]} | {values["DedicatedUsage"]["median"]/1e9:.3f} | {values["SharedUsage"]["median"]/1e9:.3f} |')
    lines+=['',f'Largest observed dedicated counter {memory["largest_observed_bytes"]["DedicatedUsage"]/1e9:.3f}GB; '
        'sampled maximum is not a true lifetime peak or physical-residency proof. '
        'There is no matched large-batch comparator memory cohort yet. See '+link('memory-attention-4b-b8-512-summary.json')+'.','']
lines+=['## MTP feasibility — separate from baseline acceptance','',
    'Experimental native MTP is now connected behind a default-OFF profile (depth1/3/7). '
    'On one traced2B trajectory, independent and native implementations propose the same31 tokens, with24/31 '
    'matching teacher next tokens. Target-head reconstruction is32/32. This consumes saved native target rows; '
    'it is not end-to-end inference or a latency claim. Three synthetic MTP profiles still fail the unchanged '
    'strict numerical bound; input/unwritten-cache preservation, graph replay and borrowed-head lifetime checks '
    'completed separately. A reciprocal-normalization hypothesis did not fix the failure and was reverted.','',
    'Conditional native drafting from teacher-correct prefixes suggests18/12/11 hypothetical target passes at '
    'depth1/3/7 on this single32-token trajectory. This omits actual verification/commit/corrected-draft work '
    'and does not establish rollout acceptance or speedup. The subsequent verifier/session implementation '
    'passed rollback/prefix/capacity/stop/reset tests: six synthetic profiles with24cases each, plus real2B/4B '
    'fresh-image probes with exact32tokens and all final target state bytes equal to native serial controls. '
    'This is native target equivalence, not broad independent-model quality acceptance.','',
    'The full2B B1 C16 depth3 MTP cohort completed28requests, all28sequences matching the C16 control. '
    'Across16four-way equal32groups, latency increased353.2359→384.22405ms (+8.77%); retained ownership '
    'increased4,578,699,912→4,745,831,440B (+167,131,528). Median20target forwards include12verification '
    'and8prefix-recomputation passes. This speed candidate is rejected and stays default off. '
    'Saved vLLM disables speculation; LM Studio runtime drafting is audited above. Algorithm settings differ. See '+link('matched512-2b-b1-verified-mtp-d3-candidate.json')+'.','',
    'Indexed draft heads and recurrent prefix replay are measured separately below. '
    'Every emitted proposal remains target-verified; no overall goal or quality acceptance is implied.','']
indexed=read('matched512-2b-b1-verified-mtp-d3-indexed8192-candidate.json')
if indexed:
    w=indexed['median_group_ms'];owned=indexed['retained_owned_bytes']['candidate']
    lines+=['## Indexed MTP head — measured experiment','',
        'The proposal head reads selected rows directly from the existing tied BF16 table. A GPU stable sort selects8192 '
        'candidates from the previous target distribution. Every emitted proposal is checked by the full target head; '
        'no second embedding table or persistent gathered weight matrix is stored. Fixed independent FP64 dot bounds, '
        'CPU stable sorting, guards/invalid/alias/nonfinite/graph tests and CUDA memory/race/synchronization checks passed. '
        'Native API6CPU-oracle cases and seven synthetic session profiles passed, plus fresh-image2B/4B probes with exact '
        'serial target outputs and all final target-state bytes. This preserves native target behavior on tested cases; '
        'the pre-existing independent-model quality gaps remain.','',
        f'The28-request2B B1 C16 D3 cohort matched all{indexed["same_sequence_cases"]} control output sequences. '
        f'Across{indexed["equal_32_four_way_groups"]}common equal32groups, median latency was{w["candidate"]:.2f}ms '
        f'versus{w["control"]:.2f}ms for non-speculative C16 Garnet, {w["lmstudio"]:.2f}ms for LM Studio and{w["vllm"]:.2f}ms for vLLM. '
        f'Retained ownership was{owned[0]/1e9:.3f}GB. Speculation differs from the saved OFF comparator cohorts. '
        'The2× target is unmet; this option remains default off. See '+link('matched512-2b-b1-verified-mtp-d3-indexed8192-candidate.json')+'.','',
        '![Completed2B experiments](../../../work/qwen35-cuda/progress-speculative-2b-b1.png)','']
replays=[read(f'matched512-2b-b1-verified-mtp-d{depth}-indexed8192-delta-replay-candidate.json') for depth in (3,7)]
if any(replays):
    lines+=['## Recurrent prefix replay — measured experiment','',
        'Partial commits retain projected inputs and rerun only the small convolution/delta recurrence from saved state. '
        'They restore unaccepted attention KV suffixes and select the accepted target hidden rows. No per-token copies '
        'of all recurrent states or second set of large projection weights is stored. Helper small weights and buffers, '
        'parent retained payloads and all session allocations are counted. The original restore/recompute path remains available.','',
        'Four transactional verifier profiles and seven24-case session profiles passed exact native state/output/slot and '
        'stop/limit/rollback/capacity checks. Fresh-image2B/4B probes each passed two MTP requests plus serial controls, '
        'matching all32output IDs and every final target-state byte. These tests do not close independent quality gaps.','',
        '| Depth | Equal32groups | Garnet ms | LM Studio ms | vLLM ms | Same sequences /28 | Owned GB |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for candidate in replays:
        if not candidate:continue
        w=candidate['median_group_ms'];owned=candidate['retained_owned_bytes']['candidate'];assert len(owned)==1
        lines.append(f'| {candidate["candidate_speculative_mtp_depth"]} | {candidate["equal_32_four_way_groups"]} | {w["candidate"]:.2f} | {w["lmstudio"]:.2f} | {w["vllm"]:.2f} | {candidate["same_sequence_cases"]}/28 | {owned[0]/1e9:.3f} |')
    lines+=['','All replay requests report zero full prefix recomputations; recurrent replays are counted separately. '
        'Saved vLLM disables speculation; LM Studio runtime drafting is audited separately; algorithms differ. Default OFF,2× target unmet. '
        'No isolated process peak/residency comparison is implied by retained ownership.','']
hybrids=[read(f'matched512-2b-b1-hybrid-p{width}-d4-mtp3-k8192-replay-candidate.json') for width in (32,64,128)]
if any(hybrids):
    lines+=['## Shared projection profiles — measured experiment','',
        'Separate bounded prefill/decode graphs borrow one readonly BF16 projection pool and share the same target state/last-hidden tensors. '
        'Header-audited target matrices total2,745,434,112B for2B and7,138,181,120B for4B; real-model retained pool counters match these payloads. '
        'Both graph sets, small constants, runtime buffers and the MTP session remain included in ownership totals. '
        'Synthetic constant-versus-pooled and independent explicit-copy controls pass; this is not process-residency proof.','',
        '| Prefill / decode | Equal32groups | Garnet ms | LM Studio ms | vLLM ms | Same sequences as older C16 /28 | Owned GB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for candidate in hybrids:
        if not candidate:continue
        w=candidate['median_group_ms'];owned=candidate['retained_owned_bytes']['candidate'];assert len(owned)==1
        width=candidate['candidate_suffix'].split('-p')[1].split('-')[0]
        lines.append(f'| {width} /4 | {candidate["equal_32_four_way_groups"]} | {w["candidate"]:.2f} | {w["lmstudio"]:.2f} | {w["vllm"]:.2f} | {candidate["same_sequence_cases"]}/28 | {owned[0]/1e9:.3f} |')
    lines+=['','Cross-profile precision remains open: a synthetic narrow-versus-wide comparison differs only in key-cache values (max7.15e-7), '
        'while an identical-shape control with independent explicit state copies matches byte-for-byte. '
        'Real2B and4B probes match their own serial target outputs and final state bytes;4B differs from the older saved caption after token26. '
        'That original assertion failure remains retained. Changed full-cohort2B trajectories also need independent analysis. '
        'All options stay default OFF; no quality/default/2× acceptance. Saved vLLM disables speculation; LM Studio runtime drafting is audited separately.','']
packed=read('matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-candidate.json')
lines+=['## Packed whole-decoder experiment','',
    'An explicit default-off profile compiles each target chunk into one graph with a canonical packed state tensor. '
    'Wide prefill and narrow decode share one readonly BF16 projection pool and state. FP32 layer boundaries remain materialized. '
    'Partial acceptance uses one graph of small delta-core replays and exact accepted-span KV selection; target matrix projections are not rerun. '
    'Eight two-layer packed-program profiles passed byte-exact same-shape native controls. Four transactional verifier profiles '
    'and seven24-case speculative profiles passed rollback, partial acceptance, EOS, length and capacity checks. '
    'Independent pretrained quality, process memory and2× acceptance remain open.','']
if packed:
    w=packed['median_group_ms'];owned=packed['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=[f'Completed2B B1 cohort: {w["candidate"]:.2f}ms versus same-profile per-layer control {w["control"]:.2f}ms, '
        f'saved LM Studio {w["lmstudio"]:.2f}ms and vLLM {w["vllm"]:.2f}ms across{packed["equal_32_four_way_groups"]}equal32groups. '
        f'{packed["same_sequence_cases"]}/28 output sequences match the same-profile control; retained ownership {owned[0]/1e9:.3f}GB. '
        'Saved vLLM disables speculation; LM Studio runtime drafting is audited separately. No quality/default/2× acceptance is inferred.','']
else:lines+=['No completed packed-program matched timing cohort is available yet. Component/diagnostic times are not speed evidence.','']
resident=read('matched512-2b-b1-hybrid-p64-d4-mtp3-k8192-replay-resident-candidate.json')
if resident:
    w=resident['median_group_ms'];owned=resident['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=['## Inline request controls and direct chunk embeddings','',
        'The default-off residentControls profile transports small scalars through CUDA launch parameters and gathers the shared BF16 table into caller-owned padded chunks. '
        'It changes data transport and temporary allocation behavior while preserving model arithmetic. Sixteen independent IO profiles, '
        'four parent controls, four transactional verifier profiles and eight24-case speculative profiles passed exact native comparisons; '
        'all three native IO sanitizer checks found zero errors.','',
        f'Completed2B B1 cohort: {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms for the same-profile old IO control, '
        f'across{resident["equal_32_four_way_groups"]}equal32groups; all{resident["same_sequence_cases"]}/28sequences match. '
        f'Retained ownership {owned[0]/1e9:.3f}GB. Asynchronous prefill grew the MTP context pool from1to8contexts, '
        'adding31.1MB. A separate default-off serial graph reuse experiment addresses this growth; its result appears below when available. '
        'The2× target remains unmet, independent quality gaps remain, and saved vLLM disables speculation; LM Studio runtime drafting is audited separately.','']
serial=read('matched512-2b-b1-hybrid-p64-d4-mtp3-k8192-replay-resident-serial-candidate.json')
if serial:
    w=serial['median_group_ms'];owned=serial['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=['## Serial MTP graph reuse — measured memory fix','',
        f'The same16equal32groups measure {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms for resident IO with the old pool policy. '
        f'All{serial["same_sequence_cases"]}/28output sequences match. Retained ownership returns to {owned[0]/1e9:.3f}GB, '
        'saving31,098,368B with oneMTPcontext. No latency gain is claimed. '
        'Pending replay requires a captured graph, identical bindings, static shapes, same stream/thread semantics and no provider owner. '
        'Changed cases retain conservative retirement. Gated real-TRT pool tests and all three sanitizers passed; '
        'ten24-case MTP profiles preserve exact states/outputs/stops/limits. Default OFF; independent quality, physical memory and2× acceptance remain open.','']
register=read('matched512-2b-b1-hybrid-p64-d4-mtp3-k8192-replay-resident-serial-register128-candidate.json')
if register:
    w=register['median_group_ms'];owned=register['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=['## Fixed128 recurrent state kernel — measured experiment','',
        'The optional kernel keeps the original sequential FP32 recurrence/reductions and convolution workspace, '
        'using fixed128-state loops and volatile shared Q/K. Native parent, prefix-commit and MTP controls pass exact bytes; '
        'all49pretrained final target-state hashes match the same-profile old kernel. Standalone independent FP64 and all three sanitizers pass. '
        'Native compiler resources differ from the standalone probe, so component speeds do not predict full-model gains.','',
        f'Completed2B B1 cohort: {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms, '
        f'across{register["equal_32_four_way_groups"]}equal32groups, all{register["same_sequence_cases"]}/28output sequences matching. '
        f'Retained ownership remains {owned[0]/1e9:.3f}GB. Saved LM Studio {w["lmstudio"]:.2f}ms and vLLM {w["vllm"]:.2f}ms have different or unresolved proposal settings. '
        'The2× target, independent quality, physical-memory and all-profile acceptance remain open; option stays default OFF.','']
flat=read('matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-flat-candidate.json')
if flat:
    w=flat['median_group_ms'];owned=flat['retained_owned_bytes'];assert len(owned['candidate'])==len(owned['control'])==1
    lines+=['## Flat packed-state concatenation','',
        'The optional variadic concatenation joins layer payloads once instead of retaining a chain of intermediate concatenations. '
        'CPU row-order controls include batch8 and129 parts; native parent, verifier and speculative controls preserve exact bytes. '
        'The real2B probe preserves canonical-state and last-hidden hashes.','',
        f'The flat-only cohort measures {w["candidate"]:.2f} ms versus {w["control"]:.2f} ms for chained packing, '
        f'with {flat["same_sequence_cases"]}/28 sequences matching. Retained ownership is {owned["candidate"][0]/1e9:.3f} GB, '
        f'a reduction of {owned["control"][0]-owned["candidate"][0]:,} bytes. This is retained allocation accounting, not physical peak VRAM.','']
lines+=['## Combined packed, resident, serial-reuse and register profiles','',
    'These experiments combine several options. Their differences from the original target can include chunk-size arithmetic changes; '
    'same-sequence counts compare the saved control indicated by each artifact. '
    'Independent quality remains open. The2B batch1 row uses the completed vLLM MTP3 comparator; other vLLM rows use saved serial cohorts. '
    'LM Studio runtime drafting is audited above. Each row uses its own common equal-output subset.','',
    '| Model | Batch | Equal32 groups | Garnet ms | LM Studio ms | vLLM ms | 2× target ms | Owned GB | Same sequences /28 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
combined_suffix='-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat'
for size in ('2b','4b'):
    for batch in (1,2,4,8):
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-candidate.json')
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-candidate.json') or candidate
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-candidate.json') or candidate
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-candidate.json') or candidate
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-candidate.json') or candidate
        if not candidate:continue
        w=candidate['median_group_ms'];owned=candidate['retained_owned_bytes']['candidate']
        target=min(w['lmstudio'],w['vllm'])/2
        owned_text=f'{min(owned)/1e9:.3f}' if len(owned)==1 else f'{min(owned)/1e9:.3f}–{max(owned)/1e9:.3f}'
        lines.append(f'| {size.upper()} | {batch} | {candidate["equal_32_four_way_groups"]} | {w["candidate"]:.2f} | '
            f'{w["lmstudio"]:.2f} | {w["vllm"]:.2f} | {target:.2f} | {owned_text} | {candidate["same_sequence_cases"]}/28 |')
lines+=['','Missing rows have no completed cohort. No projected speed or memory values are included. '
    'Ownership ranges retain allocation growth instead of selecting the lowest observation. '
    'The2× goal remains unmet; these options remain default OFF.','']
mtp_compare=read(f'matched512-2b-b1{combined_suffix}-mtp3-cache128-candidate.json')
if mtp_compare:
    w=mtp_compare['median_group_ms'];target=min(w['vllm'],w['lmstudio'])/2
    lines+=['## New vLLM MTP comparator — completed2B batch1','',
        'The same pinned BF16 checkpoint, JPEGs, prompt IDs,512-token capacity and32-token limit now run with vLLM native MTP depth3. '
        'Cache budgets96MiB and112MiB per sequence both failed admission; their errors remain preserved. '
        '128MiB is the smallest successful tested budget, not a proven mathematical minimum. '
        'The four-request admission smoke is separate from the full28-request cohort.','',
        f'Across {mtp_compare["equal_32_four_way_groups"]} common equal32 groups, Garnet measures {w["candidate"]:.2f} ms, '
        f'vLLM MTP {w["vllm"]:.2f} ms and LM Studio {w["lmstudio"]:.2f} ms. '
        f'The current2× threshold is {target:.2f} ms. Garnet is slower than this MTP comparator. '
        'vLLM MTP matches17/28 saved vLLM serial sequences; independent quality remains open. '
        'FP32 convolution/recurrent caches remain requested. Runtime selects the Triton GDN speculative path and logs that '
        'the fused CUDA path requires BF16 convolution cache. A separately identified native-cache profile may be faster and is still pending. '
        'Exact proposal settings and intermediate arithmetic are not equivalent across engines.','',
        'Evidence: '+link('matched512-vllm-2b-b1-mtp3-cache128.json')+', '+link(f'matched512-2b-b1{combined_suffix}-mtp3-cache128-candidate.json')+'.','']
capture_controls=[read(f'packed-capture-lifetime-g{g}-s{s}-memcheck-exit.json') for g,s in ((0,0),(1,0),(1,1))]
if all(capture_controls):
    lines+=['## CUDA capture lifetime gate — unresolved','',
        'All18 target-serial native integration profiles preserve states, tokens, stops, limits, rollback and one-context ownership. '
        'However, the full batch8 memcheck reports use-before-allocation errors in a TensorRT Myelin fused kernel. '
        'A same-profile target-reuse-OFF control also fails. Target-only isolation with no MTP/replay passes when capture is OFF '
        'and fails when capture is ON, with and without target serial reuse. '
        'Target serial reuse remains default OFF; no latency or memory improvement has been measured for it.','',
        'Eight independent plain-CUDA allocation/capture controls pass memcheck with zero errors. '
        'Skipping unchanged context binding setters did not fix the TensorRT failure; that diagnostic is rejected as the fix. '
        'The failing10,633,216-byte buffer is absent from the synchronous runtime allocator trace. '
        'Overriding the distinct TensorRT temporary-storage allocator also did not fix it. '
        'No sanitizer error is suppressed or treated as a pass. The current capture gate prevents acceptance.','']
zero_aux=read('zero-aux-capture-control-comparison.json')
if zero_aux:
    assert len(zero_aux)==3 and all(v['exit']==0 and v['all_states_ids_equal_old_noncapture'] for v in zero_aux)
    lines+=['## Single-stream capture route — validation in progress','',
        'The default wide target graph uses two auxiliary streams; decode and final norm use zero. '
        'Its captured prefill graph has52 nodes and no allocation/free/child nodes, so the failing buffer was allocated outside that graph. '
        'The optional zero-auxiliary-stream build passes all three target-only memcheck controls: capture OFF, capture ON, '
        'and capture ON with serial target reuse. All final state hashes, tokens and slots match the older uncaptured control. '
        'Only serial target reuse bounds all three graph context counts to one. '
        'This is a measured workaround for the isolated route, not a universal TensorRT bug diagnosis or performance claim.','']
    gates=[read(f'target-serial-session-b8-p64-{kind}-aux0-exit.json') for kind in ('memcheck','racecheck','synccheck')]
    lines+=['Full batch-profile and MTP sanitizer gates: '+('all three completed with exit0; inspect retained logs for scope.' if all(gates) and all(v['exit_code']==0 for v in gates) else 'full MTP memcheck passed; full integration racecheck was interrupted and remains incomplete. Full integration synccheck was not run.'),'']
    component=[read(f'native-register128-program-b8-c4-{kind}-aux0-exit.json') for kind in ('racecheck','synccheck')]
    if all(component) and all(v['exit_code']==0 for v in component):
        lines+=['Exact native-DLL two-layer C4/B8 packed-versus-per-layer component checks pass racecheck and synccheck with zero findings and12 exact records each. '
            'The Garnet kernel-name filter excludes vendor kernels. This bounded component coverage does not replace the incomplete full24-case MTP race/sync checks.','']
gm=read('memory-combined-4b-b8-512-summary.json')
lm=read('memory-lmstudio-4b-b8-512-partitioned-summary.json')
if gm and lm:
    ge=read('memory-combined-4b-b8-512-session.json');le=read('memory-lmstudio-4b-b8-512-partitioned-session.json')
    gh=[e['input_sha256'] for e in ge if 'input_sha256' in e]
    lh=[e['result']['input_sha256'] for e in le if 'result' in e]
    assert gh==lh and len(gh)==6
    lines+=['## Matched4B batch8 physical memory diagnostic','',
        'Each engine ran alone with the same six JPEG groups,8 slots and512-token independent capacity. '
        'Garnet uses the combined MTP profile; the saved4B GGUF lacks MTP tensors. Proposal configuration differs. '
        'Loading is observed only for Garnet. Sampling and idle holds invalidate latency measurement; the primary cohorts above are separate.','',
        '| Engine | Warm dedicated GB | Warm shared GB | Largest observed dedicated GB | Retained owned GB |',
        '|---|---:|---:|---:|---:|']
    for label,data in (('Garnet combined',gm),('LM Studio',lm)):
        steady=data['phases']['steady_after']['bytes'];owned=data['owned_bytes_after_warmup']
        lines.append(f'| {label} | {steady["DedicatedUsage"]["median"]/1e9:.3f} | {steady["SharedUsage"]["median"]/1e9:.3f} | '
            f'{data["largest_observed_bytes"]["DedicatedUsage"]/1e9:.3f} | '+(f'{max(owned)/1e9:.3f}' if owned else 'unavailable')+' |')
    lines+=['','The short Garnet diagnostic kept one target-prefill context. The longer28-request timing cohort allocated a second context '
        'and reached16.480GB retained ownership, with several requests above6seconds. Both observations are retained; '
        'the short diagnostic does not describe the longer cohort. Shared usage also remained after model close, so its full amount '
        'cannot be attributed to live model spill. A causal paging diagnosis remains unproven. '
        'LM Studio unloaded its worker before an enclosed close-phase sample was available. '
        'Largest sampled usage is not a true lifetime peak. Memory acceptance remains open.','',
        'Evidence: '+link('memory-combined-4b-b8-512-summary.json')+', '+link('memory-lmstudio-4b-b8-512-partitioned-summary.json')+'.','']
new_route=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-candidate.json')
deferred=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-candidate.json')
if new_route:
    w=new_route['median_group_ms']
    lines+=['## Bounded target contexts with zero auxiliary streams — measured4B batch8','',
        f'Across{new_route["equal_32_four_way_groups"]}common equal32groups, the new route measures {w["candidate"]:.2f}ms versus '
        f'{w["control"]:.2f}ms for the older combined control; all{new_route["same_sequence_cases"]}/28 output sequences match. '
        'The longer timing cohort now retains one wide target context, removing945,818,112B of duplicate context allocation. '
        'This combines serial target reuse with zero auxiliary streams; the latency change does not isolate either option. '
        'The short old memory diagnostic already had one context, so it cannot establish the same physical-memory improvement.','']
if deferred:
    w=deferred['median_group_ms'];owned=deferred['retained_owned_bytes'];target=min(w['vllm'],w['lmstudio'])/2
    lines+=['## Deferred canonical state commit — measured4B batch8','',
        'The optional transaction borrows canonical state unchanged while target verification computes proposed outputs. '
        'Rollback restores small request buffers; accepted prefixes write canonical state only at the final commit. '
        'Pending transactions reject other parent mutation. A final-write failure poisons the decoder until reset. '
        'This removes the full canonical backup without changing the CUDA recurrence or precision. Default OFF.','',
        'All12 integration profiles exited0, including parent ownership, invalid-use, partial/full acceptance and injected failure recovery. '
        'The batch8 verifier and full24-case MTP session both passed unfiltered memcheck with zero errors. '
        'These controls do not resolve existing independent pretrained quality gaps.','',
        f'The28-request real-image cohort measures {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms on '
        f'{deferred["equal_32_four_way_groups"]}common equal32groups, with all{deferred["same_sequence_cases"]}/28 sequences unchanged. '
        f'Retained ownership falls from{max(owned["control"]):,}B to{max(owned["candidate"]):,}B '
        f'(-{max(owned["control"])-max(owned["candidate"]):,}B). '
        f'vLLM measures{w["vllm"]:.2f}ms, LM Studio{w["lmstudio"]:.2f}ms; the2× threshold{target:.2f}ms remains unmet.','',
        'Evidence: '+link(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-candidate.json')+', '
        +link('deferred-state-verifier-b8-memcheck-exit.json')+', '+link('deferred-state-session-b8-memcheck-exit.json')+'.','']
route_memory=[('Garnet zero-aux / serial target','memory-combined-4b-b8-512-targetserial-aux0-summary.json'),
              ('Garnet plus deferred commit','memory-combined-4b-b8-512-targetserial-aux0-deferred-summary.json'),
              ('Garnet plus native prefix commit','memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-summary.json'),
              ('Garnet plus shared output storage','memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-summary.json'),
              ('LM Studio','memory-lmstudio-4b-b8-512-partitioned-summary.json')]
if read(route_memory[1][1]):
    lines+=['## Latest isolated4B batch8 memory counters','',
        'Sequential runs use the same six JPEG groups. Dedicated/shared counters are sampled WDDM process usage; '
        'they do not prove lifetime peaks or physical residency. Loading was not observed for LM Studio. MTP configurations differ.','',
        '| Profile | Warm dedicated GB | Warm shared GB | Largest observed dedicated GB | Owned GB |',
        '|---|---:|---:|---:|---:|']
    for label,name in route_memory:
        data=read(name)
        if not data:continue
        steady=data['phases']['steady_after']['bytes'];owned=data['owned_bytes_after_warmup']
        lines.append(f'| {label} | {steady["DedicatedUsage"]["median"]/1e9:.3f} | {steady["SharedUsage"]["median"]/1e9:.3f} | '
            f'{data["largest_observed_bytes"]["DedicatedUsage"]/1e9:.3f} | '+(f'{max(owned)/1e9:.3f}' if owned else 'unavailable')+' |')
    lines+=['','Evidence: '+', '.join(link(name) for _,name in route_memory if read(name))+'. Memory acceptance remains open.','']
native_gates=[read(f'native-prefix-session-b8-{kind}-exit.json') for kind in ('memcheck','racecheck','synccheck')]
if all(native_gates) and all(g['exit_code']==0 for g in native_gates):
    lines+=['## Native state-only prefix commit — experimental','',
        'The default-off nativePackedPrefixCommit route replaces the packed replay TensorRT graph with separate CUDA state-only kernels. '
        'It recomputes accepted convolution/recurrent state in place and copies only accepted KV slots from authoritative target outputs. '
        'The canonical state stays unchanged until this final commit; failed final writes poison the decoder. '
        'It requires the deferred-state and register128 profiles. Small convolution/dt/A_log parameters and convolution workspace are explicitly counted. '
        'TensorRT input/output aliasing is not used.','',
        'All12 numerical profiles pass exact ordinary-native states/outputs, including B1/2/4/8, C4/C16 and P32/P64. '
        'The B8 verifier and full24-case MTP session pass unfiltered memcheck. '
        'Racecheck/synccheck pass with zero findings for the three new packedCommit kernels inside the full MTP session; other kernels are excluded from these two checks. '
        'Invalid spans/layouts are rejected before mutation, and post-write injected failures require reset. '
        'These scopes do not replace missing full integration race/sync coverage or existing independent pretrained quality gaps.','']
    native_probe=read('native-prefix-pretrained-4b-b8.json')
    if native_probe:
        lines+=['The real4B B8 probe '+('passes' if all(native_probe[k] for k in ('same_recorded_sequences','same_ordinary_outputs','all_state_hashes_equal')) else 'FAILS')+
            ': two native-prefix requests followed by a separately loaded non-MTP target. '
            'Full finite target-state hashes stream bounded column chunks; timings are invalidated by diagnostic hashing. '
            'Evidence: '+link('native-prefix-pretrained-4b-b8.json')+'.','']
    native_cohort=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-nativecommit-candidate.json')
    if native_cohort:
        w=native_cohort['median_group_ms'];owned=native_cohort['retained_owned_bytes']
        lines+=[f'The full28-request4B B8 cohort measures {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms on '
            f'{native_cohort["equal_32_four_way_groups"]}common equal32groups, with{native_cohort["same_sequence_cases"]}/28 unchanged sequences. '
            f'Retained ownership is{max(owned["candidate"]):,}B versus{max(owned["control"]):,}B. '
            f'vLLM serial{w["vllm"]:.2f}ms and LM Studio{w["lmstudio"]:.2f}ms; target{min(w["vllm"],w["lmstudio"])/2:.2f}ms remains unmet.','']
arena_storage=read('output-arena-storage-fusion.json')
if arena_storage:
    lines+=['## Shared packed output storage — experimental','',
        'A prospective design records one maximum-sized allocation with stable exact-shape SDK storage views for sequential packed prefill/decode outputs. '
        'The generic binding API checks static output shape, dtype, device, contiguity, alignment and storage span before accepting it; input/output overlap and late rebinding are rejected. '
        'The decoder counts the shared pool once and excludes borrowed output views from each graph\'s owned-output bytes. '
        'The initial test harness failed before engine execution because it omitted the required fusion graph boundary; its source/archive/log/exit1 are retained. '
        'The corrected harness preserves the same exact-output and lifecycle assertions.','',
        'The generic corrected test passes two interleaved graph shapes, capture/replay, caller-reference drop, closure of other owners and invalid storage boundaries. '
        'Model validation and sanitizer results are listed only when their explicit exit records exist. Default OFF. '
        'Evidence: '+link('shared-output-arena-design.json')+', '+link('output-arena-storage-fusion.json')+'.','']
    gates=[read(f'output-arena-{mode}-b8-memcheck-exit.json') for mode in ('storage','verifier','session')]
    if all(gates) and all(g['exit_code']==0 for g in gates):
        lines+=['All12 packed verifier/MTP profiles pass exact states and outputs. Generic storage, B8 verifier and full24-case B8 MTP session pass unfiltered memcheck with zero errors. '
            'This covers memory access/lifetime, not full integration race/synccheck or independent quality acceptance.','']
    arena_probe=read('output-arena-pretrained-4b-b8.json')
    if arena_probe:
        lines+=['The real4B B8 shared-output probe '+('passes' if all(arena_probe[k] for k in ('same_recorded_sequences','same_ordinary_outputs','all_state_hashes_equal')) else 'FAILS')+
            ': two production MTP requests followed by a separately loaded ordinary target with independent outputs. '
            'All full finite canonical-state and last-hidden hashes match. '
            f'Probe retained ownership is{arena_probe["native_memory"]["owned_model_bytes"]:,}B; this is not process VRAM or cohort timing. '
            'Evidence: '+link('output-arena-pretrained-4b-b8.json')+'.','']
    cohort=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-candidate.json')
    if cohort:
        w=cohort['median_group_ms'];owned=cohort['retained_owned_bytes']
        lines+=[f'Completed4B B8: {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms on{cohort["equal_32_four_way_groups"]}common equal32groups; '
            f'{cohort["same_sequence_cases"]}/28 unchanged sequences. Owned{max(owned["candidate"]):,}B '
            f'versus{max(owned["control"]):,}B, a measured reduction of{max(owned["control"])-max(owned["candidate"]):,}B. '
            'Retained allocation accounting is separate from the isolated physical-counter table. The2× goal remains unmet.','']
whole_validation=read('whole-program-session-b8-p64-d4-exit.json')
if whole_validation:
    lines+=['## Scoped whole-program CUDA state-kernel compilation','',
        'The resource audit isolates relocatable device code as a register-count difference: the same native and volatile-template source bodies use168 registers without RDC and255 with it. '
        'Those object counts precede final device linking and do not establish model speed. '
        'The initial audit mirrored compiler FLAGS but omitted the hard-coded CMake rule addition; the second run adds only-rdc=true and records the correction.','',
        'The default-off GARNET_QWEN35_WHOLE_PROGRAM_CUDA option moves only the self-contained delta and packed-prefix CUDA units to a non-separable object target. '
        'Other CUDA sources retain their existing linking mode. No algorithm, precision, fast-math, projection or layout change is requested. '
        'The actual DLL reports168 registers/zero stack for target and prefix recurrence; a marker compiled in the CUDA unit identifies the real policy. '
        'Benchmark/probe labels must match that marker. All12 original verifier/MTP profiles pass exact state/output/lifecycle controls.','',
        'Evidence: '+link('qwen35-whole-program-design.json')+', '+link('register-template-audit-comparison.json')+', '+link('whole-program-dll-resources.txt')+'.','']
    lines+=['| Model / batch | Groups | Previous ms | Whole-program ms | Same sequences | Owned GB |',
        '|---|---:|---:|---:|---:|---:|']
    for size,batch in (('2b',1),('4b',8)):
        suffix='-mtp3-cache128' if size=='2b' else ''
        d=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-wholeprogram{suffix}-candidate.json')
        if not d:continue
        w=d['median_group_ms'];owned=d['retained_owned_bytes']['candidate'];assert d['candidate_whole_program_cuda'] is True
        lines.append(f'| {size.upper()} / {batch} | {d["equal_32_four_way_groups"]} | {w["control"]:.2f} | {w["candidate"]:.2f} | {d["same_sequence_cases"]}/28 | {max(owned)/1e9:.3f} |')
    lines+=['','Missing rows are pending. Independent quality, other batch profiles and 2× acceptance remain open. '
        'The 2B batch-1 result is a latency regression despite the lower register count. '
        'The option remains experimental and default off; this table reports measured candidates, not accepted improvements.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'whole-program-pretrained-{size}-b{batch}-cross-rdc-state-control.json';d=read(name)
        if d:
            lines+=[f'{size.upper()} B{batch} cross-build pretrained finite-state/output preservation: '+('PASS' if d['all_passed'] else 'FAIL')+
                '. Identical real inputs/profile; full ordinary and repeated speculative state hashes/signatures compared. Evidence: '+link(name)+'.','']
    exits=[read(f'whole-program-{mode}-b8-memcheck-exit.json') for mode in ('storage','verifier','session')]
    if all(exits) and all(d['exit_code']==0 for d in exits):
        lines+=['Generic storage, B8 verifier and full24-case MTP unfiltered memchecks pass with zero errors. '
            'Exact-DLL C4/B8 two-layer program/control race/sync checks pass with zero findings. '
            'New prefix-kernel race/sync checks also pass in the full MTP session. Their kernel-name filters limit scope; full integration race/sync remains incomplete.','']
finite_check=read('finite-f32-bytes-validation.json')
finite_memcheck=read('finite-f32-bytes-memcheck-exit.json')
finite_device=read('finite-f32-bytes-device-preservation.json')
if finite_check and finite_memcheck and finite_device:
    lines+=['## Bounded finite-state audit transfer','',
        'A host diagnostic exports at most 1,048,576 contiguous FP32 words (4 MiB) per call, '
        'rejects every NaN/Inf and returns exact little-endian bytes for SHA-256. '
        'The original column chunk order and full target-state coverage remain unchanged. '
        'The inference path does not call this helper, so this is not an inference speed improvement.','',
        f'Exact finite-byte, signed-zero, subnormal, maximum-finite, legacy SHA, rejection, '
        f'logical-view and lifetime checks: {len(finite_check["checks"])} passed. '
        f'Unfiltered memcheck exit: {finite_memcheck["exit_code"]}. '
        f'CUDA device-fatbin equality with the prior build: {finite_device["device_fatbin_equal"]}. '
        'Device equality does not by itself establish host correctness.','',
        'Evidence: '+link('finite-f32-bytes-validation.json')+', '+link('finite-f32-bytes-memcheck-exit.json')+', '+link('finite-f32-bytes-device-preservation.json')+'.','']
    interrupted=read('whole-program-pretrained-4b-b8-exit.json')
    if interrupted and interrupted.get('status')=='INTERRUPTED_INCOMPLETE':
        lines+=['The earlier 4B batch-8 scalar-conversion probe was interrupted while CPU-active after over 13 minutes. '
            'Its logs and incomplete exit are retained. No success, kernel failure or timing result follows from that interruption. '
            'Redirected stdout can be buffered, so the absence of a printed phase marker does not localize the delay.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'whole-program-fastbytes-pretrained-{size}-b{batch}-cross-rdc-state-control.json';d=read(name)
        if d:
            lines+=[f'{size.upper()} B{batch} bounded-byte real probe versus saved RDC full-state/output hashes: '
                +('PASS' if d['all_passed'] else 'FAIL')+'. Evidence: '+link(name)+'.','']
    d=read('finite-f32-bytes-2b1-cross-method-control.json')
    if d:
        lines+=['2B B1 bounded-byte versus saved same-policy legacy full-state/output hashes: '
            +('PASS' if d['all_passed'] else 'FAIL')+'. Evidence: '+link('finite-f32-bytes-2b1-cross-method-control.json')+'.','']
all_name='matched512-4b-b8-whole-program-all-baselines.json'
all_baselines=read(all_name)
if not all_baselines:
    all_name='matched512-4b-b8-native-arena-all-baselines.json';all_baselines=read(all_name)
if all_baselines:
    lines+=['## All tested 4B batch-8 comparators on one subset','',
        f'Every following value uses the same {all_baselines["common_equal32_groups"]} equal-32 groups, including serial and MTP vLLM. '
        'This is distinct from the 15-group four-way cohort table.','',
        '| Profile | Median group ms |','|---|---:|']
    for name,value in all_baselines['median_group_ms'].items():lines.append(f'| {name} | {value:.2f} |')
    lines+=['',f'The fastest tested baseline is {all_baselines["fastest_baseline"]}; the half-latency target of {all_baselines["half_fastest_baseline_ms"]:.2f} ms remains unmet. '
        'Evidence: '+link(all_name)+'. The source manifest identifies the exact candidate and control; '
        'the whole-program candidate remains experimental, with a 2B batch-1 regression and an unrepeated small 4B batch-8 difference.','']
v4mtp=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-mtp3-cache128-convbfloat16-candidate.json')
if v4mtp:
    w=v4mtp['median_group_ms']
    lines+=['## Additional vLLM4B batch8 MTP baseline','',
        'A separate28-request vLLM MTP3 profile uses BF16 convolution cache and FP32 recurrent state, with128MiB cache per sequence and512-token capacity. '
        'Its runtime log explicitly selects the CUDA GDN speculative kernel. '
        f'On{v4mtp["equal_32_four_way_groups"]}common equal32groups it measures{w["vllm"]:.2f}ms versus '
        f'{w["candidate"]:.2f}ms for the deferred Garnet profile. '
        'The saved vLLM serial cohort is faster, so this slower MTP result does not replace it as the goal baseline. '
        'Proposal algorithms and intermediate arithmetic differ, and independent quality remains open.','']
matrix=read('cublaslt-profile-grid-summary.json')
if matrix:
    lines+=['## Isolated projection-layout screen','',
        f'{len(matrix["profiles"])} matrix shapes cover decode rows 4/8/16/32/64 and prefill rows 64/128/256/512 '
        'for the existing 2B/4B gate/up/down dimensions. Historical quick/full grids remain unchanged. '
        'Candidates retain BF16 inputs/output, FP32 accumulation, eligible compute-type reductions and the original 128 FP64 sample bounds per result.','',
        '| Screened candidate class | Median component ratio | Maximum component ratio |',
        '|---|---:|---:|']
    for key,label in [('best_same_layout_zero_workspace','Current layout, zero workspace'),
                      ('best_same_layout','Current layout, bounded workspace'),('best_any_layout','Allow transposed layout')]:
        ratios=[r[key]['sweep_speed_ratio'] for r in matrix['profiles']]
        lines.append(f'| {label} | {statistics.median(ratios):.3f} | {max(ratios):.3f} |')
    lines+=['','Ratios summarize heterogeneous isolated matrices and are not weighted model-speed estimates. '
        'The 256 MiB sweep does not establish L2 residency. Candidates were selected from multiple algorithms in one screen, '
        'so repeats and full-model numerical/lifecycle validation would be required before integration. '
        'Algorithm IDs alone do not describe complete heuristic configurations. No projection change was promoted. '
        'Evidence: '+link('cublaslt-profile-grid-summary.json')+'.','']
wide_exits=[read(f'wide8-session-b{b}-p{p}-d8-mtp7-exit.json') for b in (1,2,4,8) for p in (32,64)]
wide_exits += [read(f'wide8-verifier-b{b}-c{c}-exit.json') for b,c in ((8,8),(4,8),(8,16),(4,16))]
if all(wide_exits) and all(e['exit_code']==0 for e in wide_exits):
    lines+=['## Wider D8/MTP7 experiment','',
        'Four verifier profiles (B4/B8, C8/C16, verification width 8) and eight speculative-session profiles '
        '(B1/B2/B4/B8, P32/P64, D8/MTP7) pass exact state/output/lifecycle checks. '
        'Saved graph markers confirm the restored RDC build. These synthetic checks do not establish pretrained quality or speed.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'wide8-pretrained-{size}-b{batch}.json';d=read(name)
        if d:
            lines+=[f'{size.upper()} B{batch} real probe: ordinary-output match {d["same_ordinary_outputs"]}, '
                f'full finite-state hashes match {d["all_state_hashes_equal"]}, '
                f'saved D4 sequences match {d["same_saved_d4_sequences"]}. Evidence: '+link(name)+'.','']
        suffix='-mtp3-cache128' if size=='2b' else ''
        name=f'matched512-{size}-b{batch}-packed-p64-d8-mtp7-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena{suffix}-candidate.json'
        d=read(name)
        if d:
            w=d['median_group_ms']
            lines+=[f'{size.upper()} B{batch}, {d["equal_32_four_way_groups"]} equal-32 groups: '
                f'D8/MTP7 {w["candidate"]:.2f} ms versus D4/MTP3 {w["control"]:.2f} ms; '
                f'{d["same_sequence_cases"]}/28 same sequences. Independent quality and 2× acceptance remain open. '
                'Evidence: '+link(name)+'.','']
conditional=[read(f'conditional-control-api13-{kind}-exit.json') for kind in ('plain','memcheck','racecheck','synccheck')]
if all(conditional) and all(e['exit_code']==0 for e in conditional):
    lines+=['## Conditional GPU-control feasibility probe','',
        'An isolated WHILE graph containing a captured child graph passes 84 bounded-counter cases on SM89, '
        'including zero/short/32 limits, mixed rows, invalid limits and repeated reset. '
        'The source capture is destroyed before parent instantiation and launch. '
        'Normal execution and unfiltered memcheck/racecheck/synccheck pass with zero findings. '
        'The initial compile failure used an older API signature; its log and source remain retained.','',
        'This is a control/lifetime prerequisite, with no model weights, speculative acceptance algorithm or inference-speed claim. '
        'A future resident serving loop would need stable child graphs and resource ownership outside the conditional body. '
        'NVIDIA documents restricted body-node types in its '
        '[CUDA 13.2 graph guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements). '
        'Evidence: '+link('conditional-control-api13-identity.json')+', '+link('conditional-control-design.json')+'.','']
static_name='matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-staticbindings-mtp3-cache128-candidate.json'
static=read(static_name)
if static:
    w=static['median_group_ms']
    lines+=['## TensorRT binding-reuse diagnostic','',
        f'2B B1, {static["equal_32_four_way_groups"]} common equal-32 groups: '
        f'{w["candidate"]:.2f} ms versus historical control {w["control"]:.2f} ms and vLLM MTP {w["vllm"]:.2f} ms. '
        f'{static["same_sequence_cases"]}/28 same saved sequences; retained ownership unchanged. '
        'All 12 C4 numerical profiles, three unfiltered memchecks and a real full finite-state/output probe pass. '
        'The small timing difference is unrepeated; the option remains unselected/default off. '
        'This does not establish the 2× target. Evidence: '+link(static_name)+'.','']
packing=read('prompt-pack-native-plain.json');serving=read('prompt-pack-serving-fixed-plain.json')
if packing and serving:
    lines+=['## Copy-only prompt packing — validation in progress','',
        'A default-off packed-decoder option assembles one [time,batch,hidden] FP32 request buffer with pitched copies '
        'and exposes contiguous token views. Batched vision output is snapshotted before encoding the next row because the vision graph reuses its buffer. '
        'No model arithmetic, weights, precision, prompt cache or image cache changes.','',
        f'The native helper passes {len(packing["checks"])} exact-byte/legacy-copy/invalid-input/lifetime checks. '
        f'The production prompt-preparation path passes {len(serving["cases"])} cases using deliberately reused fake vision output with distinct images and ragged text. '
        'Both component suites pass normal execution plus unfiltered memcheck, racecheck and synccheck with zero findings. '
        'These are component results; broad pretrained quality and full-model race coverage are not established.','',
        'Source review caught a missing vision snapshot in the first serving integration before any pretrained timing cohort. '
        'The owned chain was interrupted, partial numerical logs retained, and the corrected chain uses separate evidence names. '
        'The CUDA device fatbin is byte-identical to the preceding RDC binary; host correctness is checked separately. '
        'Evidence: '+link('prompt-pack-device-preservation.json')+', '+link('prompt-pack-first-chain-interrupted-exit.json')+', '
        +link('prompt-pack-native-plain.json')+', '+link('prompt-pack-serving-fixed-plain.json')+'.','']
    for size,batch in (('2b',1),('4b',8)):
        suffix='-mtp3-cache128' if size=='2b' else ''
        name=f'matched512-{size}-b{batch}-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack{suffix}-candidate.json'
        d=read(name)
        if d:
            w=d['median_group_ms']
            lines+=[f'{size.upper()} B{batch}, {d["equal_32_four_way_groups"]} common equal-32 groups: '
                f'packed inputs {w["candidate"]:.2f} ms versus saved control {w["control"]:.2f} ms; '
                f'{d["same_sequence_cases"]}/28 same saved sequences. Requires paired repetition before promotion. '
                'Evidence: '+link(name)+'.','']
memory_rows=[]
for size,batch in (('2b',1),('4b',8)):
    name=f'memory-combined-{size}-b{batch}-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json'
    d=read(name)
    if d:
        values=d['phases']['steady_after']['bytes']
        memory_rows.append((size,batch,values,name))
if memory_rows:
    lines+=['## Packed-input isolated Windows memory','',
        '| Model | Batch | Warm dedicated GB | Warm shared GB |','|---|---:|---:|---:|']
    for size,batch,values,name in memory_rows:
        lines.append(f'| {size.upper()} | {batch} | {values["DedicatedUsage"]["median"]/1e9:.6f} | {values["SharedUsage"]["median"]/1e9:.6f} |')
    lines+=['','Differences from the preceding memory profile are only 2 MiB for 2B B1 and 6 MiB for 4B B8; '
        'no meaningful memory reduction is claimed. Tracked Garnet allocations remain unchanged. '
        'All six image-hash groups match the saved reference and LM Studio 4B B8 memory run. '
        'These are isolated, sampled process GPU counters, separate from timing and not proof of true peak or physical residency. '
        'Evidence: '+', '.join(link(name) for _,_,_,name in memory_rows)+', '+link('prompt-pack-isolated-memory-input-control.json')+'.','']
trace=read('profile-packed-current-2b-b1-p64-d4-graph-summary.json')
if trace:
    g=trace['groups']
    lines+=['## Current packed decoder whole-graph trace','',
        f'One instrumented 2B B1 request records {g["GRAPH_TRACE"]["count"]} graph launches, '
        f'{g["MEMCPY"]["count"]} copies and {g["MEMSET"]["count"]} memsets. '
        f'Whole-graph plus uncaptured GPU activity union is {trace["gpu_activity_union_ms"]:.3f} ms '
        f'within a {trace["gpu_activity_span_ms"]:.3f} ms span; between-activity gaps total {trace["between_activity_idle_ms"]:.3f} ms. '
        'Whole-graph tracing hides internal gaps and node costs. API/idle overlap is not causation or recoverable time. '
        'The same saved input/output and request counters had 176.22 ms historical cohort latency; '
        'the instrumented Model request is 244.35 ms. This is not an interleaved profiler-overhead control or a speed forecast. '
        'Evidence: '+link('profile-packed-current-2b-b1-p64-d4-graph-summary.json')+'.','']
view=read('logical-views-fixed-plain.json');before=read('logical-views-before-fix-exit.json')
if view and before:
    lines+=['## Logical SDK-view I/O correction','',
        'The original native implementation fails the new byte-boundary oracle: zeroing a middle token view also zeroes the following token. '
        'The original failure is retained. Zero, clone and copy now use validated contiguous logical element bytes, '
        'while preserving backing-storage owners, grouped access leases and partial-overlap rejection. '
        f'All {len(view["cases"])} boundary/lifetime cases pass normal execution and unfiltered memcheck, racecheck and synccheck. '
        'CUDA device fatbin bytes are identical to the preceding DLL; host behavior changed and is checked separately. '
        'The packing inference path reads its token views and had already passed full-state preservation; '
        'this correction protects general writable-view operations. '
        'Evidence: '+link('logical-views-before-fix-exit.json')+', '+link('logical-views-fixed-plain.json')+', '+link('logical-views-device-preservation.json')+'.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'logical-views-pretrained-{size}-b{batch}-promptpack-cross-prior-state-control.json';d=read(name)
        if d and d['all_passed']:lines+=[f'{size.upper()} B{batch} complete finite target-state hashes and outputs match the preceding packed-input build: '+link(name)+'.','']
if read('reuse-verified-head-design.json'):
    lines+=['## Reusing verified target-head work','',
        'Current source inspection corrects an earlier working-note assumption: partial packed-prefix commit calls `_read()` '
        'and repeats full target projection/argmax. The traced case has one initial head, eleven verification heads and six repeated partial-commit heads; '
        'seventeen M4 CUTLASS launches plus one M1 GEMV agree with that count. '
        'The default-off change preserves masked normalization for MTP correction and reuses authoritative accepted-token predictions. '
        'Zero-prefix fallback, accepted-row candidate equality, rollback/fault behavior, state preservation and paired timing remain required. '
        'The option has no performance acceptance. Design: '+link('reuse-verified-head-design.json')+'.','']
    exits=list(root.glob('reuse-head-verifier-b*-c*-exit.json'))+list(root.glob('reuse-head-session-b*-p*-d4-exit.json'))
    passed=sum(json.loads(path.read_text())['exit_code']==0 for path in exits)
    if exits:
        lines+=[f'Default-off implementation: {passed}/12 numerical profile exits currently pass. '
            'Verifier tests compare accepted candidate IDs with a fresh full head using the same packed shape, '
            'check evaluation/reuse counters, preserve zero-prefix fallback and retain original rollback/fault assertions. '
            'Session tests retain all 24 MTP/ordinary state and stop cases per profile. '
            'These preservation tests do not establish independent pretrained quality.','']
    for size,batch in (('2b',1),('4b',8)):
        state_name=f'reuse-head-pretrained-{size}-b{batch}-promptpack-cross-prior-state-control.json'
        state=read(state_name)
        if state and state['all_passed']:
            lines+=[f'{size.upper()} B{batch} complete finite target-state hashes and output signatures match the preceding head-off build; '
                'the probe also compares a separately ordinary target. Evidence: '+link(state_name)+'.','']
        suffix='-mtp3-cache128' if size=='2b' else ''
        name=f'matched512-{size}-b{batch}-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse{suffix}-candidate.json'
        trial=read(name)
        if trial:
            w=trial['median_group_ms']
            lines+=[f'{size.upper()} B{batch}, {trial["equal_32_four_way_groups"]} common equal-32 groups: '
                f'head reuse {w["candidate"]:.2f} ms versus saved packing-only {w["control"]:.2f} ms; '
                f'{trial["same_sequence_cases"]}/28 same saved sequences. Kept GPU allocations unchanged. '
                'This single trial requires same-build repetition; it is not a 2× or independent-quality acceptance. Evidence: '+link(name)+'.','']
        study_name=f'packed-study-{size}-b{batch}-promptpack-reusehead-summary.json'
        study=read(study_name)
        if study:
            w=study['aggregate_group_median_ms']
            lines+=[f'Same-build {size.upper()} B{batch} ABCCBA repeat: A {w["a"]:.2f}, B {w["b"]:.2f}, C {w["c"]:.2f} ms. '
                'A is native-prefix/shared-arena D4, B adds prompt packing, C adds verified-head reuse. '
                'All 20 warm groups are retained, all 28 input/prompt/output/stop signatures match across six fresh processes. '
                'Each aggregate first takes the median of two repeats for each group, then the median across groups. '
                'Two repeats do not establish significance or causality; external competitors were not rerun in this study. '
                'Derived decode throughput and phase medians are recorded with their scopes in '+link(study_name)+'.','']
legacy_name='qwen3-vl-promptpack-dll-regression-fixed-root.json';legacy=read(legacy_name)
legacy_exit=read('qwen3-vl-promptpack-dll-regression-fixed-root-exit.json')
if legacy and legacy_exit and legacy_exit['exit_code']==0:
    rows=legacy['rows']
    assert legacy['status']['ready'] and len(rows)==2 and rows[0]['text']==rows[1]['text']
    assert [row['output_tokens'] for row in rows]==[16,16]
    lines+=['## Current-DLL Qwen3-VL regression smoke','',
        'Qwen3-VL-2B-Instruct starts on the current native DLL and completes two matching 16-token requests with 77 visual tokens each. '
        'The existing regression cache folder is reused. This is a legacy serving/reset/vision smoke test, '
        'not broad independent quality, historical input-equivalence proof or matched speed acceptance. '
        'The first harness attempt passed a program file where the API requires an xModel directory; its failure is retained. '
        'The corrected argument and full startup diagnostics preserve the readiness/output assertions. '
        'Evidence: '+link(legacy_name)+', '+link('qwen3-vl-startup-harness-correction.json')+'.','']
out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines),encoding='utf-8');print(out)
