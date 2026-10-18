from pathlib import Path
import datetime,json,hashlib,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-primary-simd-metadata';s=r/'out/research/worldsense-cooperative-prefill/source';old=b/'work/worldsense-cooperative-prefill';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(b/'work/worldsense-opt/ws67-archive-independent-audit.json')['decision']=='valid-reject'
seed=load(old/'correctness-source-manifest.json');assert len(seed)==1724 and all(sha(s/k)==v for k,v in seed.items())
index=load(r/'Garnet/docs/research/metal-method-evidence/index.json');assert len(index['reports'])==258
prior=[]
for row in index['reports']:
 text=row['report_text'];prior.append(dict(path=row['report_path'],sha256=row['report_sha256'],simd_metadata_mention=bool(__import__('re').search(r'(simd.{0,40}metadata|metadata.{0,40}broadcast|broadcast.{0,40}metadata)',text,__import__('re').I))))
shader=(s/'src/backends/metal/primary_block_shader.h').read_text();assert shader.count('PrimaryQuad quad=primary_quad(payload,metadata,column,k,')==2
assert 'simd_broadcast_first'not in shader and 'simd_shuffle'not in shader
corpus=load(old/'corpus-GPU-bits.json');matrices=[i for i in corpus['matrices']if i['packed']];assert len(matrices)==201 and sum(i['rows']*i['inner']for i in matrices)==4200857600
for row in matrices:assert row['rows']%4==0 and row['inner']%128==0 and row['rows']*row['inner']*2<=2**31
checks=[]
for inner in sorted({i['inner']for i in matrices}|{128,256}):
 for row in [0,4,16,260]:
  for block in range(inner//128):
   addresses=[(row//4)*(inner//128)+(lane*4+128*block)//128 for lane in range(32)]
   assert len(set(addresses))==1
 checks.append(dict(inner=inner,all_K_blocks_and_32_decode_lanes_uniform=True))
for n,k,threads in [(32,32,128),(64,64,256),(64,32,256)]:
 for columns in [260,264,272,288,320,1024,6144]:
  for colBase in sorted({0,(columns//n)*n,((columns-1)//n)*n}):
   for inner in [128,256]:
    for base in range(0,inner,k):
     for offset in range(0,k*n,threads):
      for tid in range(threads):
       i=tid+offset;kk=base+i//n;col=colBase+i%n;sourceTid=tid&~3;si=sourceTid+offset;sourceK=base+si//n;sourceColumn=colBase+si%n
       if col<columns and kk<inner:
        assert sourceColumn<columns and sourceK<inner
        assert (col//4,kk//128)==(sourceColumn//4,sourceK//128)
 checks.append(dict(tileN=n,tileK=k,threads=threads,all_threads_iterations_and_full_quad_tail_sources_valid=True))
metrics={}
for label in ['control-1','candidate-1']:
 data=load(old/'primary-speed'/(label+'.json'));rows=[]
 for item in data['items']:
  if item['warmup']:continue
  c=next(c for c in item['result']['model_response'].values()if isinstance(c,dict)and c.get('model_id')=='Qwen3.5-4B');t=c['timings'];rows.append(dict(repetition=item['repetition'],ttft=c['ttft_seconds'],total=c['total_seconds'],prefill_seconds=t['prefill_seconds'],decode_seconds=t['decode_seconds'],decode_gpu_ms=t['decode_profile']['gpu_ms'],decode_encode_ms=t['decode_profile']['encode_ms'],decode_prepare_ms=t['decode_profile']['prepare_ms'],prefill_gpu_ms=t['prefill_profile']['gpu_ms'],prefill_encode_ms=t['prefill_profile']['encode_ms']))
 metrics[label]=rows
proof=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Read-only address/control-flow proof, no source/build/model launch or speed eligibility.',seed_source=str(s),seed_commit='a26bf3059642478dcecff9b0687a0c9ac25959d3',seed_manifest=seed,prior_reports=prior,prior_explicit_metadata_broadcast_mentions=sum(i['simd_metadata_mention']for i in prior),packed_matrices=201,original_words=4200857600,checks=checks,consumers=dict(decode=['primary_bf16_decode','primary_specialized_bf16_decode'],prefill=['primary_bf16_gemm','primary_gemm_16_64_64','primary_gemm_32_64_32'],unchanged=['primary_word','primary_quad','primary_expand_words','primary_expand_vector_words','all original vector-prefill kernels and readers']),retained_WorldSense_primary_phase_calls=metrics,limits='Metadata address uniformity is not compiler/ISA/cache-line/load-transaction/bandwidth/occupancy proof; phase spans overlap and are not causal. No prior rejection overturned.')
(w/'read-only-metadata-consumer-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
design='''WS-M5-68 prospective design: SIMD sharing of lossless primary block metadata

Hypothesis and motivation
Retained actual WorldSense/Galaxy calls spend most decode time in GPU-spanning native work, while CPU encoder/preparation spans are much smaller. That does not isolate a projection, metadata cost or hardware cause. Source establishes that all 32 decode lanes in one SIMD group read the same original 128-K/four-output-row metadata value in every iteration. The ordinary primary prefill scalar tile loader has groups of four neighboring lanes with identical metadata addresses. Hypothesis: load one metadata value per proven group and share it through SIMD shuffles, retaining original payload loads and all floating arithmetic. Compiler/hardware may already coalesce those reads; branch/shuffle costs can outweigh any gain. No gain is established yet.

Prior evidence and scope
WS67 is a final valid rejection at mandatory matchedPRIMARY20, despite a favorable earlier actualWS10 screen. Do not retry it unchanged. Reviewed all258 indexed reports and current native source: prior cached matrix metadata was a HOST object cache, not this device block-metadata sharing. WS43 changed value reconstruction, WS47/48/49 vectorized prefill payload reads and changed library policy, WS63-67 changed text weight layouts/decoding/prefill. This distinct candidate uses the original PRIMARY-only profile, adds ONLY new metadata-sharing selection and does not enable those rejected later profiles. Source seed is published67/a26bf3059642478dcecff9b0687a0c9ac25959d3,1724files. Existing optional implementations remain available default-OFF and byte-preserved. This profile selection is prospective, not a posthoc rescue of67. No M4mini/vLLM claim.

Implementation contract
Default-OFF GARNET_METAL_PRIMARY_SIMD_METADATA requires primary block weights and disallows vector-prefill/full-isolated-library selection for this first ordinary-primary implementation. RAW and minimalPRIMARY remain exact. New separate generated shader string retains original primary_word/quad and all original expand/readback functions and adds helpers accepting an already fetched metadata uint. Decode emits one lane0 metadata load followed by simd_broadcast_first, then exact original PrimaryQuad payload conversion. Prefill emits one load per four-lane group followed by simd_shuffle from lane&~3, then exact original scalar primary_word conversion. Do not move FP32 accumulation, alter K order, reductions, multiply/add grouping, row masks, barrier geometry, payload layout, precision, cache/output limits or ownership. Uniform source-lane activity and admitted columns%4/inner%128 are mandatory; never shuffle from an inactive source. Primary32 admission still proves bounded32-bit addresses and maximum rawbytes. Existing payload/metadata byte counts unchanged, no private master/scratch/cache or new device queue/wait.

Full correctness proof
Before inference compare old/new original source by explicit inverse transformations and record actual constructor program strings. Original complete186projection records and original math/FP64 bounds must pass across RAW,PRIMARY,newON,newFAULT,old67-compatibility roles. Existing native allocation/ownership/alias/Abort/error tests and configuration guards remain. Add correctness-only readback kernels/API for BOTH exact new serving helpers. Decode reader uses32lanes per512-word block, four outputs/four K words per lane. Prefill reader maps four lane groups to original four-output metadata blocks. Full actual201packed matrices/five existing GPU readers each4,200,857,600words stay required; each NEW helper must separately reconstruct every original4,200,857,600FP32word. Arbitrary bounded readback ranges, tails, invalid flags and source-owner accounting also tested. Keep245raw HOST words,546direct upload words, complete original32VLM28text states/tokens, vision, prefix/capacity, private/shared/rebuild and actual API oracles. Compatibility67 remains correct and is not timed. Companion1.7B stays ordinary RAW weights with the same exact native route and no external helper. Guard adaptations may alter feature admission/roles only; no numeric/golden/metric threshold weakening.

Prospective build/profile roles
RAW: all method flagsOFF. PRIMARY: PRIMARY/WIDE/LAZY ON, all later vector/isolation/direct/cached/sole/dual/cooperative flagsOFF. ON: same PRIMARY plus new SIMDmetadata. FAULT: ON plus existing hooks. COMPAT67: original67 full profile/new flagOFF, preserving original optional correctness. Original WS10 serving budgets VLM64,text16,capacity2048,image65536,textprefix64tokens/twoentries/64MiB. Images freshly prepared/encoded in native C++/Metal/XLang3, no answer/image reuse or speculation. CPython orchestration/audits only. No camera. CUDA sources/build support and XLang3 engine remain unchanged.

Actual WorldSense performance gates and sequencing
Freeze consumed source/runtime/libraries/models/inputs/Skill/drivers/compiler-SDK-host closure and prospective method before model launch. Four actual functional calls first, timing-ineligible. Then resource8: >=10%Metal and>=5%process-footprint reduction versus freshRAW; these are postdecode snapshots, not peaks. Actual deployedWS10 screen20 must preserve each model TTFT,total,deriveddecode and wholeSkillwall within2%. MatchedPRIMARY20 must improve4BTTFT>=5% AND preserve each model/wall/Metal/process footprint within2%; it is now an isolated option contrast but hardware causality remains unproven. RAW20 then full48mirrored, each original pair independently passes all gates; no pooled rescue. All original inputs, first/warm splits, counts/order/golden checks and thresholds retained. Valid failure stops later stages/no unchanged retry. Only full passes permit canonical integration/rebuild/fullproof+48actualWS10 and app smoke before promotion. Keep WS10 until that process succeeds. No builds, consumed edits, archive copying or competing GPU work during timing; bounded observations only.

Reporting
Retain initial/warm/all failures and complete outputs, exact learned input/model/build/source identities, complete call-level timing and source-route evidence, setup/compiler/shuffle costs and no-gain possibilities. Deriveddecode includes host/stop/postloop/possibleEOS/snapshots; not pureGPU. Retained per-call phase observations motivate this hypothesis, not an isolated measured metadata bottleneck. Fixed-frame Skill tests exclude live decoding/scheduling/event persistence/sharedstates/retrieval/ranking/navigation/independentcamera fusion. Record these limits. Design precedes all new source/build/model work. Full optimization remains active; no global-optimum claim.

API reference: Apple Metal Shading Language Specification, https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf . Official downloaded bytes frozen locally; inspect full active-lane semantics before implementation. Address proof alone does not substitute for that API contract or actual native tests.
'''
(w/'native-design-before-implementation.txt').write_text(design);(b/'outputs/Garnet-4B-WS68-Design.txt').write_text(design)
(w/'protocol.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),experiment='WS-M5-68',design_sha256=sha(w/'native-design-before-implementation.txt'),audit_sha256=sha(w/'read-only-metadata-consumer-audit.json'),source_implemented=False,no_build_or_model_dispatched=True,promotion=False,prior67='valid-reject'),indent=2)+'\n')
print(json.dumps(dict(packed_matrices=201,original_words=4200857600,proof_cases=len(checks),prior_reports=258,prior_explicit_metadata_broadcast_mentions=proof['prior_explicit_metadata_broadcast_mentions'],design_sha256=sha(w/'native-design-before-implementation.txt'))))
