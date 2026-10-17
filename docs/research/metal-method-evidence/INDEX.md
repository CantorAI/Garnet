# Metal experiment evidence index

Generated UTC: 2026-10-17T19:14:32.415187+00:00

Base HEAD at generation: `30c39535429321d983416ffa6fd667ca18105443`. Reports indexed: **241**.

Report hashes describe the working-tree snapshot and may include changes after that base. For a paper, cite the commit containing this index together with the frozen experiment artifact.

Each row links to its original report. Read that report's scope, exclusions and status before citing it. The JSON snapshot retains complete report text, documented command blocks and hashes; the CSV supports analysis. Neither report presence nor a verification filename certifies a successful experiment.

| Experiment | Original report title |
| --- | --- |
| [results/20261013/attention-prefix](../../../results/20261013/attention-prefix/README.md) | GPU-verified contiguous prompt-slot lookup: repeated latency gains |
| [results/20261013/chunk-delta-algebra](../../../results/20261013/chunk-delta-algebra/README.md) | CPU foundation for a parallel recurrent prefill experiment |
| [results/20261013/compact-joined](../../../results/20261013/compact-joined/README.md) | Compact joined half-weight storage: lower memory, no repeated latency gain |
| [results/20261013/comparator-dtypes](../../../results/20261013/comparator-dtypes/README.md) | Actual comparator precision audit |
| [results/20261013/current-selected-operations](../../../results/20261013/current-selected-operations/README.md) | Refreshed operation attribution for the selected implementation |
| [results/20261013/decode-accumulators](../../../results/20261013/decode-accumulators/README.md) | Independent decode accumulators: component candidate, not a serving gain |
| [results/20261013/decode-occupancy](../../../results/20261013/decode-occupancy/README.md) | Decode work distribution: component evidence, not a model speed claim |
| [results/20261013/decode-single-output](../../../results/20261013/decode-single-output/README.md) | One-output SIMD decode: rejected |
| [results/20261013/decode-subsimd](../../../results/20261013/decode-subsimd/README.md) | Sixteen-lane decode groups: full-model integration rejected |
| [results/20261013/decode-unroll](../../../results/20261013/decode-unroll/README.md) | Rejected decode dot-loop unrolling |
| [results/20261013/decode-wide-outputs](../../../results/20261013/decode-wide-outputs/README.md) | Rejected wider-output decode kernels |
| [results/20261013/delta-norm-cache](../../../results/20261013/delta-norm-cache/README.md) | Rejected per-head/token norm cache |
| [results/20261013/dense-key-prefill](../../../results/20261013/dense-key-prefill/README.md) | Dense one-pass prefill: verified improvement, full goal still unmet |
| [results/20261013/dense-key-prefill/rejected-128](../../../results/20261013/dense-key-prefill/rejected-128/README.md) | rejected-128 |
| [results/20261013/direct-state-zero](../../../results/20261013/direct-state-zero/README.md) | Direct resident-state reset: repeated latency gains and a decode tradeoff |
| [results/20261013/dual-text-head](../../../results/20261013/dual-text-head/README.md) | Bounded dual-accumulator text head: rejected after real model execution |
| [results/20261013/equivalent-f32](../../../results/20261013/equivalent-f32/README.md) | Exact-value FP32 precision study; full goal still unmet |
| [results/20261013/equivalent-f32-linear-chunk](../../../results/20261013/equivalent-f32-linear-chunk/README.md) | Selected recurrent chunk build: current matched FP32 comparison |
| [results/20261013/exact-dense-tails](../../../results/20261013/exact-dense-tails/README.md) | Exact dense prefill tails: validated modest latency gains |
| [results/20261013/extended-half-scale](../../../results/20261013/extended-half-scale/README.md) | Extended exact-half QKV scales: correct reconstruction, no useful speed gain |
| [results/20261013/extended-prefill](../../../results/20261013/extended-prefill/README.md) | Extended prefill profiles: actual tests, no 2× claim |
| [results/20261013/finer-dense-tails](../../../results/20261013/finer-dense-tails/README.md) | Opt-in finer dense prefill profiles |
| [results/20261013/frame-frontend](../../../results/20261013/frame-frontend/README.md) | Fresh-frame copy elision: less CPU work, no clear model latency gain |
| [results/20261013/fused-mlp-decode](../../../results/20261013/fused-mlp-decode/README.md) | Rejected broad fused BF16 MLP decode |
| [results/20261013/half-vision](../../../results/20261013/half-vision/README.md) | Exact-scaled half vision projections: rejected |
| [results/20261013/joined-prefill](../../../results/20261013/joined-prefill/README.md) | Bounded joined MPS prefill: modest warm gain, costly first use |
| [results/20261013/linear-chunk](../../../results/20261013/linear-chunk/README.md) | Batched recurrent output projections: validated latency improvement |
| [results/20261013/linear-conv-chunk](../../../results/20261013/linear-conv-chunk/README.md) | Chunk convolution and preparation: verified VL prefill gain |
| [results/20261013/linear-delta-chunk](../../../results/20261013/linear-delta-chunk/README.md) | Persistent chunk recurrence: verified VL latency gain |
| [results/20261013/linear-direct-write](../../../results/20261013/linear-direct-write/README.md) | Recurrent token output writes: repeatable VL latency gain |
| [results/20261013/lossless-block-gpu](../../../results/20261013/lossless-block-gpu/README.md) | Whole-block BF16 packing: rejected |
| [results/20261013/mps-gemv](../../../results/20261013/mps-gemv/README.md) | Dedicated MPS vector multiply: dtype limit and rejected FP32 route |
| [results/20261013/mps-half-decode](../../../results/20261013/mps-half-decode/README.md) | Rejected mixed-input MPS decode |
| [results/20261013/mps-pair-probes](../../../results/20261013/mps-pair-probes/README.md) | Independent MPS scheduling and joined prefill projections |
| [results/20261013/mps-vision-context](../../../results/20261013/mps-vision-context/README.md) | Batched MPS vision value projection: rejected |
| [results/20261013/mps-vision-scores](../../../results/20261013/mps-vision-scores/README.md) | Optional FP32 MPS vision attention: validated modest latency gain |
| [results/20261013/native-bfloat-body](../../../results/20261013/native-bfloat-body/README.md) | Native BF16 body decode: small repeated gains, latency tradeoff |
| [results/20261013/native-frame-phases](../../../results/20261013/native-frame-phases/README.md) | Actual warm frame phase attribution |
| [results/20261013/packed-decode](../../../results/20261013/packed-decode/README.md) | Rejected interleaved BF16 decode layout |
| [results/20261013/parallel-delta](../../../results/20261013/parallel-delta/README.md) | Parallel recurrent reductions: validated and retained |
| [results/20261013/parallel-delta-gpu](../../../results/20261013/parallel-delta-gpu/README.md) | Parallel delta GPU prototypes: rejected |
| [results/20261013/qwen35-extended-prefill](../../../results/20261013/qwen35-extended-prefill/README.md) | Extended Qwen3.5 prefill: small batch-1 latency gain |
| [results/20261013/resident-kv-inplace](../../../results/20261013/resident-kv-inplace/README.md) | Private single-slot KV writes: retained opt-in text decode improvement |
| [results/20261013/resident-kv-vllm](../../../results/20261013/resident-kv-vllm/README.md) | Fresh Garnet / vLLM-Metal comparison after private KV writes |
| [results/20261013/retained-prefill-vllm](../../../results/20261013/retained-prefill-vllm/README.md) | Fresh comparison with the retained dense prefill policies |
| [results/20261013/selected-attention-vllm](../../../results/20261013/selected-attention-vllm/README.md) | Selected attention policy versus vLLM-Metal |
| [results/20261013/small-output-decode](../../../results/20261013/small-output-decode/README.md) | Bounded small-output decode: modest repeated gain |
| [results/20261013/split-half-prefill](../../../results/20261013/split-half-prefill/README.md) | Three-half activation prefill: rejected partial experiment |
| [results/20261013/state-zero-gpu](../../../results/20261013/state-zero-gpu/README.md) | GPU buffer-fill reset: rejected component variant |
| [results/20261013/transpose-half-prefill](../../../results/20261013/transpose-half-prefill/README.md) | MPS exact-half weight layout: no broad serving promotion |
| [results/20261013/vision-fused-context](../../../results/20261013/vision-fused-context/README.md) | FP32 vision softmax/value fusion: default-OFF experiment |
| [results/20261013/webp-thread](../../../results/20261013/webp-thread/README.md) | Threaded fresh WebP preparation: exact inputs and repeated latency benefit |
| [results/20261013/worldsense-forwarding-guard](../../../results/20261013/worldsense-forwarding-guard/README.md) | worldsense-forwarding-guard |
| [results/20261013/worldsense-inline-guard](../../../results/20261013/worldsense-inline-guard/README.md) | Verified direct WorldSense launcher in the benchmark guard |
| [results/20261014/compact-frame](../../../results/20261014/compact-frame/README.md) | Optional compact frame preparation |
| [results/20261014/constant-bf16](../../../results/20261014/constant-bf16/README.md) | Constant-address BF16 weights: no useful native decode gain |
| [results/20261014/decode-matrix](../../../results/20261014/decode-matrix/README.md) | FP32 matrix units at decode rows: slower than native dot kernels |
| [results/20261014/dirty-prefix-reset](../../../results/20261014/dirty-prefix-reset/README.md) | Bounded cache reset: repeated component signal |
| [results/20261014/global-half-alpha](../../../results/20261014/global-half-alpha/README.md) | Global lossless half scale through MPS alpha: trained components pass, native validation running |
| [results/20261014/global-half-alpha/vllm](../../../results/20261014/global-half-alpha/vllm/README.md) | Global-alpha bundle versus vLLM-Metal |
| [results/20261014/indirect-bf16](../../../results/20261014/indirect-bf16/README.md) | Recorded real BF16 projections: no broad speed benefit |
| [results/20261014/indirect-command](../../../results/20261014/indirect-command/README.md) | Recorded compute commands: lower CPU encoding, no broad wall-time gain |
| [results/20261014/indirect-mlp](../../../results/20261014/indirect-mlp/README.md) | Recorded four-kernel MLP: clean numerics, unstable wall-time gains |
| [results/20261014/joined-activation-decode](../../../results/20261014/joined-activation-decode/README.md) | One joined product for lossless input components: rejected |
| [results/20261014/joined-global-alpha](../../../results/20261014/joined-global-alpha/README.md) | One joined common-alpha MPS product with output views |
| [results/20261014/joined-global-mlp](../../../results/20261014/joined-global-mlp/README.md) | Optional joined global MLP native trial |
| [results/20261014/prefill-residual-norm](../../../results/20261014/prefill-residual-norm/README.md) | Packed Add/Layout/RMS fusion: small native latency benefit, no promotion |
| [results/20261014/private-bf16](../../../results/20261014/private-bf16/README.md) | GPU-private BF16 weights: compatible, no repeated broad speed gain |
| [results/20261014/residual-norm](../../../results/20261014/residual-norm/README.md) | Residual addition plus RMS normalization: no useful native decode gain |
| [results/20261014/single-coder-head](../../../results/20261014/single-coder-head/README.md) | Optional single-row coder vocabulary kernel |
| [results/20261014/single-row-bf16](../../../results/20261014/single-row-bf16/README.md) | Single-row BF16 specialization: coder-head component signal only |
| [results/20261014/vision-global-alpha](../../../results/20261014/vision-global-alpha/README.md) | Vision FP32 products versus exact common half alpha |
| [results/20261015/aligned-row-prefill](../../../results/20261015/aligned-row-prefill/README.md) | Physical MPS prefill row alignment: no broad promotion |
| [results/20261015/async-key-graph-prefill](../../../results/20261015/async-key-graph-prefill/README.md) | Asynchronous native keys plus combined graphs: repeated component signal |
| [results/20261015/body-shape-inference](../../../results/20261015/body-shape-inference/README.md) | Body-only fixed-shape serving experiment |
| [results/20261015/coalesced-activation-decode](../../../results/20261015/coalesced-activation-decode/README.md) | Coalesced mixed-input matrix decode: rejected |
| [results/20261015/coalesced-activation-prefill](../../../results/20261015/coalesced-activation-prefill/README.md) | Coalesced lossless activation MMA for prefill: rejected |
| [results/20261015/coalesced-body-block](../../../results/20261015/coalesced-body-block/README.md) | Exact coalesced BF16 block storage: repeated component signal |
| [results/20261015/coalesced-body-native](../../../results/20261015/coalesced-body-native/README.md) | Native lossless BF16 blocks: repeated full-model decode gains |
| [results/20261015/coalesced-body-vllm](../../../results/20261015/coalesced-body-vllm/README.md) | Fresh lossless-body comparison against vLLM-Metal: initial cohort audited |
| [results/20261015/coalesced-eight-body](../../../results/20261015/coalesced-eight-body/README.md) | Compact eight-output decode: initial exact-bit rejection; lexical follow-up |
| [results/20261015/combined-graph-error-diagnostic](../../../results/20261015/combined-graph-error-diagnostic/README.md) | Combined-graph numerical diagnosis: failures isolated to special native keys |
| [results/20261015/combined-graph-prefill](../../../results/20261015/combined-graph-prefill/README.md) | Combined projection graphs: component gains with rejected batch-2 profiles |
| [results/20261015/combined-prefill-reset](../../../results/20261015/combined-prefill-reset/README.md) | Combined bounded reset and ordered prefill: initial latency tradeoff |
| [results/20261015/concurrent-bf16-groups](../../../results/20261015/concurrent-bf16-groups/README.md) | Independent native projections in one compute pass: mixed results |
| [results/20261015/core-graph-prefill](../../../results/20261015/core-graph-prefill/README.md) | Core-only graph compilation: original key failures remain |
| [results/20261015/counter-trace](../../../results/20261015/counter-trace/README.md) | Compute-pass GPU timestamps: validated diagnostic, substantial overhead |
| [results/20261015/decode-eight-lane](../../../results/20261015/decode-eight-lane/README.md) | Eight-lane FP32 decode: numerically valid, slower, rejected |
| [results/20261015/decode-only-head](../../../results/20261015/decode-only-head/README.md) | Decode-only fused head: repeated full runs verified |
| [results/20261015/decode-splitk](../../../results/20261015/decode-splitk/README.md) | FP32 split-K decode: no broad gain, rejected for general dispatch |
| [results/20261015/deferred-state-zero](../../../results/20261015/deferred-state-zero/README.md) | Deferred request-state zero: repeated full-model latency tradeoff |
| [results/20261015/dependent-mlp-graph](../../../results/20261015/dependent-mlp-graph/README.md) | Complete dependent MLP graph: audited stress experiment |
| [results/20261015/direct-mps-body](../../../results/20261015/direct-mps-body/README.md) | Direct original-row mixed-storage MPS component |
| [results/20261015/eight-fma-body](../../../results/20261015/eight-fma-body/README.md) | Explicit-FMA eight-output decode: correct, rejected after repetition |
| [results/20261015/fastencode-streamed-mps-body](../../../results/20261015/fastencode-streamed-mps-body/README.md) | MPS argument-validation option: wall penalty persists |
| [results/20261015/four-threadgroup-probe](../../../results/20261015/four-threadgroup-probe/README.md) | Original four-output threadgroup tuning: audited; no setting selected |
| [results/20261015/fused-head-argmax](../../../results/20261015/fused-head-argmax/README.md) | Fused vocabulary projection and partial argmax: component gain repeats |
| [results/20261015/fused-projection-groups](../../../results/20261015/fused-projection-groups/README.md) | Single-dispatch independent projections |
| [results/20261015/hybrid-combined-graph-prefill](../../../results/20261015/hybrid-combined-graph-prefill/README.md) | Hybrid native-key / combined graph prefill: numerically valid, mixed performance |
| [results/20261015/interleaved-bf16-body](../../../results/20261015/interleaved-bf16-body/README.md) | Isolated interleaved BF16 decoder body component |
| [results/20261015/interleaved-body-native](../../../results/20261015/interleaved-body-native/README.md) | Native large-matrix BF16 body layout |
| [results/20261015/interleaved-head-native](../../../results/20261015/interleaved-head-native/README.md) | Native interleaved screening index: profile-dependent tradeoffs |
| [results/20261015/interleaved-screen-index](../../../results/20261015/interleaved-screen-index/README.md) | Lossless four-column screening index: repeated component gain |
| [results/20261015/joint-state-zero](../../../results/20261015/joint-state-zero/README.md) | Reset and dependent GPU read in one command buffer: repeated component signal |
| [results/20261015/mixed-accumulator](../../../results/20261015/mixed-accumulator/README.md) | Apple M4 mixed-input FP32 accumulator capability |
| [results/20261015/mlp-bf16-feed-primitive](../../../results/20261015/mlp-bf16-feed-primitive/README.md) | BF16 graph feeds: primitive passed; trained performance rejected |
| [results/20261015/mlp-command-buffer-native](../../../results/20261015/mlp-command-buffer-native/README.md) | Command-buffer MLP: full validation passed; native integration rejected |
| [results/20261015/mlp-command-buffer-primitive](../../../results/20261015/mlp-command-buffer-primitive/README.md) | MLP graph command-buffer encoding: 64-case private safety probe passed |
| [results/20261015/mlp-prefill-native](../../../results/20261015/mlp-prefill-native/README.md) | Native dependent MLP graph: correct outputs, rejected for latency and memory |
| [results/20261015/mps-screen-index](../../../results/20261015/mps-screen-index/README.md) | MPS-derived screening dot products: rejected |
| [results/20261015/mpsgraph-prefill](../../../results/20261015/mpsgraph-prefill/README.md) | MPSGraph FP32 prefill: actual component results, no broad promotion |
| [results/20261015/ndarray-matrix-api](../../../results/20261015/ndarray-matrix-api/README.md) | MPS NDArray API compatibility: failures, coefficient path and deferred views |
| [results/20261015/ndarray-native](../../../results/20261015/ndarray-native/README.md) | Native row80 NDArray: rejected after full inference |
| [results/20261015/ndarray-prefill](../../../results/20261015/ndarray-prefill/README.md) | Native MPS NDArray prefill investigation |
| [results/20261015/packed-body-prefill](../../../results/20261015/packed-body-prefill/README.md) | Packed BF16 prefill tiles: rejected |
| [results/20261015/padded-mps-body](../../../results/20261015/padded-mps-body/README.md) | Padded-row native MPS body projection: rejected |
| [results/20261015/parallel-vision-context](../../../results/20261015/parallel-vision-context/README.md) | Parallel FP32 vision value contraction |
| [results/20261015/prefill-stream](../../../results/20261015/prefill-stream/README.md) | Ordered partial prefill submissions: repeated latency tradeoff |
| [results/20261015/replanned-vl-prefill](../../../results/20261015/replanned-vl-prefill/README.md) | Retuned native VL B2 prefill on the same library |
| [results/20261015/reused-zero-source](../../../results/20261015/reused-zero-source/README.md) | Retained CPU zero source: repeated component signal |
| [results/20261015/reused-zero-source-inference](../../../results/20261015/reused-zero-source-inference/README.md) | Retained CPU zero source: full-model investigation |
| [results/20261015/screened-head](../../../results/20261015/screened-head/README.md) | Captured-query greedy head screening |
| [results/20261015/screened-head-inference](../../../results/20261015/screened-head-inference/README.md) | Native guarded greedy-head screening |
| [results/20261015/screened-head-vllm](../../../results/20261015/screened-head-vllm/README.md) | Screened greedy head versus fresh vLLM-Metal |
| [results/20261015/screened-prefill-reset](../../../results/20261015/screened-prefill-reset/README.md) | Screened head plus ordered prefill and partial cache reset |
| [results/20261015/screened-seed-abba](../../../results/20261015/screened-seed-abba/README.md) | Direct exact-seed comparison and native candidate |
| [results/20261015/screened-seed-sweep](../../../results/20261015/screened-seed-sweep/README.md) | Exact screened-head seed-width exploration |
| [results/20261015/serial-vision-context](../../../results/20261015/serial-vision-context/README.md) | Original-order vision context prefetch |
| [results/20261015/serial-vision-native](../../../results/20261015/serial-vision-native/README.md) | Repeated native inference with original-order vision prefetch |
| [results/20261015/shape-selected-two-native](../../../results/20261015/shape-selected-two-native/README.md) | Shape-selected two-output native integration: correct, rejected for promotion |
| [results/20261015/shape-specialized-bf16](../../../results/20261015/shape-specialized-bf16/README.md) | Full-shape BF16 decode specialization: head regressions, mixed body gains |
| [results/20261015/small-body-threshold](../../../results/20261015/small-body-threshold/README.md) | Smaller lossless body admission: rejected after confirmation |
| [results/20261015/small-seed-native](../../../results/20261015/small-seed-native/README.md) | Native exact seed 1024 versus 8192: small mixed gains |
| [results/20261015/sparse-counter-trace](../../../results/20261015/sparse-counter-trace/README.md) | Sparse GPU counters: much lower diagnostic overhead |
| [results/20261015/stage-rounded-mlp-graph](../../../results/20261015/stage-rounded-mlp-graph/README.md) | Dependent MLP with existing reference rounding: audited, six profiles excluded |
| [results/20261015/streamed-direct-mps-body](../../../results/20261015/streamed-direct-mps-body/README.md) | Streamed direct-row native MPS: rejected for integration |
| [results/20261015/striped-frame](../../../results/20261015/striped-frame/README.md) | Exact-coordinate parallel Pillow resize |
| [results/20261015/threaded-striped-frame](../../../results/20261015/threaded-striped-frame/README.md) | Combined threaded WebP decode and exact striped resize |
| [results/20261015/two-body](../../../results/20261015/two-body/README.md) | Two-output SIMD decode: repeated component signals, no model gain established |
| [results/20261015/two-matrix-probe](../../../results/20261015/two-matrix-probe/README.md) | Per-matrix two-output probe: three dispatch candidates selected |
| [results/20261015/vector-vision-context](../../../results/20261015/vector-vision-context/README.md) | Adjacent-dimension vision vectorization: bit-exact, initial calibration failed |
| [results/20261015/wide-batch-prefill](../../../results/20261015/wide-batch-prefill/README.md) | Exact 160-row VL batch-2 prefill investigation |
| [results/20261015/wide-batch-vllm](../../../results/20261015/wide-batch-vllm/README.md) | Fresh matched comparison for the validated latency bundle |
| [results/20261015/wide-body-decode](../../../results/20261015/wide-body-decode/README.md) | Eight-output BF16 body layout: no broad promotion |
| [results/20261016/4b-m5-chunk](../../../results/20261016/4b-m5-chunk/README.md) | M5 Qwen3.5-4B low-copy native kernel profile |
| [results/20261016/4b-m5-vision-shape](../../../results/20261016/4b-m5-vision-shape/README.md) | Further 4B optimization: shared vision weights and fixed body shapes |
| [results/20261016/block-packed-prefill](../../../results/20261016/block-packed-prefill/README.md) | K32 N64 block packed prefill rejected by numerical gate |
| [results/20261016/compact-fused-projection-groups](../../../results/20261016/compact-fused-projection-groups/README.md) | Compact-aware independent decode projections |
| [results/20261016/compact11-escape-cpu](../../../results/20261016/compact11-escape-cpu/README.md) | Three-bit exponent tags plus escapes: descriptive CPU study |
| [results/20261016/delta-value-tiles-addresses](../../../results/20261016/delta-value-tiles-addresses/README.md) | Wider delta value tiles CPU prerequisite |
| [results/20261016/delta-value-tiles-gpu](../../../results/20261016/delta-value-tiles-gpu/README.md) | Wider delta value tiles GPU component investigation |
| [results/20261016/delta-value-tiles-native](../../../results/20261016/delta-value-tiles-native/README.md) | Private native width64 recurrent prefill investigation |
| [results/20261016/half-triplet-lower-bound](../../../results/20261016/half-triplet-lower-bound/README.md) | Three exact half-storage activation planes: prefill feasibility |
| [results/20261016/joined-union-half-admission](../../../results/20261016/joined-union-half-admission/README.md) | Exact common-scale union: all learned feed-forward pairs |
| [results/20261016/native-image-input](../../../results/20261016/native-image-input/README.md) | Native image preparation, measured through XLang3 |
| [results/20261016/ndarray-selective-native](../../../results/20261016/ndarray-selective-native/README.md) | Selective no-exact-half NDArray: rejected after complete native inference |
| [results/20261016/ndarray-selective-prefill](../../../results/20261016/ndarray-selective-prefill/README.md) | Selective F32 NDArray: predeclared trained-component screen |
| [results/20261016/ndarray-weight-admission](../../../results/20261016/ndarray-weight-admission/README.md) | Full body-weight admission: one VL matrix needs the original FP32 fallback |
| [results/20261016/nibble-screen-index](../../../results/20261016/nibble-screen-index/README.md) | Four bit greedy head rejection index rejected |
| [results/20261016/paired-norm-rope](../../../results/20261016/paired-norm-rope/README.md) | Paired query and key normalization and RoPE |
| [results/20261016/paired-norm-rope-chain](../../../results/20261016/paired-norm-rope-chain/README.md) | Chained paired normalization and RoPE measurements |
| [results/20261016/paired-norm-rope-native](../../../results/20261016/paired-norm-rope-native/README.md) | Private native paired normalization and RoPE investigation |
| [results/20261016/prepared-half-inverse](../../../results/20261016/prepared-half-inverse/README.md) | Exact half inverse CPU verification |
| [results/20261016/prepared-qkv-native](../../../results/20261016/prepared-qkv-native/README.md) | Private native text Q/K/V fusion after repeated component selection |
| [results/20261016/prepared-union-mlp](../../../results/20261016/prepared-union-mlp/README.md) | Prepared exact union weights and activation-only output |
| [results/20261016/prepared-union-qkv](../../../results/20261016/prepared-union-qkv/README.md) | Prepared heterogeneous Q/K/V: actual-context GPU investigation |
| [results/20261016/prepared-union-qv](../../../results/20261016/prepared-union-qv/README.md) | QV fusion with the original key projection |
| [results/20261016/prepared-union-qv-views](../../../results/20261016/prepared-union-qv-views/README.md) | QV output views without a scatter dispatch |
| [results/20261016/qkv-union-half-admission](../../../results/20261016/qkv-union-half-admission/README.md) | Separate Q/K/V exact union: full learned CPU eligibility |
| [results/20261016/sixbit-screen-index](../../../results/20261016/sixbit-screen-index/README.md) | Six bit greedy head rejection index rejected by clean timing |
| [results/20261016/streamed-body-weights](../../../results/20261016/streamed-body-weights/README.md) | Streamed decoded body weights component screen |
| [results/20261016/streamed-body-weights-preparation](../../../results/20261016/streamed-body-weights-preparation/README.md) | Streamed decoded weight lifetime preparation |
| [results/20261016/vector-vision-calibrated](../../../results/20261016/vector-vision-calibrated/README.md) | Prospective null-first timing design for vector vision |
| [results/20261016/vector-vision-native](../../../results/20261016/vector-vision-native/README.md) | Ordered float4 vision context: private native integration |
| [results/20261016/wide-block-packed-prefill](../../../results/20261016/wide-block-packed-prefill/README.md) | Wide output block packed prefill rejected by timing |
| [results/20261016/worldsense-m5-cached-matrix-metadata](../../../results/20261016/worldsense-m5-cached-matrix-metadata/README.md) | WS-M5-34: cached immutable matrix metadata — resource gate rejected |
| [results/20261016/worldsense-m5-chunk64](../../../results/20261016/worldsense-m5-chunk64/README.md) | WS-M5-07: VLM-only prefill64 — rejected |
| [results/20261016/worldsense-m5-coalesced-nopad](../../../results/20261016/worldsense-m5-coalesced-nopad/README.md) | WS-M5-05: unpadded coalesced BF16 prefill — rejected |
| [results/20261016/worldsense-m5-coalesced-prefill](../../../results/20261016/worldsense-m5-coalesced-prefill/README.md) | WS-M5-04: padded coalesced BF16 prefill — rejected |
| [results/20261016/worldsense-m5-decode-sequence](../../../results/20261016/worldsense-m5-decode-sequence/README.md) | WS-M5-22 WorldSense repeated-search sequence diagnostic |
| [results/20261016/worldsense-m5-decode-step-profile](../../../results/20261016/worldsense-m5-decode-step-profile/README.md) | WS-M5-26 per-token native GPU and wait diagnostic |
| [results/20261016/worldsense-m5-decode256](../../../results/20261016/worldsense-m5-decode256/README.md) | WS-M5-11 shape-restricted eight-SIMD-group decode rejected |
| [results/20261016/worldsense-m5-dense-decode1024](../../../results/20261016/worldsense-m5-dense-decode1024/README.md) | WS-M5-29: 1024-thread companion decode grouping — rejected |
| [results/20261016/worldsense-m5-dense-decode256](../../../results/20261016/worldsense-m5-dense-decode256/README.md) | WS-M5-27 dense BF16 decode256: rejected companion throughput regression |
| [results/20261016/worldsense-m5-dense-decode512](../../../results/20261016/worldsense-m5-dense-decode512/README.md) | WS-M5-28: 512-thread companion decode grouping — rejected |
| [results/20261016/worldsense-m5-dual-chunk64](../../../results/20261016/worldsense-m5-dual-chunk64/README.md) | WS-M5-08: dual-model prefill64 — rejected after fresh confirmation |
| [results/20261016/worldsense-m5-fused-head](../../../results/20261016/worldsense-m5-fused-head/README.md) | WS-M5-01: fused greedy head through WorldSense — rejected screen |
| [results/20261016/worldsense-m5-graph-workspace](../../../results/20261016/worldsense-m5-graph-workspace/README.md) | WS-M5-19 ordinary graph workspace: rejected decode preservation |
| [results/20261016/worldsense-m5-half-prefill](../../../results/20261016/worldsense-m5-half-prefill/README.md) | WS-M5-02: limited exact-half prefill through WorldSense — rejected |
| [results/20261016/worldsense-m5-inputs](../../../results/20261016/worldsense-m5-inputs/README.md) | Fixed WorldSense M5 experiment inputs |
| [results/20261016/worldsense-m5-inventory](../../../results/20261016/worldsense-m5-inventory/README.md) | WS-M5-14 native graph and session buffer-owner inventory |
| [results/20261016/worldsense-m5-latency-trace](../../../results/20261016/worldsense-m5-latency-trace/README.md) | WS-M5-23 native and host latency tracing diagnostic |
| [results/20261016/worldsense-m5-lazy-defaults](../../../results/20261016/worldsense-m5-lazy-defaults/README.md) | WS-M5-15: lazy resident default storage (rejected screen) |
| [results/20261016/worldsense-m5-memory](../../../results/20261016/worldsense-m5-memory/README.md) | WS-M5-12 native WorldSense memory and lifetime diagnostic |
| [results/20261016/worldsense-m5-mixed-chunk](../../../results/20261016/worldsense-m5-mixed-chunk/README.md) | WS-M5-09: VLM64 / reasoner32 — rejected |
| [results/20261016/worldsense-m5-physical-memory](../../../results/20261016/worldsense-m5-physical-memory/README.md) | WS-M5-35: global ownership and OS graphics accounting — diagnostic |
| [results/20261016/worldsense-m5-prefix-cache](../../../results/20261016/worldsense-m5-prefix-cache/README.md) | WS-M5-10 native bounded text prefix cache and wider VLM prefill |
| [results/20261016/worldsense-m5-primary-blocks](../../../results/20261016/worldsense-m5-primary-blocks/README.md) | WS-M5-36: sole-primary lossless BF16 blocks — CPU feasibility |
| [results/20261016/worldsense-m5-primary-gpu](../../../results/20261016/worldsense-m5-primary-gpu/README.md) | WS-M5-37 lossless primary weights: resource pass, decode regression |
| [results/20261016/worldsense-m5-primary-kmajor](../../../results/20261016/worldsense-m5-primary-kmajor/README.md) | WS-M5-39 direct K-order decode: companion speed regression |
| [results/20261016/worldsense-m5-primary-u32](../../../results/20261016/worldsense-m5-primary-u32/README.md) | WS-M5-40 bounded uint32 primary weights: repeated prototype confirmation |
| [results/20261016/worldsense-m5-primary-vector](../../../results/20261016/worldsense-m5-primary-vector/README.md) | WS-M5-38 vector primary weights: companion decode regression |
| [results/20261016/worldsense-m5-private-execution](../../../results/20261016/worldsense-m5-private-execution/README.md) | WS-M5-25 actual private execution: rejected companion decode regression |
| [results/20261016/worldsense-m5-private-resident-defaults](../../../results/20261016/worldsense-m5-private-resident-defaults/README.md) | WS-M5-30: private resident deferred defaults — rejected |
| [results/20261016/worldsense-m5-private-resident-dense512](../../../results/20261016/worldsense-m5-private-resident-dense512/README.md) | WS-M5-31: private resident defaults plus512 grouping — rejected |
| [results/20261016/worldsense-m5-request-pool](../../../results/20261016/worldsense-m5-request-pool/README.md) | WS-M5-13 native request pool memory gate failed |
| [results/20261016/worldsense-m5-request-residency](../../../results/20261016/worldsense-m5-request-residency/README.md) | WS-M5-33: request-scoped bounded weight residency — rejected |
| [results/20261016/worldsense-m5-resident-admission](../../../results/20261016/worldsense-m5-resident-admission/README.md) | WS-M5-17: explicit resident build admission (rejected physical-memory feasibility) |
| [results/20261016/worldsense-m5-resident-workspace](../../../results/20261016/worldsense-m5-resident-workspace/README.md) | WS-M5-18: active resident scratch pools (rejected physical-memory gate) |
| [results/20261016/worldsense-m5-shared-half](../../../results/20261016/worldsense-m5-shared-half/README.md) | WS-M5-03: immutable shared exact-half body storage — rejected |
| [results/20261016/worldsense-m5-small-prefill](../../../results/20261016/worldsense-m5-small-prefill/README.md) | WS-M5-06: existing smaller BF16 prefill tile — rejected |
| [results/20261016/worldsense-m5-storage-scopes](../../../results/20261016/worldsense-m5-storage-scopes/README.md) | WS-M5-16: graph/session storage lifetime pools (rejected screen) |
| [results/20261016/worldsense-m5-vision-weight-stream](../../../results/20261016/worldsense-m5-vision-weight-stream/README.md) | WS-M5-24 exact vision-weight materialization: rejected companion decode regression |
| [results/20261016/worldsense-m5-vision-workspace](../../../results/20261016/worldsense-m5-vision-workspace/README.md) | WS-M5-21 scalar vision workspace: rejected companion decode preservation |
| [results/20261016/worldsense-m5-vlm-workspace](../../../results/20261016/worldsense-m5-vlm-workspace/README.md) | WS-M5-20 private dense buffers with VLM reuse: rejected resource gate |
| [results/20261016/worldsense-m5-weight-residency](../../../results/20261016/worldsense-m5-weight-residency/README.md) | WS-M5-32: bounded BF16 weight residency preparation — rejected |
| [results/20261017/worldsense-m5-direct-float-upload](../../../results/20261017/worldsense-m5-direct-float-upload/README.md) | WS-M5-50 — Direct FP32 weight upload rejected on text decode preservation |
| [results/20261017/worldsense-m5-duo-prefill](../../../results/20261017/worldsense-m5-duo-prefill/README.md) | WS-M5-49 — Minimal program and duo prefill rejected on memory preservation |
| [results/20261017/worldsense-m5-isolated-vector-prefill](../../../results/20261017/worldsense-m5-isolated-vector-prefill/README.md) | WS-M5-48 — Isolated vector prefill library rejected by matched timing |
| [results/20261017/worldsense-m5-primary-lazy](../../../results/20261017/worldsense-m5-primary-lazy/README.md) | WS-M5-45 — Joint primary weights and resident-default retirement rejected |
| [results/20261017/worldsense-m5-primary-narrow-attention](../../../results/20261017/worldsense-m5-primary-narrow-attention/README.md) | WS-M5-42 raw narrow attention with packed MLP: rejected |
| [results/20261017/worldsense-m5-primary-narrow-vocab](../../../results/20261017/worldsense-m5-primary-narrow-vocab/README.md) | WS-M5-41 semantic raw narrow vocabulary: rejected |
| [results/20261017/worldsense-m5-primary-u32-integration](../../../results/20261017/worldsense-m5-primary-u32-integration/README.md) | WS-M5-40 canonical integration: production acceptance rejected |
| [results/20261017/worldsense-m5-primary-value16](../../../results/20261017/worldsense-m5-primary-value16/README.md) | WS-M5-43 value16 sign/mantissa fold: rejected |
| [results/20261017/worldsense-m5-vector-prefill](../../../results/20261017/worldsense-m5-vector-prefill/README.md) | WS-M5-47 — Exact vector prefill loading rejected by matched timing |
| [results/20261017/worldsense-m5-wide-primary-lazy](../../../results/20261017/worldsense-m5-wide-primary-lazy/README.md) | WS-M5-46 — Wide primary weights with global default retirement rejected |
| [results/20261017/worldsense-m5-wide-storage](../../../results/20261017/worldsense-m5-wide-storage/README.md) | WS-M5-44 — Wide-checkpoint storage rejected by text decoding |
| [test2026/metal/results/20261012/vllm-metal](../../../test2026/metal/results/20261012/vllm-metal/README.md) | vLLM-Metal comparison evidence |
| [test2026/metal/results/20261012/vllm-metal/lossless-gpu](../../../test2026/metal/results/20261012/vllm-metal/lossless-gpu/README.md) | Exact GPU packing experiments |
| [test2026/metal/results/20261012/vllm-metal/partial-head](../../../test2026/metal/results/20261012/vllm-metal/partial-head/README.md) | Partial output-head bounds |
| [test2026/metal/results/20261012/vllm-metal/prefill-tiles](../../../test2026/metal/results/20261012/vllm-metal/prefill-tiles/README.md) | Validated wide prefill tiles |
| [test2026/metal/results/20261012/vllm-metal/pruned-head-gpu](../../../test2026/metal/results/20261012/vllm-metal/pruned-head-gpu/README.md) | Measured native GPU partial-head prototypes |
| [test2026/metal/results/20261013/decode-input-tiles](../../../test2026/metal/results/20261013/decode-input-tiles/README.md) | Rejected BF16 decode input tiles |
| [test2026/metal/results/20261013/decoder-codec](../../../test2026/metal/results/20261013/decoder-codec/README.md) | Rejected lossless decoder packing |
| [test2026/metal/results/20261013/delta-split](../../../test2026/metal/results/20261013/delta-split/README.md) | Validated direct delta state outputs |
| [test2026/metal/results/20261013/encoder-reuse](../../../test2026/metal/results/20261013/encoder-reuse/README.md) | Rejected compute encoder reuse |
| [test2026/metal/results/20261013/exact-half-prefill](../../../test2026/metal/results/20261013/exact-half-prefill/README.md) | Validated exact scaled FP16 MPS prefill |
| [test2026/metal/results/20261013/exact-half-probe](../../../test2026/metal/results/20261013/exact-half-probe/README.md) | Exact scaled FP16 storage feasibility |
| [test2026/metal/results/20261013/half-body-decode](../../../test2026/metal/results/20261013/half-body-decode/README.md) | Rejected exact-scaled FP16 body decode |
| [test2026/metal/results/20261013/kv-chunk-vector](../../../test2026/metal/results/20261013/kv-chunk-vector/README.md) | Validated vector chunk KV-cache updates |
| [test2026/metal/results/20261013/vector-slices](../../../test2026/metal/results/20261013/vector-slices/README.md) | Validated vector state slices |
