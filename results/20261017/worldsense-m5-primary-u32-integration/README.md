# WS-M5-40 canonical integration: production acceptance rejected

Canonical source builds and correctness checks passed. The additional fresh comparison against the actual deployed WS10 binary rejected promotion: companion text decode regressed 3.186845% in pair 1, beyond the unchanged 2% preservation bound. Pair 2 passed; it cannot rescue pair 1. WS10 normal service was restored, with four file sources online, no camera and benchmark POST returning 404.

## Method and identities

Canonical code commit `f21c1a258`, frozen 1,715-file source closure. Builds OFF/ON/fault passed seven native tests each; 186 complete projection records matched byte for byte. Both GPU readers reconstructed all 398 packed learned matrices; full 32-step VLM and 28-step text state/token, prefix, vision, checkpoint and failure checks passed. CUDA sources and original FP32 math remain unchanged. The optional diagnostic inventory path is portable; no external CPython image processing or inference.

The predeclared actual-production order was control-1 / candidate-1 / candidate-2 / control-2. Each fresh worker used three fixed video-derived frames and a query, repeated first/warm: 48 model calls in total. Every complete output, prompt, effective config, image, Skill and query-plan reference matched. Compiler/SDK/native-host closure (1,864 files) and both source closures stayed unchanged. No concurrent experiment or archive copy during timing.

## Actual deployed-control comparison

Warm medians remain per worker/pair; all first and warm calls are retained. Percent changes below are calculated within the corresponding pair.

| Pair | Model | TTFT change | Total change | Derived decode change |
| --- | --- | ---: | ---: | ---: |
| 1 | Qwen3-1.7B | -35.420636% | -32.578479% | -3.186845% |
| 1 | Qwen3.5-4B | -4.419074% | -16.843845% | +22.440778% |
| 2 | Qwen3-1.7B | -43.784559% | -38.498590% | +28.656336% |
| 2 | Qwen3.5-4B | -8.397766% | -17.923743% | +16.473803% |

Pair 1: native Skill wall median 35.410189417 → 19.548730166 seconds (-44.793489%). Result: REJECT.
Pair 2: native Skill wall median 35.891538042 → 19.642010958 seconds (-45.273978%). Result: PASS.

## Scope and decision

The actual WS10 control predates the new memory API: its telemetry is unavailable, not zero. Candidate sampled Metal bytes are diagnostic; no memory percentage is fabricated for this cohort. The separate prototype resource result (Metal -12.84%, sampled process footprint -13.42%) and prototype confirmations retain their original control scope and do not override production rejection.

Derived decode is `(generated_tokens-1)/(total-TTFT)`, including host/stop/post-loop overhead and possible extra EOS work; it is not isolated GPU throughput. Profile GPU and wait spans overlap. Setup/first costs stay separate. No causal head/ALU/bandwidth attribution or cross-cohort speed ratio is claimed. These fixed-frame tests exclude live video scheduling/event commits/retrieval/ranking. The original M4 mini / six-profile vLLM target remains separate and unmet.

This tested candidate is integrated behind an opt-in build profile, but is **not selected for normal service**. No promotion or live candidate smoke was attempted after rejection. Preserve all failed evidence and keep the gate. A distinct selective text-storage hypothesis requires a new design, correctness/resource/speed gates and fresh confirmation.

Archive includes source reconstruction against an immutable Git base, full native projection words, raw calls, independent metric/output/coverage verifier and source/build/model/runtime dependencies. Absolute fixture paths may require regeneration. The broader historical documentation checker is separate from these experiment checks.
