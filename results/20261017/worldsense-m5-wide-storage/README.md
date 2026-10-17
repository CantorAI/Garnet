# WS-M5-44 — Wide-checkpoint storage rejected by text decoding

The candidate reduced memory allocation and passed full correctness, but the companion text model decoded 14.02% slower. It failed the unchanged 2% preservation gate and was not promoted. The validated WS10 configuration was restored with four file sources online, no camera, and the benchmark endpoint disabled (valid request returns HTTP 404).

## Method declared before implementation

Start from immutable canonical `f21c1a2584031ee9d1c129d1e58818f9d1c826d2`, not a rejected mixed-storage or value16 variant. An eight-file overlay adds the default-OFF Metal-only `GARNET_METAL_WIDE_CHECKPOINT_STORAGE` flag, which requires primary BF16 blocks and lazy resident defaults. It classifies checkpoint metadata once: consistent positive 2D embedding/head matrices with width greater than 2048 qualify; absent, malformed or conflicting metadata fails closed. Qwen3.5-4B qualifies at width 2560; Qwen3-1.7B remains raw and eager at width 2048.

Only eligible graphs use existing lossless primary weights and existing safe retirement of unused default buffers. Narrow graphs return before retirement planning. Cross-scope BuildShared requests fail before GPU allocation; same-scope ownership remains intact. Shader, CPU codec, FP32 arithmetic/reductions, native compute, Model, serving wrapper, profiles, budgets, cache policy and CUDA were unchanged. No per-token architecture lookup or CPython inference/image path was introduced.

## Correctness and identities

OFF, ON and fault builds exited zero; all seven native tests per build passed. All 186 complete original projection records remain byte-identical under the original FP64 bounds. Both GPU readers checked 4,200,857,600 original FP32 words each across 201 packed VLM matrices. The 245 raw buffers include all 197 large text matrices and 48 original raw fallbacks; they are not claimed as primary-reader bit coverage. Retained weight payloads plus raw buffers total 9,850,930,176 bytes.

Full 32-layer VLM and 28-layer text states/tokens, trained layers, vision/grid, prefix/capacity, host/private/shared close/rebuild, invalid-input and original fault checks passed. Actual API VLM primary ownership and retirement are positive; text primary ownership and retirement remain zero. Generic/resident VLM alternation rehydrates correctly; invalid cross-scope sharing preserves the parent state. The old dead lazy-fault branch remains dead and is not claimed as executed. The private source closure contains 1716 files; all 1865 build/SDK/host dependency identities remained unchanged through both timing stages.

## Fresh WorldSense measurements

The resource stage used eight actual WorldSense Skill/Galaxy calls over fresh OFF/ON workers. Sampled maximum post-decode Metal allocation fell from 21,903,376,384 to 18,754,076,672 bytes (**14.378147%**). Sampled process footprint fell from 18,053,462,296 to 16,647,747,744 bytes (**7.786399%**). These are snapshots, not peaks or free RAM. Resource timings do not establish speed.

The separate speed stage used twenty calls, five single-car requests per fresh worker, with first results retained and four warm samples per model. All complete outputs, tokens, prompts, effective configurations, frame/Skill identities and interpreted semantics matched the frozen oracle. Warm medians:

| Model | TTFT seconds control / candidate | Total seconds control / candidate | Derived tokens/s control / candidate | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-1.7B | 2.415376 / 1.841711 | 3.744757 / 3.430095 | 25.803965 / 22.185747 | -14.021949% |
| Qwen3.5-4B | 12.864069 / 6.624684 | 21.734369 / 15.031475 | 10.487633 / 10.739647 | +2.402964% |

Native Skill request wall median changed from 25.133104958 to 18.521131334 seconds (-26.307826%). Text TTFT changed -23.750552% and total -8.402733%; VLM TTFT changed -48.502418% and total -30.840068%.

## Decision and limits

**REJECT.** Text derived decoding regressed beyond 2%, despite favorable VLM, latency, memory and complete Skill wall results. No additional actual-WS10 screen, full mirrored confirmation, canonical integration or production promotion was dispatched. No unchanged retry. All 28 resource/speed calls and 14 unique parent requests are retained. Parent Skill wall is counted once per frame, not once per model phase.

Derived decoding is `(generated_tokens - 1)/(total - TTFT)` and includes host/stop/post-loop/possible extra EOS work. Native owner/encoded counters do not prove physical residency or completed GPU operations; GPU/wait spans can overlap. The control shows substantial within-worker TTFT drift, which remains visible in the raw first/warm rows; no favorable subset is selected and no causal kernel claim follows. Changing storage only for the VLM does not guarantee isolated companion performance on a shared device.

Fixed video-derived frames traverse the real WorldSense current Skill → Galaxy → native Garnet path. They exclude live decoding, scheduling, event persistence and search retrieval/ranking. The second car file is a correlated control, not independent multi-camera fusion. This M5/24 GiB/4B experiment is separate from the unmet M4 Mac mini/vLLM goal. No competing investigation GPU job, build, source/driver mutation or archive copy ran during timing. All first/warm results, numerical fixtures, native word binaries, source/build/model/input/runtime/SDK identities, failed decision and restoration evidence are retained; fixture absolute paths may require remapping.

The archive builder's stale WS43 protocol/report guard was corrected before any archive copy. It now asserts WS44 and derives the destination from the work name. No benchmark input, source or measured result changed.
