# WS-M5-04: padded coalesced BF16 prefill — rejected

Paired WorldSense/Galaxy archived-frame inference and text planning, using Qwen3.5-4B and Qwen3-1.7B on the M5. This is not video-to-persisted-event or search-retrieval latency. See protocol.json for prospective gates, cases manifests for fixed inputs, and raw paired JSON for every warmup and measured call.

The method changes contiguous-K global reads and transposes into 65-column shared tiles. It preserves BF16 weights, FP32 arithmetic order, decoder and budgets; shared memory increases by 256/128 bytes per wide tile. Full native reference tests and all 32-layer state bits pass. All 72 model-call outputs match, control counters are zero and candidate counters positive, with zero decode dispatches.

Worker 1 uses ABBA, fresh worker 2 BAAB after both modes warm. Four-B TTFT increases 16.36% / 13.70%; total increases 5.64% / 4.63%. Decode changes +1.11% / -0.67%. Both fail the declared 10% TTFT and 5% total improvement gates. No favorable rerun or promotion. Original production library restored. analysis.json contains exact medians, counts and checks.

Representative GPU prefill is 3471 ms control versus 4565 ms candidate; CPU encoding approximately 10.5 ms in both. These measurements establish the observed GPU-stage cost, not a hardware occupancy explanation. Process RSS and swap are retained but are not peak GPU allocation measurements. Small samples, one Mac and correlated fixed inputs limit generalization.

The initial duplicate-Argmax shader assembly failure is retained; it generated no model timing. Source snapshots describe the isolated prototype, not a deployed kernel. Controller Python is lifecycle automation; timed requests run through XLang3 and native WorldSense/Galaxy/Garnet. Credentials are supplied through environment variables and are not part of this archive.

Archive repair: added the previously omitted native backend/CMake/48-case test source overlay from the retained isolated tree. Raw measurements and decisions unchanged. This is an evidence packaging correction, not a new run.
