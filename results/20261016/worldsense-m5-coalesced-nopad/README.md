# WS-M5-05: unpadded coalesced BF16 prefill — rejected

Prospective single-car current-event screen through WorldSense/Galaxy, preserving original shared stride 64 while coalescing global reads. See protocol.json for gates declared before timing. Both modes warmed, two ABBA blocks, four measured samples per policy/model. Two models per frame: Qwen3.5-4B and Qwen3-1.7B. All 20 model outputs match, actual routes validate, and correctness-only native/32-layer state-bit tests pass. No direct-model timing used.

4B median TTFT increases 35.47%, total increases 16.55%, decode decreases 2.95%. Text TTFT increases 12.14%, total increases 6.60%. Skill native-call median increases from 15.226 to 17.569 seconds. Failed prospective gates; rejected without full confirmation or unchanged rerun. Original production library and live assignments restored. No optimization promoted.

These are warm archived-frame model/Skill measurements, excluding video decode, scheduling, persistence and retrieval. Small sample and single frame limit generalization. RSS/swap are not peak GPU memory. Full raw calls include warmups; analysis.json preserves metrics, exact checks and route counters. source/ contains isolated prototype code, not deployed production kernels. Controller Python performs lifecycle automation; timed HTTP requests use XLang3. Native tests provide correctness evidence only.
