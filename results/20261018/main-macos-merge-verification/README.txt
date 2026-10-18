Garnet main integration and Mac verification — 2026-10-18

Outcome: latest Metal and already-included Pi5B work merged into main. Mac Metal native tests, model state/prefix guards, image preparation, and actual WorldSense -> Galaxy -> Garnet checks passed. The verified main build is deployed as the normal launchd Cantor service with four file sources online, camera unmounted, and the benchmark endpoint disabled (valid POST returns HTTP404).

Validated main tree: 2dc886e918e67f340eaab7ccf6bcb37d8c145b23
Latest Metal included: 0e80c8820acac641827ab5ad8296916e6cf51718
Previous remote main: bb5870fc5a20d5a8b32a9f1159a5677c65b8c134
The new Metal commit preserves 25 Pi investigation artifacts; no runtime sources changed in that resync. Pi5B history includes b34bbce26, 4b96e1bf8, c349ad7ba, 74141b61c and d4c8ef0e8. The user confirmed Pi5B was already included.

Integration fix: c441ef2310a628882ca2a5a3e5e66ffd91fb123a excludes Metal entry/backend and separately linked native-image sources from the CUDA target's recursive source globs. Explicit Metal/CPU-only branches retain their own selections. A source-selection CMake check passed; CUDA and Pi hardware were not executed on this Mac.

Build: Release, full Xcode toolchain, CMake MetalWorldSense4BPrefix preset, native XLang3 SDK/runtime. Source is the main checkout /Users/shawn/CantorAI/Garnet, not an archived optimization snapshot. All common profile flags equal the validated WS10 build; additional experimental flags are OFF. Rejected optimization experiments were not enabled. The previous optimization goal remains paused.
Build directory: /Users/shawn/CantorAI/out/build/garnet-main-macos-verify-20261018
Deployed libgarnet SHA256: 6cb8ea9b3d196398dac22e4d90b059d07919e659e2b00436570b75b305fa2f5d
Launchd/LRPC verified PID: 98337

Correctness:
- Native CTest: 7/7 passed.
- Complete native projection stream: all 3,662,272 bytes equal frozen validated output, SHA256 2712ac0dab8babc0058e1c11be00405a9929ebb359d17fdf4ed3ab420d86c275.
- Native C++ image preparation: 24 fixtures exactly equal full FP32 reference bytes/grids. Corrupt JPEG, truncated JPEG, and zero normalization standard deviation all rejected.
- Qwen3.5-4B: complete 32-layer state bytes and generated tokens equal between chunk16/chunk64 and independently equal frozen WS10 results. Finite states, capacity rejection/recovery, and visual prefix rejection passed.
- Qwen3-1.7B: complete 28-layer state bytes/tokens equal uncached and cached results and frozen WS10, including changing suffixes and 64-token warm restores.
- Prefix safety: all 32 original checks match frozen evidence, including eviction, invalid handles, ownership/generation, nonfinite state rejection, clear/reset/rebuild/recovery.
- 326 consumed source, binary, runner, reference and model identities frozen before testing and unchanged afterward. Model execution and native image preparation used XLang3/C++; CPython only orchestrated jobs and audited JSON.

Actual WorldSense verification:
- 3 recorded frames, 9 Skill evaluations, 9 VLM calls + 6 Skill LLM calls + 3 search-planning LLM calls = 18 native model calls.
- Vehicle-positive, elephant-negative and bunny search/indexing cases.
- Complete generated token lists, decoded text, prompt hashes, configurations, Skill/frame hashes and event/search results match frozen WS10 candidate references. Search planning returns passenger-car terms through the LLM route.
- Every call reports the main checkout serving source; VLM reports native-cpp image preparation. Experimental primary encoded deltas remain zero. Steady Skill LLM calls restore 64 prefix tokens. The benchmark restores assignments and does not commit world states/events; normal file-source assignments were then resumed and checked online.
- Fixed-frame suite HTTP wall: 422.246 seconds, including warmups and all calls.

Descriptive steady medians (6 VLM calls, 4 Skill LLM calls; search calls excluded):
- Qwen3.5-4B: total 19.621s; TTFT 6.566s; decode 9.00 tokens/s.
- Qwen3-1.7B: total 8.466s; TTFT 7.121s; decode 23.11 tokens/s.
Decode rate is median per-call (generated_tokens - 1)/(total_seconds - ttft_seconds), not pure GPU throughput. This is an integration verification cohort, not a paired speedup experiment; no optimization or performance improvement is claimed.

Evidence: raw WorldSense JSON, full state/token fingerprints, native stream, frozen input identities, CMake/profile checks, runner snapshots and deployment checks are preserved beside this report. Runners retain original absolute workstation paths and were executed from the documented work directory; private authentication helper content and model weights are excluded. The initial offline audit expected optional event fields on the search-only case; it was corrected to compare optional fields when present on either result. Runtime execution and golden values were unchanged.

Publication is a following evidence-only commit on main; its parent tree above is the validated runtime. The final remote commit is recorded in the local user-facing publication report after the push.
