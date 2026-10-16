# Garnet Metal optimization handoff prompt

Copy the prompt below into Codex on your MacBook. The MacBook is the development console; the benchmark target remains the existing M4 Mac mini. Transfer the accompanying `Garnet-Metal-MacBook-Handoff.zip` to the MacBook and extract it before starting. It includes this prompt, the draft comparison scripts and the selected evidence. It does not include model weights, credentials or the complete repository.

## Prompt to paste into Codex

Continue my Garnet native Metal/MPS optimization investigation. Read this entire handoff and its accompanying evidence before running experiments. Set a goal in this chat for the objective below. The previous Windows chat has been asked to pause; do not assume its goal state transfers automatically.

### Objective and constraints

Achieve at least 2 times vLLM-Metal's per-row decode tokens per second AND at most half its time to first token on the SAME M4 Mac mini with 16 GiB memory, for all six profiles: Qwen3-0.6B text, Qwen2.5-Coder-0.5B-Instruct, and Qwen3.5-0.8B vision language, each at batch 1 and batch 2. Use Garnet's native Metal/MPS backend. This is an actual execution and optimization task; do not substitute another inference engine for Garnet or declare completion from component timings.

Use matched learned checkpoints, FP32 arithmetic, prompts, preprocessing, output limits and full output validation. The established comparison uses different lossless weight storage layouts; describe that precisely. It is not a stock BF16 vLLM comparison or an equal-memory comparison. Retain the existing correctness and numerical bounds. Do not loosen them to admit a faster candidate.

Never run competing inference engines simultaneously. Never copy archives or run other investigation GPU jobs during a timing cohort. Disable prefix, image and answer reuse and speculative decoding. Freshly encode every image request. Record initialization, first request after initialization, warm requests, per-row decode, aggregate throughput and peak RSS separately. Hardware caches are not flushed; do not call these hardware-cold measurements. Preserve every failed attempt and excluded cohort with reasons. Only actual repeated matched measurements may satisfy the goal.

The MacBook must not silently replace the M4 Mac mini as the benchmark machine. If we later benchmark the MacBook, give that hardware a separate cohort and baseline.

### Access and protected work

The Windows investigation checkout is `D:\CantorAI\Garnet-metal`, branch `feature/qwen35-metal`. Its last committed revision at handoff preparation was `3da4d65413fe6cc98d592c09b38e359bbe0e44f8`. The seven new comparison scripts are UNEXECUTED drafts included with this document in the subsequent handoff commit on `feature/qwen35-metal`. Fetch that branch and record its actual HEAD; the revision above is its previously committed base. Check actual Git availability and preserve existing changes before creating or updating your local working copy. Do not reset an existing checkout, push, create a PR, or alter unrelated branches as part of this handoff.

Preserve these Windows checkouts completely:

- `D:\CantorAI\Garnet`, branch `feature/gpt-oss-120b`.
- `D:\CantorAI\Garnet-pi5b`, branch `feature/qwen35-vl-pi5b`.

Preserve the Mac's original checkout `/Users/zsdx/CantorAI/Garnet`. The investigation root on the Mac mini is `/Users/zsdx/garnet-metal-investigation` (called ROOT below). Use private experiment directories and freeze consumed source before launch. Never mutate source, drivers or libraries used by a running job. `.work/` contains local working material and must not be staged indiscriminately.

The previously verified Windows SSH route was:

```sh
ssh -C -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o HostKeyAlias=192.168.193.215 zsdx@100.112.239.81
```

The reachable address was a Tailscale route. The HostKeyAlias is a pinned identity, not proof of a reachable LAN address. Earlier ZeroTier connections were unstable. On the MacBook first inspect its own saved SSH configuration, authorized authentication and host trust. Do not copy private keys or passwords, print secrets, disable host-key checking or invent trust from this document. If access is missing, ask me for the necessary connection setup. For Windows SCP the working invocation also required `-O`; do not assume that is necessary on macOS.

Known target: Mac16,10, Apple M4, 10 CPU cores (4 performance and 6 efficiency), 10 GPU cores, 16 GiB RAM, macOS 15.6.1, clang 17. Recheck the live machine and versions. The target did not have `rg`; use available alternatives when needed.

Important installed paths:

```text
/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3
/Users/zsdx/CantorAI/xlang3/sdk
ROOT/comparators/vllm/venv/bin/python
ROOT/models/<model>
ROOT/models-exact-f32/<model>
```

Previously recorded comparator versions were vLLM 0.30, vLLM-Metal 0.30, MLX 0.32.1, mlx-lm 0.32, mlx-vlm 0.6.17, Torch 2.13 and Transformers 5.19. Verify installed versions; do not upgrade the comparator casually.

### Verified status and distance from the goal

The goal is UNMET. The last complete repeated matched comparison used the older `c186` library and produced 144 fully validated outputs and six actual GPU dtype audits across two cohorts. Decode passed in 4 of 6 profiles in both cohorts, TTFT in 2 of 6, and BOTH targets in 0 of 6 consistently.

Authoritative file: `results/20261015/coalesced-body-vllm/remaining-budgets.json`. The following numbers are the repeat cohort's warm means, rounded for display:

| Model | Batch | Garnet row tok/s | vLLM row tok/s | Garnet TTFT ms | vLLM TTFT ms | Decode ratio | TTFT fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 74.506 | 37.538 | 35.444 | 78.212 | 1.985 | 0.453 |
| Qwen3-0.6B | 2 | 69.930 | 25.428 | 57.417 | 97.651 | 2.750 | 0.588 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 87.241 | 46.944 | 27.554 | 58.111 | 1.858 | 0.474 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 83.788 | 28.440 | 43.484 | 81.572 | 2.946 | 0.533 |
| Qwen3.5-0.8B | 1 | 60.468 | 26.251 | 177.379 | 254.335 | 2.303 | 0.697 |
| Qwen3.5-0.8B | 2 | 55.111 | 17.085 | 299.110 | 429.147 | 3.226 | 0.697 |

Ratio is Garnet/vLLM; decode target is at least 2 and TTFT fraction at most 0.5. Relative to these Garnet repeat means, remaining decode gains are about 0.77% for text batch 1 and 7.62% for coder batch 1. Remaining TTFT reductions are about 14.96% for text batch 2, 6.21% for coder batch 2 and 28.3% for both VL batches. These gaps are against the OLD implementation and do not establish current candidate performance. Aggregate throughput is row throughput times batch only when all rows and lengths are validated.

### Selected implementation needing a fresh comparator

The newer accepted native candidate uses ordered vector vision, abbreviated `d52`. It passed full native correctness gates and two 72-output native comparison cohorts. It has NOT yet been compared against fresh vLLM cohorts. Do not divide d52 native results by older c186 vLLM measurements.

```text
Selected source: ROOT/vector-vision-native-investigation-20261016/source
Selected library: ROOT/vector-vision-native-investigation-20261016/build-vector-vision-width4/libgarnet.dylib
Library SHA256: d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e
Source manifest: ROOT/vector-vision-native-investigation-20261016/preparation.json
Manifest source_sha256 count: 1406
```

Local evidence under `results/20261016/vector-vision-native/`:

- `preparation.json`: selected source hashes.
- `comparison/initial/verification.json`: 72 outputs, actual vector vision routes verified; archive SHA256 `4391422a668b2e4b551c981635e252e00e5274f3ffa8eabd7259a325759bded5`.
- `comparison/repeat/verification.json`: 72 outputs, actual routes verified; archive SHA256 `e7536a9bca8493fc0b61589a8ef54953490f8e150dee0d094ce5d75452de6b90`.
- `comparison/decision-repeat.json`: native confirmation passed; fresh_vllm_run, target_count_update and full_goal_met are false.

Native repeat warm VL TTFT changed 179.488 to 174.999 ms at batch 1 and 297.751 to 287.726 ms at batch 2. Text batch 1 TTFT increased 35.843 to 36.799 ms. Include the costs when describing this change.

Keep the older control intact:

```text
ROOT/source
ROOT/build-coalesced-bf16-body/libgarnet.dylib
SHA256 c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1
```

The current working WorldSense process guard is in `ROOT/source/test2026/metal/compare_vllm_metal.py`; the d52 snapshot may have an older allowlist. Validate live ownership without printing unrelated command lines or environments. Do not kill unrelated jobs to get quiet measurements.

Shared exact image preprocessing is combined threaded WebP decode plus exact-coordinate strip resize, identically used by both runtimes, with fresh image input every request. The bridge is `ROOT/logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib`, SHA256 `3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3`. The wrapper resolves its bridge relative to the source parent. Before selecting the d52 source, verify that `ROOT/vector-vision-native-investigation-20261016/logs/webp-thread-probe-confirm-20261013/jobs/libwebp_thread.dylib` also exists with the same bytes. That alias has not been checked in the new draft preflight.

### First actions

1. Establish authorized SSH access. Inspect live process state and terminal exit records before any launch. At the last check, the latest nibble and two six-bit controllers had exited successfully and had complete archives; no new matched comparison was launched. Historical PID files alone do not prove current activity or ownership. Check process birth identity to avoid PID reuse mistakes.
2. Read `docs/research/metal_optimization_research_log.md`, `docs/research/metal-method-evidence/INDEX.md`, the matched budgets and the d52 decision/proof files. The last committed index recorded 207 reports and 589 links. The Mac has tested source snapshots and raw archives, but may not have the latest Windows Git history or the seven draft files.
3. Inventory existing code and output folders before copying anything. Use an isolated local working copy and private remote tools directory; preserve ROOT/source, the d52 snapshot and the original Mac checkout. If full Windows commits are unavailable, use the manifested tested Mac snapshot and clearly record provenance. The small handoff ZIP is not a substitute for a full repository clone.
4. Review and syntax-check the seven draft scripts described below. Complete the listed missing checks. Do not launch them merely because they are packaged here.
5. Verify source, libraries, runtime, frontend, learned assets, prompts, output oracles and versions; freeze their complete dependency closure. Run exactly two predeclared fresh matched cohorts after choosing the candidate. Archive and independently audit the first before starting the second. Confirmation depends on validity, not favorable speed; retain a valid performance failure and do not add a favorable third cohort.
6. Preserve the existing G/V/V/G order, two fresh workers per runtime, three requests per worker with four warm requests per runtime overall, six profiles and three actual dtype audits per cohort. Require all 72 full outputs per cohort, prompt IDs, generated IDs, stop reasons, preprocessing and cache invariants. Retry only genuine resource contention under the established rules and retain failed attempts.
7. Join both cohorts only after source/asset/case/version/selection identity checks. Update decode, TTFT and aggregate throughput charts, init/first/RSS costs, per-profile remaining gaps and repeated success counts. Never claim a target from rounded display values. Use phase measurements to choose the next bottleneck; do not repeat rejected experiments unchanged.
8. Continue optimization until the objective is actually achieved or I pause it. Log every method and failure in the research record so we can use it for a thesis paper.

### Draft drivers and known work remaining

These files under `test2026/metal/` were written in the Windows worktree but have not been deployed or run on the Mac. All seven passed Python AST syntax checks before the handoff push. They still require runtime preflight and remain drafts:

```text
compare_vector_vision_vllm.py
run_vector_vision_vllm.py
verify_vector_vision_vllm.py
prepare_vector_vision_vllm.py
launch_vector_vision_vllm.py
archive_vector_vision_vllm.py
series_vector_vision_vllm.py
```

They expand the previous comparison wrappers, select d52 source for text and VL, pin source/library/runtime/guard hashes, add actual vector-route assertions, prepare a new selection, archive source/evidence and schedule two sequential cohorts. No fresh matched result exists from them.

Review and resolve these specific gaps before launch:

- Series/launcher PID tracking should record and compare process birth identity; polling a PID alone can hang on reuse. Require the intended controller's terminal exit record.
- Save an explicit frozen output oracle with provenance, prompt IDs, generated token IDs and stop reasons. Runtime assertions against historical outputs should also be reproducible in the independent archive verifier.
- Prepare expects three proof files in a `proofs/` directory beside the tools: `native-decision-repeat.json`, `native-initial-verification.json`, and `native-repeat-verification.json`. Copy the corresponding packaged decision and verification JSONs with those exact names and preserve hashes. They were not deployed at handoff.
- Check the frontend bridge alias, full source closure, selection path and compatibility with ROOT's current guard. New selection is intended at `ROOT/work/vector-vision-next-comparator-selection.json`.
- Smoke-test archiver/verifier compatibility on synthetic or existing evidence before spending GPU time. Do not allow a packaging failure to lose a valid run.
- Implement or verify the two-cohort join, complete output comparisons, exact selection/version/source/asset/case identities and full costs. `summarize_coalesced_body_vllm_trials.py` provides the prior approach.

### Rejected recent methods

Read their complete reports before reconsidering them. Their rejection does not change the matched goal counts.

- Four-bit rejection indexes: correct checks, but component GPU time 1.394 to 1.600 times control and much larger survivor sets. Rejected without repeat or integration. Evidence: `results/20261016/nibble-screen-index/`.
- Six-bit rejection indexes: first cohort excluded because our archive transfer overlapped timing. The clean cohort failed the predefined speed/preservation screen; GPU fractions roughly 0.967 to 1.065 and wall fractions 1.004 to 1.089 across profiles. No clean confirmation or integration. Evidence: `results/20261016/sixbit-screen-index/`. Windows verified restored full manifest files; it did not verify the full six-bit gzip blobs directly. Full archives remain on the Mac.
- Contiguous BF16 prefill tiles: first candidate failed original numerical controls; restricted wide-output follow-up passed controls but was slower (GPU 1.453 to 2.132 times, wall 1.228 to 1.755). Rejected. Evidence: `results/20261016/block-packed-prefill/` and `wide-block-packed-prefill/`.
- Native width64 delta value tiles: full stage and output checks passed, but native TTFT gains were mixed and too small, with coder batch 2 regression. No confirmation or promotion. Streamed body weight consumption was numerically correct but failed the speed/preservation screen.
- Earlier paired norm/RoPE, QKV packing, selective NDArray, MLP graph, encoder reuse, prefill tiling and vision reorder experiments already have reports. Consult the index instead of rediscovering failed configurations. Reordered vision reductions failed trained numerical bounds; the selected vector vision uses preserved ordering.

### Research and reporting requirements

For every experiment save hypothesis, exact control/change, code and binary hashes, model/asset identity, commands, dependency versions, numerical gates, full output checks, sample design, individual measurements, exclusions, setup and memory costs, decision and remaining uncertainty. Archive the entire consumed source closure before changing it. No source mutation while any process consumes it. Do not multiply component gains to predict whole-model performance or describe a candidate selected from many trials as an unbiased confirmation.

The repository documentation checks are:

```sh
python3 -X utf8 docs/research/build_metal_method_index.py
python3 -X utf8 docs/research/verify_metal_research_log.py
```

These check documentation and evidence consistency, not numerical correctness or benchmark success. Report the exact current implementation and matched cohort when showing charts. Keep source, artifacts and logs reviewable. Do not claim the 2 times / half-TTFT objective until both fresh repeated matched cohorts meet both requirements in all six profiles.

## Package integrity

The accompanying ZIP contains `handoff-manifest.json`, listing SHA256 hashes of every included file other than the manifest itself. Verify it after extraction. The scripts are explicitly draft and the source revision above is the prior committed base; the subsequent handoff commit includes these drafts. Access to the full repository and existing Mac artifacts must be established separately.
