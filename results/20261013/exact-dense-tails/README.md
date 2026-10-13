# Exact dense prefill tails: validated modest latency gains

Dense Qwen can opt into 22- and 26-token prefill profiles to remove padding
for the existing matched text and coder prompts:

```python
decoder.enable_prefill(32, 'shared', finer_chunks=True, exact_tail=True)
```

This requires a resident split-key Metal session, shared projection and
hidden size <= 1024. The option defaults false. Existing scheduling remains
unchanged unless enabled. Switching the option closes cached prefill graphs.
Unsupported combinations reject before changing decoder state. There is no
new native kernel, precision change, checkpoint change or output shortcut.
Both A/B policies use the same private-KV library, SHA256
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.

## Actual validation and native measurements

Independent NumPy references pass all 48 comparisons: both checkpoints,
first/last layers, batches 1/2, both chunk sizes, three carried-state steps,
inactive slots and overwrites. The worst tolerance ratio is
0.9936574155; the existing bounds were not widened. CPU admission/planning
checks pass 2,048 cases, including default scheduling, capability rejection,
state preservation and graph disposal when switching policy.

Initial sequential native ABBA measurements preserve all 48 complete
requests, prompt IDs, stop rules and chunk plans. Each policy has two fresh
processes, three requests per process and four non-first requests in its
warm mean. Original BF16 weight storage and FP32 arithmetic remain. The
hardware is the same M4 Mac mini with 16 GiB and macOS 15.6.1.

| Model | Batch | Control TTFT ms | Exact-tail TTFT ms | Reduction | Decode ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 52.99 | 52.85 | 0.3% | 0.997 |
| Qwen3-0.6B | 2 | 92.77 | 87.11 | 6.1% | 0.994 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 40.88 | 39.02 | 4.5% | 0.995 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 63.40 | 59.23 | 6.6% | 0.997 |

Text batch 2 has a separate clean confirmation: 91.86 -> 86.50 ms, a
5.84% reduction; decode is 58.38 -> 58.65 steps/s/row (ratio 1.0045).
All 36 confirmation requests preserve complete outputs and prompt IDs.
Text batch 1 showed no useful gain and was not repeated to seek a favorable
result. Coder confirmation reduces batch-1 TTFT from 39.30 to 38.34 ms
(2.45%) and batch-2 TTFT from 62.70 to 58.19 ms (7.18%). Their decode ratios
are 1.0004 and 0.9953. The coder latency direction repeats, with a smaller
batch-1 effect than the initial run. These are prompt-specific results;
they do not establish a benefit for arbitrary lengths or larger models.
Decode throughput is effectively unchanged in this experiment.

The confirmation manager completed with exit code 0.
Coder batch-1 attempt 0 was interrupted and discarded when a separate
release-test runtime appeared. No measurements from that attempt are used.
The external runtime finished, and the manager automatically executed the
remaining clean profiles. Failed-attempt logs are retained separately.
The external process was not stopped or exempted based on its low CPU usage.
The manager waits for a five-second clear interval, monitors workers every
0.5 seconds and has a 900-second quiet-window deadline. It retries only
contention failures. This sampling cannot prove absence of arbitrarily
short external jobs between polls.

Peak child RSS, initialization, first-use requests and phase timings are
preserved separately in the records. Filesystem/driver caches were not
flushed; fresh processes do not establish cold-hardware behavior.
This is a native A/B experiment, not a fresh vLLM-Metal comparison. It does
not establish the full 2x decode / half-TTFT goal or stock BF16 speedups.
Qwen3.5 VL is unaffected. The goal remains active and unmet.

## Commands and provenance

Public runners under `test2026/metal`:

```text
python3 test_exact_dense_tail_planning.py SOURCE_ROOT OUTPUT_DIR
python3 run_exact_dense_tail_checks.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_exact_dense_tail_profiles.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_exact_dense_tail_profiles.py INVESTIGATION_ROOT OUTPUT_DIR --confirm
```

These commands execute in the isolated Mac investigation. Exact runtime
commands, checkpoint/library/source hashes and original frozen prompts are
in the evidence. The archive SHA256 is recorded in `archive-sha256.txt`.
`evidence-proof.json` verifies 48 references and 84 clean timed requests
and explicitly records `confirmation_complete: true`. Only the local
experiment's confirmation is complete; the full comparator goal is unmet.

Two initial reference-run failures are retained: the guard helper initially
lacked its output binding, before inference; then a 64-slot reference fixture
exhausted capacity at the third 26-token step. The corrected fixture uses
128 slots, without changing numerical bounds. The benchmark depending on
that failed reference run never executed; its failure is also preserved.
Source history retains those versions and the later confirmation runner.
