# Bounded dual-accumulator text head: rejected after real model execution

The two-accumulator / 64-thread vocabulary-head component improved text
batch 1 by 7.3% initially and 8.6% in its component repeat. This prototype
tested that exact algorithm inside Garnet, only for one-row, 1024-input,
151936-output BF16 projections. Other shapes and batch 2 retained the
existing dispatch. The experimental build option defaulted OFF. Original
BF16 learned values and FP32 inputs/accumulation/output remained; summation
order changed. There was no quantization or reuse shortcut.

All eleven standard stages passed: CTest, serving, mixed-image pipeline,
Qwen3.5 layer references, both dense-model layer references, all three full
vocabulary references at batches 1/2, and small/pretrained vision references.
All three private-cache suites also passed, covering 24 cache profiles.
Existing numerical bounds were not widened. The vocabulary references
include actual normalization and full logits, with matching greedy IDs.

The subsequent sequential native ABBA suite preserved all 72 complete
requests, prompt IDs, finished flags and fresh-image identities. Each
runtime/policy/profile has two fresh workers, three requests per worker and
four non-first requests in the warm mean. The same M4/16 GiB Mac mini,
checkpoints, capacities, prompts, output limits and stop rules are retained.
Both policies use 24 slots for text batch 1, 22 for text batch 2, 26 for
coder, and VL plans 80 / 32+32+16. No inference runtimes run in parallel.

| Model | Batch | Candidate/control decode ratio | Candidate/control TTFT fraction |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 0.99867 | 1.00001 |
| Qwen3-0.6B | 2 | 1.01767 | 0.99756 |
| Qwen2.5-Coder-0.5B-Instruct | 1 | 0.99482 | 0.98098 |
| Qwen2.5-Coder-0.5B-Instruct | 2 | 0.98959 | 1.00248 |
| Qwen3.5-0.8B | 1 | 0.99974 | 0.98987 |
| Qwen3.5-0.8B | 2 | 0.99429 | 1.00022 |

The affected text batch-1 model shows no gain. Apparent timing changes in
profiles whose head dispatch is unchanged do not establish an algorithm
benefit. No additional confirmation or fresh vLLM study is warranted.
The component timing is not a whole-model speed claim. The prototype is
rejected; all four edited production/worker files were restored byte-for-byte
on Windows and the isolated Mac source. Experimental binary and evidence
remain for audit. No changes were made to the GPT-OSS or Pi checkouts.

Selected control library SHA256:
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.
Rejected candidate library SHA256:
`80fa97a68f83eb069279c4d634704875495feae115bbf7e6aa4a9bfcd9c4cffc`.
Archive SHA256:
`8b05c5608df3b11ebb8232bb5df8e0ee7d0ade1f4798a19bfddb6f54d29a79e4`.

## Actual commands and provenance

```text
python3 run_dual_text_head_checks.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_dual_text_head_profiles.py INVESTIGATION_ROOT OUTPUT_DIR
```

These reproduction runners require the archived candidate source bundle;
the current production source intentionally omits the rejected prototype.
The build/check job and timed suite both have authoritative exit code 0.
Exact build/runtime commands, CMake options, library/source/checkpoint hashes,
all timing/output records, numerical checks and 0.5-second process monitors
are in `evidence/`. Only the owned worker group is exempt during monitoring
and only owned processes may be terminated. Sampling cannot prove absence
of arbitrarily short external jobs between polls. Fresh processes do not
establish cold hardware; initialization and first-use requests are retained
separately and filesystem/driver caches were not flushed.

Original and candidate source bytes are both archived. Fresh image tensors
and RGB files are omitted from Git with identity hashes/sizes retained.
`evidence-proof.json` verifies the successful stages, 72 selected requests,
unchanged chunk plans and retained control/candidate library identities.
This experiment does not meet or redefine the three-model 2x decode /
half-TTFT goal, which remains active and unmet.
