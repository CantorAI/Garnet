# Independent MPS scheduling and joined prefill projections

These component experiments address independent first-layer gate/up prefill
projections on the same M4 / 16 GiB Mac mini. They use real checkpoint BF16
weights admitted by the selected exact-half codec, FP32 inputs/accumulation/
outputs, identical per-row scales and the retained rescaling expression.
No learned values are rounded, no images or prefixes are reused, and no
serving source or selected library changes in this experiment.

## Scheduling: reject shared-event branching

The first run compares two projections/rescales in one command buffer with
two independent GPU queues. All 24 shape/model checks are bit-identical.
Static-input completion-span speedups are often 1.1–1.3, but do not include
graph producer/consumer synchronization.

The second run adds a GPU producer, independent branch queues, three shared
events and a dependent SiLU-times-up consumer. A distinct completion event
per branch avoids one branch's higher signal value prematurely satisfying
the other branch's dependency. Increasing generations and 16 alternating
GPU-produced inputs per shape verify that stale event values do not let
consumers read an old result. All 24 checks and 384 generation comparisons
are bit-identical, including output padding.

Every synchronized completion-span ratio is below 1: roughly 0.44–0.95.
Most host-wall ratios regress too. Four command buffers and event costs erase
the static pair's apparent gain. This path is rejected for serving. GPU
command-buffer interval overlap is recorded; overlapping intervals alone do
not prove simultaneous execution of individual shaders.

## Packing: joined MPS gate/up candidate

Two other runs concatenate the exact encoded gate/up weight rows and integer
scales, execute one larger MPS projection, then split/rescale into the two
existing outputs with one kernel. A producer and dependent consumer stay in
the same command buffer on both sides. The control has two MPS projections
and two rescalers. No command-queue events or intermediate host waits are
introduced. Joined packing is checked byte-for-byte against the two encoded
source rows; half-scale admission is unchanged.

Each run covers rows 22/24/26/32/44/52/64/80 for all three models: 24 shape
checks plus 384 alternating-generation comparisons. Both the gate/up outputs
and dependent consumer are bit-identical across all checked cases in both
runs. Existing bounds remain 5e-6 + 5e-5*abs(control); none is widened.
The consumer additionally checks every output for finiteness and padding.

| Serving workload's physical rows | Initial GPU completion speedup | Repeat speedup | Repeat host-wall speedup |
| --- | ---: | ---: | ---: |
| Text B1 / 24 | 1.130 | 0.986 | 1.039 |
| Text B2 / 44 | 1.162 | 1.295 | 1.257 |
| Coder B1 / 26 | 1.096 | 1.126 | 1.087 |
| Coder B2 / 52 | 1.042 | 0.968 | 1.005 |
| VL B1 / 80 | 1.006 | 0.781 | 0.811 |
| VL B2 chunk / 64 | 1.033 | 1.030 | 1.048 |

Ratios are control/candidate; below 1 is slower. Each timing uses two ABBA
rounds, three warmups and twelve measurements per position. Absolute GPU
times vary across scans, so each scan uses its paired control. Completion
span and host wall include producer/consumer work in the joined and event
experiments. The static experiment does not include that work.

The evidence supports investigating only a bounded full-model candidate:
text at 44 physical rows / K=1024 / N=3072, and coder at 26 rows / K=896 /
N=4864. Text B1 and coder B2 signals do not repeat; broad VL promotion would
regress its 80-row profile. VL 64-row gain is small. Packing all model layers
would allocate additional joined half buffers, and cold/first-use packing
cost is outside these component timings. Those costs require measurement
before a serving decision.

## Integration work still required

No actual model TTFT, decode TPS, model hidden-state validation or fresh
vLLM comparison is established here. Only first-layer gate/up weights and
deterministic sinusoidal inputs are used. The all-model 2x decode / half-TTFT
goal remains active and unmet.

A bounded integration should recognize adjacent BF16 projections with the
same input and distinct outputs, preserve unsupported and split-key paths,
and admit only the repeated shapes. Pack caches must retain their source
weight ownership to prevent pointer-key reuse after graph disposal. Reject
invalid shapes/devices/output aliases before output mutation. A dispatch
counter should prove the real model uses the new route. Check trained
first/last layers and carried state with the existing 22/26-token references,
then measure full native ABBA inference, memory and first-use behavior.
Component gain alone is not a serving promotion criterion.

## Commands and evidence

```text
python3 run_concurrent_mps_pair.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_concurrent_mps_dependency.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_joined_mps_pair.py INVESTIGATION_ROOT OUTPUT_DIR
python3 run_joined_mps_pair.py INVESTIGATION_ROOT CONFIRMATION_OUTPUT_DIR
python3 archive_mps_pair_probes.py INVESTIGATION_ROOT ARCHIVE_PATH
python3 verify_mps_pair_probe_evidence.py EVIDENCE_DIR PROOF_JSON
```

All four durable jobs are terminal with exit 0. The archive retains exact
compiler/GPU commands, source/checkpoint hashes, versions, all raw ABBA
windows, checks, generation counts, 0.5-second process monitors, statuses
and sizes/hashes of omitted matrix assets and binaries. Other inference
runtimes are rejected; only owned probe process groups may be terminated.
Arbitrarily short external jobs between monitor samples cannot be excluded.
Protected GPT-OSS/Pi checkouts and the Mac original source remain untouched.
The selected library is still SHA256
`73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2`.

Archive SHA256:
`733fda3e04b54873620f03134ce2a1e4f8250b956c61ecd32d72dfe049da2742`.
