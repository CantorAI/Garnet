# Rejected per-head/token norm cache

The candidate computes query/key normalization factors once per head/token,
then uses them in the retained recurrent chunk kernel. It removes repeated
normalization and two per-token threadgroup barriers from each value tile.
Original FP32 arithmetic and accumulation order are otherwise preserved.

All 48 primitive chunk-versus-sequential comparisons are bit-identical,
including inactive rows, initial/carried state, chunks 1/4/16/64 and value
widths 16/32/128. The eleven standard validation checks and all 36 extended
40/80-slot real-layer references pass with the original tolerance unchanged.

One sequential native ABBA executes 24 complete image requests at batches
1/2, using chunk 80/32 on both candidate and retained control. All output IDs,
prompt IDs, finished flags and fresh RGB/patch bytes match. Each build/batch
uses two fresh workers with three requests, with four non-first warm samples.

| Batch | Candidate/control TTFT ms | TTFT fraction | Decode ratio |
| --- | ---: | ---: | ---: |
| 1 | 219.48 / 219.17 | 1.001 | 0.996 |
| 2 | 385.08 / 383.75 | 1.003 | 1.005 |

No meaningful latency or decode gain is established. The added prepass does
not improve complete requests, so the prototype C++ integration/build option
are reverted and no repeat is run solely to search for a favorable sample.
No new vLLM speed claim or 2x/half-TTFT achievement is made.

The prototype is separate from the CPU triangular-solve formulation.
It does not implement that matrix algorithm. The validated CPU formulation
still needs actual GPU implementation, numerical/model checks and measured
performance. The full three-model goal remains active and unmet.

First-use, initialization, process RSS and phase counters remain in raw
records; warm timing excludes initialization. Filesystem/driver caches were
not flushed. CTest primitive stdout is retained from LastTest.log because
normal passing ctest output does not print the test's stdout.

Completed source, build flags, commands, terminal statuses, bit/numerical
checks and timing/resource/output evidence are in completed/. Private
tensor/image bytes and binaries remain on the Mac with hashes in
evidence-proof.json. The retained control binary remains unchanged.

Candidate SHA256: c633169e2c9990c1d1851ecb4a3b8bb6526617ef1a034007cc2ca933e5194f98.

Retained control SHA256: cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8.

Archive SHA256: e777c0abea47bf10ae558424e22d573943e09c5d5effcec92401292424d8f076.
