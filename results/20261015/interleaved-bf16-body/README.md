# Isolated interleaved BF16 decoder body component

Both component trials passed and have some promising timings. The large-matrix
policy improved all six wall-time profiles in both trials. This is not a
complete-model speed improvement. The full 2x decode /
half-TTFT target remains unmet.

The private generator copies the current native context and header into the
job directory. It stores groups of four output columns together at each input
coordinate and checks every packed BF16 value and padded zero against the
original bytes before publication. The candidate shader loads two uint4
vectors per four input coordinates instead of four separate ushort4 vectors.
The original FP32 multiply/add order and SIMD reduction remain unchanged.
Original BF16 masters and the existing prefill representations remain present.
Only matrices with 256 < output width <= 8192 and input width divisible by four
receive the private layout. Small-output, odd-width and vocabulary paths keep
their original route. Production source files are not rewritten.

The queued test uses the 29 checkpoint-verified first-layer projections across
Qwen3-0.6B, Qwen2.5-Coder-0.5B-Instruct and Qwen3.5-0.8B, B1/B2. Four input
patterns must match the original native outputs bit for bit and remain inside
the unchanged independent FP64 envelope. Three static selection policies test
all admitted matrices, MLP only, and matrices of at least one million values.
The independent projections share one command submission. Their timings are
component measurements, not complete transformer execution or generation.

The harness also checks tail padding, fallback geometry, pending/foreign
command ownership, uninitialized input rejection, abort and recovery. Actual
private-layout dispatch counts must agree with the selected projections.
Only after numerical gates pass does it time two ABBA rounds per model/batch/
policy, with eight measurements per window. Packing and initialization are
outside these steady-state component intervals and cannot be advertised as
end-to-end gains.

PID82899 waited for validation PID69725 and native ABBA PID72436 to terminate,
then exited 0 after 19.3129 s of actual component work. An independent audit
verified 36 files, 72 numerical checks, 12 safety/fallback profiles and 1152
timed submissions. Every tested output was bit-identical to its original
native control and passed the unchanged independent FP64 envelope.

For the policy selecting all admitted matrices, initial candidate/control
wall-time fractions were:

| Model | Batch 1 | Batch 2 |
| --- | ---: | ---: |
| Qwen3-0.6B | 0.9650 | 0.8739 |
| Qwen2.5-Coder-0.5B-Instruct | 0.9411 | 0.8904 |
| Qwen3.5-0.8B VL | 0.9701 | 0.9663 |

Other policies are mixed: MLP-only text B1 was 1.0853 and coder B2 was 1.0180,
so selective packing is not automatically better. All policies and raw GPU/
wall intervals are in verification.json and the archived logs. These repeatedly
resident first-layer projection sets do not reproduce all model cache traffic.
Packing, duplicate-weight memory and initialization remain outside timing.

Initial archive SHA256:
`8b31bd20e7e7c75ecb91d7f075d9573d3377c0e2610133133b009f06fb1b815a`.
Confirmation PID98725 waited for native confirmation PID98672, then exited0
in19.2655s. The combined audit verified69 files,144 numerical checks,24 safety/
fallback profiles and2304 timed submissions. All original-bit/FP64 gates pass.

The large-matrix policy (at least one million values) is retained for the next
native integration study. Its wall-time fractions repeated as follows:

| Model | Batch | Initial fraction | Repeat fraction |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B |1 |0.9230 |0.9518 |
| Qwen3-0.6B |2 |0.9669 |0.9373 |
| Qwen2.5-Coder-0.5B-Instruct |1 |0.9608 |0.9429 |
| Qwen2.5-Coder-0.5B-Instruct |2 |0.9444 |0.9828 |
| Qwen3.5-0.8B VL |1 |0.9902 |0.9859 |
| Qwen3.5-0.8B VL |2 |0.9720 |0.9544 |

The all-admitted policy is less consistent: coder B2 changed from0.8904 to
1.0267. No complete-model speedup is inferred. Production native sources remain
unchanged by the component. Combined archive SHA256:
`2a148cbf9f835bc4d8a8fb9f336a8fd0f2840f117289a545ce6a6802135c520b`.

```text
python3 launch_interleaved_bf16_body.py ROOT ROOT/logs/interleaved-bf16-body-20261015 component
# Only after successful terminal execution:
python3 archive_interleaved_bf16_body.py ROOT ARCHIVE
python -X utf8 verify_interleaved_bf16_body.py EXTRACTED_RESULTS VERIFICATION_JSON
```

Before considering native integration, inspect actual numerical results and
both GPU/wall timings. Any promising component requires a fresh full-model
output and latency/throughput study. No vLLM ratio can be derived from these
component measurements.
