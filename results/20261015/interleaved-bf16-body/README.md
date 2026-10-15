# Isolated interleaved BF16 decoder body component

This is a pending component experiment, not a measured speed improvement.
The full 2x decode / half-TTFT target remains unmet.

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

PID 82899 is waiting for the exact prior validation PID69725 and native ABBA
PID72436 to terminate successfully. It then requires a verified quiet interval
and guards resources during every owned compile/GPU subprocess. No component
GPU work runs alongside those jobs. Do not change its drivers while live.

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
