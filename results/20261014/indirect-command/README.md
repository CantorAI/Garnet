# Recorded compute commands: lower CPU encoding, no broad wall-time gain

This new component probe records a dependent FP32 addition chain into a
Metal indirect command buffer once, then executes it with fresh input
values on every request. Commands retain buffer bindings and parameter
buffers, declare resource usage and use an explicit barrier after each
dispatch. The same shader, buffers and arithmetic serve three controls:
separate direct encoders, one direct encoder with buffer barriers, and
recorded indirect commands. There is no output, prefix or encoder-result
reuse. This is a synthetic command-encoding gate, not Qwen inference.

Actual execution on the M4/macOS 15.6.1 passed. The shader compiles with
fast math disabled and actual indirect-pipeline support enabled. Nine
profiles cover 896/1024/2048 elements and 16/64/256 dependent dispatches.
Two rounds use direct/indirect ABBA and single/indirect ABBA windows,
three untimed warmups and twelve timed samples per window. All 2,160
complete outputs match the CPU dyadic reference bit for bit, including
warmups and changing inputs; 1,728 timing records remain. Compilation
and the guarded child exited 0. Owned durable PID 96760 completed after
4.16 seconds; no probe from this job remains running.

Across all component profiles/rounds, the ratio of pooled means is
0.21746 for CPU encoding time, 1.08386 for wall time and 1.53763 for GPU
time (indirect/direct). Encoding improves about 78%, but mean wall time
is about 8% worse. Individual wall-time ratios are mixed; 64/256-dispatch
chains are mostly slower. The repeated 1024-element/16-dispatch case
improves 5.4%, while many other repeats regress. The one-encoder control
is generally faster than indirect execution as well. Encoding, GPU and
wall clocks are separate, overlapping observations and are not added.
Recording/setup time is separate in the raw log.

This gate does not justify broad native integration. The addition chain
does not prove behavior or performance of real BF16 matvec, MPS prefill,
attention, KV writes or mixed MPS/custom-kernel graphs. A future realistic
kernel experiment would need actual numerical and wall-time evidence;
the CPU encoding reduction alone is insufficient. Selected Garnet
libraries are unchanged, and no fresh vLLM comparison is claimed.
The full 2x decode / half-TTFT goal remains unmet.

The source uses the actual installed SDK declarations for
`MTLIndirectComputeCommand`, `setBarrier`, `setKernelBuffer`,
`concurrentDispatchThreads`, `executeCommandsInBuffer` and pipeline
`supportIndirectCommandBuffers`. Apple documents [indirect compute
commands](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand)
and [barriers](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setbarrier()).
Actual pipeline creation and execution, rather than SDK declarations
alone, establish compatibility on this Mac.

Commands:

```text
python3 launch_indirect_command_probe.py ROOT ROOT/logs/indirect-command-probe-20261014
xcrun clang++ -std=c++17 -fobjc-arc -O3 -framework Foundation -framework Metal -framework MetalPerformanceShaders indirect_command_probe.mm -o OUTPUT/indirect_command_gpu
OUTPUT/indirect_command_gpu
python summarize_indirect_commands.py results/20261014/indirect-command
```

The local summarizer verifies terminal exits, guarded samples, record
counts, exact flags, source identities and independently recomputes all
encoding/wall/GPU means. Its first postprocessing attempt had a Python
`round` name collision; that was corrected without rerunning or changing
the raw measurements. The successful summary is `summary.json`.
Raw logs/metadata/terminal state are in `evidence`; source and helper
files remain in `test2026/metal`. The 0.5-second monitor does not rule out
arbitrary shorter external activity.
