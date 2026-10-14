# Global lossless half scale through MPS alpha: toy API gate passed

Native exact-half prefill currently chooses each output row's lossless scale,
runs mixed-storage MPS multiplication, then launches a separate rescale kernel.
This investigation asks whether one exact common matrix scale can place the
inverse power of two in MPS's alpha coefficient, avoiding the temporary and
rescale dispatch. Learned values and FP32 activations/output remain the same.
No native implementation has been changed.

A CPU-only audit of 22 sampled layer-zero body matrices finds 20 with a common
exact scale: text 7/7, coder 7/7 and VL 6/8. Every accepted encoded value restores
the original BF16-expanded FP32 bits exactly. VL in_proj_qkv and in_proj_z have
unrepresentable values under the common scale and must retain fallback paths.
This is eligibility evidence, not GPU model compatibility or performance.

Owned toy gate PID 5180 exited 0 after 3.67 seconds on Apple M4. Real execution
uses FP32 inputs, exact scaled FP16 weights and FP32 output with alpha 2^-16,
without a separate rescale. Shapes (M/K/N) 3/7/5 and 16/1024/64 pass independent
FP64 checks under 5e-6 + 5e-5*abs(reference); maximum absolute errors are
4.38558e-9 and 2.45605e-7. The weights round-trip bit for bit. No trained GPU
projection, native full-model inference or speed claim is established yet.

The next required experiment is trained projection parity and timing against
the existing row-scaled MPS route across the actual prefill shapes, with fresh
inputs, numerical/precision checks and exclusive execution. The unchanged
overall 2x decode / half-TTFT goal remains unmet; selected libraries and
protected checkouts remain untouched.

```text
python3 audit_global_half_scale.py ROOT REPORT_JSON
python3 launch_mps_global_alpha_probe.py ROOT ROOT/logs/mps-global-alpha-toy-20261014
python3 archive_mps_global_alpha_toy.py ROOT ARCHIVE
python verify_mps_global_alpha_toy.py RESULTS VERIFICATION_JSON
```

Archive SHA-256:
`23c3283c17056fea76d4f1e21d2642751264900cf776261c9c62a037956021bb`
(24 files). The local verifier checks all hashes, executed sources, terminal
stages and the actual toy output gates. Feasibility/compatibility evidence
does not establish end-to-end latency or throughput.
