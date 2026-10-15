# MPS NDArray API compatibility: failures, coefficient path and deferred views

Status: all three API experiments terminal, archived and independently audited
on the M4 Mac and Windows. These are synthetic compatibility tests, not trained
model inference, backend promotion, latency improvements or memory measurements.

The prior [physical half transpose](../../20261013/transpose-half-prefill/README.md)
was already tested and did not establish a repeatable broad benefit. Avoid
rerunning it unchanged. Instead inspect installed Xcode SDK headers and test
`MPSNDArrayMatrixMultiplication`, typed buffer aliases, alpha/beta and deferred
transpose views. No external package installation or serving-code change.

Host: Apple M4,16GiB,macOS15.6.1; clang17, runtime Metal compilation. SDK
headers, actual compiler commands, generated sources, stage logs and exclusive
guards are frozen in each evidence directory. Original control is FP32 A,
exact scaled-half B, FP32 output through MPSMatrix with inverse-power-two alpha.
Candidate uses FP32 A/output and either exact-half or exact-expanded FP32 B.

## Actual outcomes

| Experiment | Actual PID / exit | Numerical records | Outcome |
|---|---|---:|---|
| Two-source alpha2^-8 | 76186 / 1 | 2 failed | Both storage modes failed the original bounds |
| Unit alpha / explicit third zero source | 76652 / 0 | 64 passed | All four storage/coefficient modes passed |
| Deferred transpose, same coefficient modes | 77981 / 0 | 64 passed | Logical descriptor/alias checks and all numerical checks passed |

Wall times10.254942542s,13.493525500s,13.520210875s respectively include setup,
compilation and correctness work. They are not inference latency benchmarks.

Initial two-source results are approximately256 times expected with alpha2^-8.
The native control passes. A diagnostic inverse scaling matches the control
closely, but is never used to admit the failed candidate. This is an observation
about this configuration/host, not proof of an Apple bug or unsupported dtype.
The raw field `complete_checks` counts emitted numerical records, including the
first failure, rather than successfully completed checks.

The follow-up separately tests mixed/unit, mixed/three, F32/unit and F32/three:
two sources with alpha1, or three with alpha2^-8 and an immutable FP32 zero
addend, beta0. Each mode passes16 checks. Property getters match requested
coefficients. Explicit zero storage, weights and activation buffers remain
unchanged; all16 output sentinels remain intact per check.

The deferred-view diagnostic tests both packed-row preferences across four
shapes and all four modes:32 view records. Both preferences return typed aliases.
Array `lengthOfDimension` retains physical K,N, while its descriptor reports
logical N,K after transpose. This explains the first
[trained-context assertion failure](../ndarray-prefill/README.md#initial-native-failure).
The diagnostic actually multiplies using the deferred view with packed-row
preference disabled and passes64 checks; it does not prove arbitrary layouts,
OS versions, dimensions, ownership recovery or model-level performance.

All gates preserve the original FP64 reference envelope
`gamma(ceil(K/32)+32)*sumabs+1e-7` and native comparison
`5e-6+5e-5*abs(native)`. Shapes(M,K,N) are(3,8,16),(16,1024,64),
(80,64,32),(160,64,32), with four signed/scaled/impulse input patterns.
Independent NumPy FP64 audits reconstruct every saved output reference and
exact BF16/half/transposed/F32 fixture. Statistic reconciliation tolerance
only checks reported summaries; it does not relax output acceptance bounds.

## Frozen evidence and identities

| Evidence | Files | Archive SHA256 |
|---|---:|---|
| [Initial failed gate](initial/verification.json) | 24 | `1274759a25ebfdb31746f2f9d8595b8a92348156d1c9259cce3e5e7f8c64116a` |
| [Coefficient diagnosis](alpha-diagnostic/verification.json) | 34 | `4bc6ecef917beb0c9c77b2c68c7722023c77bc4b1bea14168a166d594dd1f5b5` |
| [Deferred-view diagnosis](deferred-view/verification.json) | 35 | `528ed48d0f0e19aeaf3667ffc28a0b5e14ec885ad04b0c77e16708e1fb7c925a` |

The initial gate was archived before the coefficient diagnostic; that diagnostic
was archived before the native attempt; the native failure was archived before
the descriptor repair. An early local read during archive transfer failed due
to Windows file locking; extraction was retried only after transfer completion.
That observation failure did not trigger inference reruns or gate changes.

Reproduce independent CPU audits from repository root:

```text
python -X utf8 test2026/metal/verify_ndarray_compat.py results/20261015/ndarray-matrix-api/initial OUT_INITIAL.json
python -X utf8 test2026/metal/verify_ndarray_compat.py results/20261015/ndarray-matrix-api/alpha-diagnostic OUT_ALPHA.json
python -X utf8 test2026/metal/verify_ndarray_view_probe.py results/20261015/ndarray-matrix-api/deferred-view OUT_VIEW.json
```

The auditors preserve failure status and do not convert compatibility into a
speed claim. Serving remains the retainedc186 native implementation. The
full2x/half-TTFT goal remains unmet; trained component/full-model/fresh matched
comparisons are still needed before any promotion.
