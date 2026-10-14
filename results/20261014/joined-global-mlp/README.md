# Optional joined global MLP native trial

The overall 2x decode / half-TTFT goal remains unmet. This optional experiment
passes the full numerical suite and complete-output comparison, but is not
selected: only text B2 shows repeated warm latency improvement, first-use
latency regresses substantially, and there is no established decode gain.

`GARNET_METAL_JOINED_GLOBAL_MLP` defaults OFF and requires global half alpha.
It recognizes adjacent BF16 gate/up products followed by SiLU. Eligible pairs
use one common-alpha MPS product, followed by a split-SiLU encoder writing all
three existing independent gate/up/activation buffers. Public outputs retain
zero offsets, their own strides, readiness and ownership. This differs from
the preceding component probe's shared output views because native custom
kernels currently assume zero buffer offsets. No internal MPS kernel count is
claimed. The same FP32 SiLU expression is retained.

Only the tested text physical rows 24/44, coder rows 26 and VL rows 32 are
considered. Both weights must already have identical exact common scales;
other pairs fall back. The packed half weights replace the individual half
allocations using MPS subviews, while raw BF16 decode buffers and learned
values remain unchanged. Cache owners retain both weights. Begin/Abort and
session clear/reset/run/failure reset the counter; the session publishes it
only after successful submission. Adapters record actual prefill block counts.

The sampled VL layer-0 gate/up scales are 18/19. Although the component can
encode their union exactly, this conservative native version does not reencode
different scales and has **zero actual VL coverage** in both frozen profiles.
Therefore its observed VL timing changes are not an algorithm gain.

## Actual validation and full-six trial

Source installation checked all twelve prior hashes against branch base
`9a5d332d` and preserved original bytes. Thirteen candidate files include the
new native test. Candidate library SHA-256:
`fc78c4354e2f6d5264739c5c95b8fadec7ada21b074ec6b1727ff9262c6dfdea`.
The unchanged global-alpha control is
`887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f`.

Owned validation PID 53015 exits 0 in 1203.85 seconds. All 19 original stages
pass, including eight complete original dense-tail fixtures per dense model,
all three private-KV suites, model layers/heads, images, vision and extended
prefill. Both CTests pass. The new test checks all three complete output bits
on four admitted layouts with fresh inputs and compact-weight reuse, scale
mismatch and alias fallback, invalid shapes, uninitialized input, foreign
pending outputs, Begin/Abort/recovery, and unchanged raw BF16 decode against
independent FP64. Existing numerical envelopes are unchanged.

Dependency PID 56650 verified the exact live validation handle and waited for
terminal exit 0 before benchmarking all six profiles. It exits 0 in 1160.18
seconds including that wait. All **72 complete outputs** match the frozen
prompt IDs, tokens, finished flags and fresh image hashes.

| Model / batch | Actual candidate blocks per request | Control -> candidate tokens/s per row | Control -> candidate warm TTFT ms |
| --- | ---: | ---: | ---: |
| Text / 1 | 17 | 58.9193 -> 58.9880 | 49.2853 -> 49.9472 |
| Text / 2 | 17 | 57.4675 -> 58.0676 | 83.4236 -> 79.5944 |
| Coder / 1 | 13 | 71.2169 -> 72.2150 | 33.7151 -> 32.6480 |
| Coder / 2 | 0 | 70.4435 -> 70.6646 | 50.9782 -> 52.0180 |
| VL / 1 | 0 | 47.9196 -> 47.8693 | 202.2766 -> 194.4799 |
| VL / 2 | 0 | 44.2317 -> 44.3724 | 361.1835 -> 359.6120 |

Four warm requests/policy/profile from two fresh sequential workers; each
worker's first request is excluded. B2 aggregate TPS is twice the per-row rate.
Controls have zero joined blocks. Unrelated experimental norm/head/reset
features stay off. First requests AFTER initialization for the three active
profiles regress: text B1 162.43 -> 253.13 ms, text B2 192.75 -> 304.86 ms,
coder B1 125.19 -> 235.68 ms. These phases are separate from initialization;
OS caches are unflushed and no hardware-cold sum is claimed.

## Confirmation and decision

Fresh confirmation PID 77790 exits 0 in 38.11 seconds. It repeats only the two
initial positive warm signals and verifies another 24 complete matching
requests. The full six-profile initial trial, including failures and zero
coverage, remains preserved.

| Profile | Repeat control -> candidate tokens/s | Repeat control -> candidate warm TTFT ms | First request after init ms |
| --- | ---: | ---: | ---: |
| Text / 2 | 57.9456 -> 57.8718 | 83.1893 -> 80.0942 | 187.39 -> 232.90 |
| Coder / 1 | 71.6534 -> 72.3919 | 32.0510 -> 32.2905 | 125.01 -> 169.98 |

Text B2's warm reduction repeats at 3.72%, while decode is -0.13%; coder's
initial 3.17% warm reduction disappears and repeat TTFT is 0.75% worse.
Both first-use penalties repeat, about 45 ms. Modest decode differences are
inconclusive: the decode arithmetic/storage route itself is unchanged. No
general admission or fresh vLLM comparison follows. Selected libraries and
the protected GPT-OSS and Pi checkouts remain untouched.

The local verifier independently checks 699 archived file hashes, all nineteen
accepted stages/cases and monitors, source/binary identities, the new CTest,
all **96 complete requests**, actual route counts, warm means and separate
first-use observations. Full precision data are `native/verification.json`
and `native/confirmation.json`; prefill phase telemetry is recorded separately.
GPU/wait intervals overlap and must not be added. The 0.5-second guard rejects
other inference/probe jobs and stops only owned groups; arbitrarily short
external activity between samples cannot be excluded.

Initial archive SHA-256:
`587ca488197dc7925672263116aa7e724f362a3f020f92a6e044b2399592e1ca`
(627 files). Combined archive SHA-256:
`481318e6fd54db91c4f97f81ec5557e12396afc519747234980ba28cf2436252`
(699 files). Historical live snapshots remain in `native/pending`; they do
not supersede the completed evidence in `native/evidence`.

```text
python add_joined_global_mlp_counters.py
python prepare_joined_global_mlp_sources.py
python3 install_joined_global_mlp_sources.py ROOT
python3 launch_joined_global_mlp_checks.py ROOT ROOT/logs/joined-global-mlp-checks-20261014 checks
python prepare_joined_global_mlp_profiles.py
python3 queue_joined_global_mlp_profiles.py ROOT ROOT/logs/joined-global-mlp-native-20261014
python3 launch_confirm_joined_global_mlp_profiles.py ROOT ROOT/logs/joined-global-mlp-confirm-native-20261014
python3 archive_joined_global_mlp_native.py ROOT ARCHIVE
python verify_joined_global_mlp_native.py RESULTS VERIFICATION_JSON
python verify_joined_global_mlp_native.py RESULTS CONFIRMATION_JSON --confirm
```

The counter preparation command was used once while editing the base source;
it rejects reapplication to this completed branch. Rebuild from the archived
candidate source bundle or the committed implementation, preserving the pinned
control rather than rerunning preparation on an already modified tree.
