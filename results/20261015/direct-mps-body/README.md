# Direct original-row mixed-storage MPS component

Actual owned PID 60430 exited 0 after 34.2084 s. An isolated generated copy
of Garnet's context enables its common-scale MPS matrix multiplication at the
original logical B1/B2. A controlled host flag chooses raw BF16 control or
MPS for the same original-sized inputs. The original serving context and
selected libraries remain unchanged. This differs from the earlier rejected
FP32-expanded MPS GEMV and from physical-row padding.

Independent verification covers 46 archived files, 216 sampled numerical
checks, 54 admitted profiles and 3456 timed submissions. All checks pass the
unchanged FP64 envelope and original-control comparison bounds, with one
actual global-scale MPS dispatch per candidate. Original learned values are
unchanged; reduction order may change FP32 result bits. Two of the 29 trained
matrices do not admit common-scale half storage and are excluded.

| Model / batch | Sum GPU fraction | Sum wall fraction |
| --- | ---: | ---: |
| Text / 1 | 0.9011 | 1.1154 |
| Text / 2 | 0.9399 | 1.1117 |
| Coder / 1 | 0.8752 | 1.0379 |
| Coder / 2 | 0.9097 | 1.0870 |
| VL / 1 | 0.9258 | 1.0701 |
| VL / 2 | 0.9280 | 1.0682 |

GPU intervals improve by about 6–12.5%, while wall intervals worsen in every
aggregate. Fractions are ratios of sums of measured individual matrix means,
not full-model performance. Inputs remain resident, input upload/validation
is outside timing and each timed projection completes a separate submission.
Four input patterns, 250 ms common warmup and two ABBA rounds with eight timed
operations per position match the padded experiment. Numerical failures would
exclude a policy from timing. No production integration or TPS/TTFT claim
follows from GPU intervals alone.

The next test encodes multiple real projections in one command submission to
check whether the wall penalty persists when submission overhead is amortized.
Its results are in [the streamed study](../streamed-direct-mps-body/README.md).
The full 2x/half-TTFT goal remains unmet.

Archive SHA-256:
`16aae610b16f66421ed2c3777a751590810418000ef32628f804189d0fa5b61f`.
The audit regenerates the exact context/harness from archived source, checks
the original 17-file source manifest, compiler definitions, actual route
counts, numerical admissions and every ABBA window. Checkpoint/raw identities,
commands, guards and binary hashes are preserved in `evidence/`.

```text
python3 launch_direct_mps_body.py ROOT ROOT/logs/direct-mps-body-20261015 checks
python3 archive_direct_mps_body.py ROOT ARCHIVE
python -X utf8 verify_direct_mps_body.py RESULTS VERIFICATION_JSON
```
