# Apple M4 mixed-input FP32 accumulator capability

The isolated hardware probe exited 0 in 2.39 seconds after the sparse profiling
job ended. Metal accepts half SIMD matrix operands and an explicitly declared
FP32 accumulator on this M4. The two actual numeric fixtures distinguish this
from half accumulation:

| Half inputs | Initial accumulator | FP32 result | Half control |
| ---: | ---: | ---: | ---: |
| 0.25 | 2048 | 2112 | 2048 |
| 2^-14 | 0 | 2^-18 | 0 |

Each fixture executes 128 matrix multiply-accumulate iterations using operands
loaded from a device buffer; all 64 FP32 outputs per fixture are checked exactly.
Fast math is disabled. Independent verification covers 17 files and 128 outputs.
This proves a hardware precision capability only. The subsequent learned-matrix
coalesced component passed its error gates but was slower and is rejected.

Archive SHA-256: `ea84e70df7488c3f305828ddbb310115a5a3110c7cd0684ac40c5648577ad29f`.
Exact compilation/execution commands, source, monitored stages and dependency
record are in `evidence/`.

```text
python3 run_mixed_accumulator_capability.py ROOT ROOT/logs/mixed-accumulator-capability-20261015/jobs
python3 archive_mixed_accumulator_capability.py ROOT ARCHIVE
python -X utf8 verify_mixed_accumulator_capability.py EVIDENCE VERIFICATION_JSON
```
