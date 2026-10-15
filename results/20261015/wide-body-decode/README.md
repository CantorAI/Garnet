# Eight-output BF16 body layout: no broad promotion

The private context packs eight output channels per SIMD group rather than
the retained four-channel layout. It tests64/128/256-thread groups with the
same per-output FP32 accumulation and SIMD reduction order. Original learned
BF16 bits remain. A separate packed buffer is prepared before timing and
checked exhaustively against the original CPU bits.

Initial PID86588 exited1 in11.6461s because the Metal compiler rejected mixed
scalar/vector grid parameter declarations. No numerical or timing test ran.
The39-file failed archive is preserved under `failed/`, SHA256
`f6d804a7e59428d98f83055fa392d5fcf28c7faa519946295797efdd31367614`.
Only after terminal archival, the threads-per-group input was changed to a
scalar matching the grid-position input.

Repaired PID86913 exited0 in21.2496s. Independent audit verifies44 archived
files,72 original-bit/FP64 checks,189 tail/fallback/prefill/ownership/abort
safety profiles and1152 timed submissions. All checks pass. Twenty-nine
original first-layer matrices retain the special float-key and large-body
admission policies; all original raw tensor hashes are checked against their
checkpoint slices. Two ABBA rounds and eight samples per position are used.

| Model / batch | Wall fraction64 threads | Wall fraction128 threads | Wall fraction256 threads |
| --- | ---: | ---: | ---: |
| Text / 1 | 1.0264 | 1.0394 | 1.0085 |
| Text / 2 | 1.0444 | 1.0770 | 0.9642 |
| Coder / 1 | 0.9744 | 0.9896 | 1.0115 |
| Coder / 2 | 1.0281 | 0.9700 | 1.0427 |
| VL / 1 | 1.0422 | 1.0237 | 1.0167 |
| VL / 2 | 1.0172 | 1.0290 | 1.0052 |

Fractions are candidate/control; above1 is slower. There is no broad gain,
and the small isolated differences do not justify native integration or an
unchanged repeat. This is independent, repeatedly resident projection work,
not a whole-model TPS, TTFT or vLLM result. Production code is unchanged.
The full three-model2x decode / half-TTFT goal remains active and unmet.

Repaired archive SHA256:
`75c7c5900fc723ce3e9f37678fadb8bcb2fc4e275f6e3754109aa5369c4af671`.

```text
python3 launch_wide_body_decode.py ROOT ROOT/logs/wide-body-decode-repaired-20261015 checks
python3 archive_wide_body_decode.py ROOT ARCHIVE wide-body-decode-repaired-20261015
python -X utf8 verify_wide_body_decode.py RESULTS VERIFICATION_JSON
```
