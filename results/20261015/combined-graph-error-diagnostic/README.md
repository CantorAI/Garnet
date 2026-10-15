# Combined-graph numerical diagnosis: failures isolated to special native keys

PID26997 waited for exact comparator PID23200, then exited0 after69.6834s of diagnostic work. Independent audit verifies34 files,48 aggregate and464 per-projection checks,12 first calls and zero timed windows.

Exactly four checks fail: text/coder batch2 pattern1, policies1/2, all original `model.layers.0.self_attn.k_proj.weight` matrices. All other tested projections/patterns pass. Both policies produce the same failing values. The graph remains inside the independent FP64 envelope; this does not excuse failure of the separate original native comparison bound. No tolerance changes.

| Model / physical rows | Worst index | Native value | Graph value | Original comparison bound |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B / 44 | 36107 | -0.20107841491699219 | -0.20109567046165466 | 1.505392074584961e-05 |
| Qwen2.5-Coder-0.5B-Instruct / 52 | 4958 | 0.082279682159423828 | 0.0822891965508461 | 9.1139841079711923e-06 |

Text worst comparison ratio1.1462492; coder1.0439333. This fixture-level diagnosis does not prove a general library defect. Generated code preserves inputs/graph math and adds error coordinates, removing timing. The verifier reconciles individual maxima with aggregate gates and recomputes worst-coordinate errors. Source/binary/library/asset hashes and guards remain in evidence.

Archive SHA256: `441f84a2d734024f4b476dae111bb16e0df1ea628a51e871d9752a89c7463b56`.

Next test a fixed hybrid policy retaining all special keys on the original native route and graphing all remaining projections. Include both submissions in candidate timing and omit no matrix. It requires fresh original numerical gates, repeated component timing and trained-model validation. No hybrid gain is established yet.

```text
python3 launch_combined_graph_error_diagnostic.py ROOT ROOT/logs/combined-graph-error-diagnostic-20261015 component
python3 archive_combined_graph_error_diagnostic.py ROOT ARCHIVE
python -X utf8 test2026/metal/verify_combined_graph_error_diagnostic.py RESULTS RESULTS/verification.json
```
