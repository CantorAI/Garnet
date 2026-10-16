# Chained paired normalization and RoPE measurements

Final component selection: text and coder decode qualify after two complete
unchanged cohorts. VL decode fails confirmation because an own-null wall
fraction exceeds 1.05. Prefill never qualifies. Do not rerun unchanged failed
workloads or infer complete-model speed from these component measurements.
[Confirmed selection and cohort identities](decision-repeat.json).

## Distinct calibration workload

The earlier one-pair timing cohort had unstable own-nulls in 16/24 profiles.
Preserve it. This calibration executes 32 dependent normalization pairs in one
command batch, retaining independent outputs for every stage. Each stage
consumes the previous stage's output; the weights and positions of a selected
learned layer remain fixed across the 32 stages. This is a repeated operator
chain, not 32 actual model layers or a request benchmark.

The paired kernel and original arithmetic are unchanged. Use actual learned
native Q/K projection fixtures, checkpoint norm weights, original 32-lane
reductions and precise trigonometry. Compare every stage's output bit for bit
to 32 original pairs. No intermediate output is omitted. Original hidden
fixture preparation is outside the timed chain and identical across policies.
The component workload introduces no serving model cache or speculation.

All 58 separate full-attention layers, four decode/prefill profiles and four
changed-input variants are tested: 232 profiles and 928 cases per cohort.
Each complete numerical cohort compares 1,514,274,816 original FP32 values bit for bit;
the timed endpoints additionally compare 236,322,816 FP32 values per cohort.
[Initial audit](initial/verification.json), [repeat audit](repeat/verification.json).

## Fixed timing design and selection

Fix first/last separate full-attention layers text 0/27, coder 0/23 and VL
3/23 before execution. Include decode rows 1/2 and original serving prefill
row counts. Keep 2 s alternating warmup, own-null followed by comparison,
10 ABBA rounds, 3 warm calls and 5 samples/window. Each cohort retains all
9,600 samples in 1,920 windows. Every measured window checks all stage outputs.
GPU, encode and wall intervals describe the entire 32-stage chain separately.

Evaluate decode and prefill independently. A model/workload requires >=5%
GPU AND wall gain at all four first/last-layer x B1/B2 endpoints, with every
own-null within 5%. Nulls are exclusion tests and never normalize gains.
Only a fully passing first workload admits one complete unchanged confirmation.
The confirmation repeats every numerical and timing profile, including
previous failures; a favorable repeat cannot rescue a failed initial workload.

Initial PID 48751 exits 0 in 149.624152750 s; all 24 own-nulls pass. Confirmation
PID 51322 exits 0 in 148.896526375 s; 21 own-nulls pass. Source, generated C++,
shared context, build flags, checkpoint/config identities and timing design
are identical. The launch record verifies identities before confirmation.
Across both cohorts, all 3,028,549,632 numerical-stage FP32 values match bit for
bit, and all 19,200 timing samples pass source/data audits on Mac and Windows.
The JSON count original_output_bits_checked counts FP32 elements compared as
32-bit words; it does not count individual bit positions.

| Model | Workload | Initial screen | Repeat screen | Confirmed |
| --- | --- | --- | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | decode | Pass | Pass | Yes |
| Qwen2.5-Coder-0.5B-Instruct | prefill | Fail | Fail | No |
| Qwen3-0.6B | decode | Pass | Pass | Yes |
| Qwen3-0.6B | prefill | Fail | Fail | No |
| Qwen3.5-0.8B | decode | Pass | Fail | No |
| Qwen3.5-0.8B | prefill | Fail | Fail | No |

## All endpoint comparisons

Fractions are candidate/control, lower is better. Pass requires both gains
and own-null stability. These are operator-chain results, not token speeds.

| Model | Layer | Rows | Initial GPU | Initial wall | Repeat GPU | Repeat wall | Initial pass | Repeat pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 1 | 0.544933 | 0.840129 | 0.538735 | 0.790163 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 2 | 0.549620 | 0.817298 | 0.565602 | 0.822891 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 26 | 0.806725 | 0.851189 | 0.867046 | 0.907566 | Yes | No |
| Qwen2.5-Coder-0.5B-Instruct | 0 | 52 | 0.941462 | 0.935846 | 0.919079 | 0.929085 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 1 | 0.575500 | 0.826912 | 0.597890 | 0.836222 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 2 | 0.531692 | 0.733536 | 0.548838 | 0.822841 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 26 | 0.825780 | 0.882007 | 0.826763 | 0.891881 | Yes | Yes |
| Qwen2.5-Coder-0.5B-Instruct | 23 | 52 | 0.940040 | 0.952523 | 0.951639 | 0.963813 | No | No |
| Qwen3-0.6B | 0 | 1 | 0.542679 | 0.766026 | 0.557993 | 0.817144 | Yes | Yes |
| Qwen3-0.6B | 0 | 2 | 0.558081 | 0.771393 | 0.562425 | 0.774483 | Yes | Yes |
| Qwen3-0.6B | 0 | 24 | 0.951578 | 0.967572 | 0.911570 | 0.927091 | No | Yes |
| Qwen3-0.6B | 0 | 44 | 0.995719 | 0.997366 | 0.990090 | 0.987355 | No | No |
| Qwen3-0.6B | 27 | 1 | 0.567556 | 0.775895 | 0.558646 | 0.779106 | Yes | Yes |
| Qwen3-0.6B | 27 | 2 | 0.552325 | 0.744287 | 0.549319 | 0.735030 | Yes | Yes |
| Qwen3-0.6B | 27 | 24 | 0.937172 | 0.954517 | 0.953879 | 0.952541 | No | No |
| Qwen3-0.6B | 27 | 44 | 0.950316 | 0.962730 | 1.001757 | 1.005519 | No | No |
| Qwen3.5-0.8B | 3 | 1 | 0.536298 | 0.737160 | 0.528548 | 0.725241 | Yes | Yes |
| Qwen3.5-0.8B | 3 | 2 | 0.550590 | 0.742165 | 0.523339 | 0.723537 | Yes | No |
| Qwen3.5-0.8B | 3 | 80 | 0.974112 | 0.968312 | 0.978603 | 0.971341 | No | No |
| Qwen3.5-0.8B | 3 | 160 | 1.063984 | 1.048762 | 1.017936 | 1.018648 | No | No |
| Qwen3.5-0.8B | 23 | 1 | 0.533629 | 0.728561 | 0.549400 | 0.736980 | Yes | Yes |
| Qwen3.5-0.8B | 23 | 2 | 0.532647 | 0.720705 | 0.544487 | 0.729156 | Yes | Yes |
| Qwen3.5-0.8B | 23 | 80 | 1.079534 | 1.070652 | 0.991281 | 0.980156 | No | No |
| Qwen3.5-0.8B | 23 | 160 | 1.049214 | 1.037838 | 0.979912 | 0.974526 | No | No |

Repeat exclusions: coder layer 0/rows 26 own-null wall fraction 1.053693;
VL layer 3/rows 2 wall 1.060406; VL layer 3/rows 160 wall/GPU
0.895106/0.883358. The VL decode comparison looks favorable, but its own-null
failure excludes it. Coder prefill had already missed initial layer 23/rows
52 wall gain (fraction 0.952523), so no prefill confirmation is admitted.
[All raw means, samples and nulls](repeat/analysis.json).

The chain harness retains stage outputs and reference buffers solely for
verification. Initial numeric/timing peak RSS is 703,709,184/419,332,096 bytes;
repeat is 613,023,744/529,317,888 bytes. These are component process peaks,
not model memory or added serving allocations. The paired operator itself
adds no joined weight or output buffers to its original two operations.

Frozen archive SHA256 values:

- Initial, 51 files: `38039a6a59314a074a44e56bc6da917589d4ec019abcfc92233388e2348317e0`.
- Confirmation, 52 files: `3de80d08742a5ff4980dbe2aa5c03b455ce77c2a6d143790e59e4b34bf8e7573`.

Both complete archives are independently extracted, source-regenerated and
audited on Mac and Windows. A first local extraction was attempted before
SCP finished and could not open the file; no evidence was extracted or edited.
The same transfer completed 0 before successful extraction and audit.

The selected text/coder row-1/2 policy proceeds only to a private native
investigation. Native correctness, complete-model timing and fresh matched
vLLM measurements remain separate gates; the overall goal is still unmet.
