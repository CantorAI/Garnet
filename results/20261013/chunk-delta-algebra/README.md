# CPU foundation for a parallel recurrent prefill experiment

This is mathematical/numerical evidence, not a GPU implementation or speedup.
The current serving recurrence remains unchanged by this prototype.

With state layout key-by-value, normalized keys/queries and update vectors U:

- S_i = d_i S_initial + sum(j<=i) D_ij k_j U_j.
- U_i = beta_i (v_i - d_i k_i^T S_initial)
  - sum(j<i) beta_i D_ij (k_i^T k_j) U_j.
- y_i = d_i q_i^T S_initial + sum(j<=i) D_ij (q_i^T k_j) U_j.

D_ij is the product of original FP32 exp(log_decay) factors from j+1 through i.
d_i is the product from the start through i. Inactive rows use decay 1/beta 0
and return zero output. No clipping, log(0), inverse decay or learned-value
change is introduced. A unit-lower triangular solve produces U; matrix
contractions produce token outputs and final state. Blocks may also be
processed consecutively using their actual carried state.

The latest explicit-contraction CPU version passes 144 synthetic cases:
batches 1/2, lengths 2/16/32/40/64/80, ordinary/no decay, exact zero decay
from exp(-1000), inactive/all-inactive tokens, beta 0/1 and carried random
state. Both whole and 32-token segmented forms are checked.

It also passes 108 real-weight full-layer comparisons for Qwen3.5-0.8B:
first/last linear layers 0/22, batches 1/2, chunks 16/32/40/64 and chunk 80
at batch 1, three state-carry steps, inactive rows, and whole/32-slot
segmented forms. Outputs, convolution state and recurrent state are compared
against the existing independent sequential reference, including normalization,
output projection and MLP. The unchanged bound is 5e-6 + 5e-5*abs(reference).

Worst real-layer bound ratio is 0.127150753 (<= 1 passes).


The first two real-weight attempts failed. Diagnostics show prepared
sequential recurrence matches the original reference (state differences
<= 1.2e-7), while the batched NumPy matmul formulation returns inconsistent
results across repeated identical inputs. Changing a zero-stride view alone
did not resolve this. The same formula with explicit FP32 einsum contractions
is stable and passes; the specific backend cause remains unproven. Failed
source/results and both diagnostics are preserved rather than discarded.

The installed MLX-VLM Qwen3.5 source was inspected for its chunk formulation,
but its comments about CUDA performance are not M4 evidence. Its decay
clipping is not adopted. The algebra here uses actual decay products and
forward-substitutes the right-hand side, without constructing a full inverse.
An MPS/Metal implementation still requires primitive/full-layer validation,
complete generated-output checks, repeated real performance measurements
and a matched comparator study before any serving promotion.

A separate bit-preserving norm-cache experiment follows the observation
that current Garnet repeats query/key norms for every value tile and uses
two threadgroup barriers per token. It does not implement this matrix solve.

No 2x/half-TTFT target is achieved by CPU proof. The full goal remains active.

Evidence archive SHA256: 320c7cdf813d4ca12afa26053aa72f847df88c04b998165fe26c5e2e5511cbe8.
