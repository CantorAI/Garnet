The 128-term partial-sum candidate is rejected for extended dense profiles. Its 144 exact dyadic kernel cases and 11 ordinary model checks pass, and it reduces the original failing pair ratio from 1.5165756 to 0.5281886. Broader checks nevertheless fail at Qwen3 layer 0, batch 2, chunk 64, step 1: key-pair ratio 1.6368473 exceeds the existing tolerance. No speed or extended-profile support is claimed from this candidate.

The slower scalar reference job was intentionally terminated after confirming its exact owned child; the same acceptance rules were then evaluated with NumPy. Overlapping results agree. Both logs and exit status 1 are retained.

The raw diagnostic tensors remain private on the Mac. The complete evidence archive SHA256 is d7230a89dc73aab6fac465c11ea8790fc11725a78a793f0ed99c10abfb7ca81a.
