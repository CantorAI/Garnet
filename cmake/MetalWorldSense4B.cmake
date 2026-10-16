# Opt-in low-copy Metal kernel profile for WorldSense's Qwen3.5-4B.
# Use: cmake -C cmake/MetalWorldSense4B.cmake -S . -B BUILD ...
# Original BF16 values / FP32 arithmetic; no extra body-weight layouts.
# Fresh M5/24GiB smoke cohorts are documented in results/20261016/4b-m5-chunk.
set(GARNET_METAL_PARALLEL_DELTA_SPLIT ON CACHE BOOL "Parallel recurrent dot reductions")
set(GARNET_METAL_LINEAR_CHUNK_CONV ON CACHE BOOL "Chunk causal convolution")
set(GARNET_METAL_LINEAR_CHUNK_DELTA ON CACHE BOOL "Register-resident chunk recurrence")
set(GARNET_METAL_LINEAR_CHUNK_DIRECT_WRITE ON CACHE BOOL "Direct recurrent output writes")
set(GARNET_METAL_ATTENTION_PREFIX ON CACHE BOOL "GPU-verified contiguous prompt slots")
set(GARNET_METAL_RESIDENT_KV_INPLACE ON CACHE BOOL "Private resident token-slot updates")
