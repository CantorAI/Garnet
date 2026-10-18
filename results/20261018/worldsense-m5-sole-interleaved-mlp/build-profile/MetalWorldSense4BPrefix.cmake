# Opt-in M5 WorldSense profile, WS-M5-10 repeated real-pipeline confirmation.
# Keep the base profile available for cache-disabled control deployments.
include("${CMAKE_CURRENT_LIST_DIR}/MetalWorldSense4B.cmake")
set(GARNET_METAL_QWEN35_4B_WIDE_PREFILL ON CACHE BOOL "Validated4B prefill64")
set(GARNET_METAL_DENSE_PREFIX_CACHE ON CACHE BOOL "Bounded exact64-token dense text KV snapshots")
# Serving profiles must separately select VLM prefill64 and textPrefixCache=true.
# Evidence: results/20261016/worldsense-m5-prefix-cache/README.md
