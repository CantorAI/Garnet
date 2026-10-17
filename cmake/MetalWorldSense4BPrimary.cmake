# Opt-in M5 WorldSense primary-weight profile, confirmed in WS-M5-40.
# The existing WS10 prefix preset remains available for raw-weight rollback.
include("${CMAKE_CURRENT_LIST_DIR}/MetalWorldSense4BPrefix.cmake")
set(GARNET_METAL_PRIMARY_BF16_BLOCKS ON CACHE BOOL "Lossless bounded32-bit primary BF16 blocks")
set(GARNET_METAL_LAZY_RESIDENT_DEFAULTS OFF CACHE BOOL "Keep validated default storage lifetimes")
set(GARNET_METAL_CACHED_MATRIX_METADATA OFF CACHE BOOL "Keep validated live matrix metadata")
set(GARNET_METAL_PRIMARY_TEST_FAULTS OFF CACHE BOOL "Test-only allocation failures disabled")
# Serving still selects VLM prefill64/text prefill16 + exact64token prefix cache.
# Evidence: results/20261016/worldsense-m5-primary-u32/README.md
