// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
// Native payload only: never expose function addresses as script integers.
struct GarnetOperatorModuleBridge {
    uint32_t abi;
    uint32_t size;
    const char* (*manifest)();
    int (*register_backend)();
    void* (*resolve_callback)(const char* name);
    const char* (*binary_path)();
};
inline constexpr const char* GarnetOperatorBridgeType = "garnet.operator_module_bridge.v1";
