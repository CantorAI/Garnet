// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
// Native payload only. Streams and owners never travel as script integers.
struct GarnetOperatorExecutionServices {
    uint32_t abi, size;
    const char* plugin_id;
    const char* backend;
    uint32_t ranks, phases;
    void (*retain)(void* owner);
    void (*release)(void* owner);
    int (*bind_phase)(void* owner, uint32_t phase);
    int (*enter)(void* owner, uint32_t phase, uint32_t rank, void** stream);
    // Retires all rank work, checks provider faults and clears provider TLS.
    int (*leave)(void* owner, uint32_t rank);
    const char* (*status_json)(void* owner);
};
struct GarnetOperatorExecutionPayload {
    uint32_t abi, size;
    const GarnetOperatorExecutionServices* services;
    void* owner;
};
inline constexpr const char* GarnetOperatorExecutionType="garnet.operator_execution_group.v1";
inline constexpr const char* GarnetOperatorExecutionSymbol="GarnetOperatorExecutionServicesV1";
