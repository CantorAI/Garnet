// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#if defined(_WIN32)
#if defined(GARNET_SERVING_EXPORTS)
#define GARNET_SERVING_API __declspec(dllexport)
#else
#define GARNET_SERVING_API __declspec(dllimport)
#endif
#else
#define GARNET_SERVING_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Returns UTF-8 JSON. requiredCapacity includes the trailing NUL. Passing a
// null output buffer, or a buffer that is too small, returns 2 after reporting
// the required capacity. catalogRoot may be null to use Garnet discovery.
GARNET_SERVING_API int GarnetListAvailableModelsJson(
    const char* catalogRoot,
    char* output,
    int outputCapacity,
    int* requiredCapacity);

// Enumerates live model-serving instances owned by this Garnet runtime.
GARNET_SERVING_API int GarnetListLoadedModelsJson(
    char* output,
    int outputCapacity,
    int* requiredCapacity);

#ifdef __cplusplus
}
#endif
