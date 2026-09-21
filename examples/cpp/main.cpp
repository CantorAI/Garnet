// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#include "xlang3/xlang3.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::string Env(const char* name, const std::string& fallback = {}) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : fallback;
}

X::Value Call(X::Runtime& runtime, const X::Value& function,
              const std::vector<X::Value>& arguments = {}) {
    X::Value result;
    if (!function.Call(arguments, result)) {
        throw std::runtime_error(runtime.LastError());
    }
    return result;
}

int main() {
    try {
        X::Runtime runtime;
        const auto runtimeRoot = Env("GARNET_RUNTIME_ROOT", fs::current_path().string());
        runtime.AddImportRoot(runtimeRoot);
        X::Module garnet(runtime, "garnet");

        const auto accelerationRoot = Env("GARNET_ACCELERATION_ROOT");
        if (!accelerationRoot.empty()) {
            std::cout << Call(runtime, garnet["activate_acceleration_path_json"],
                              {X::Value(runtime, accelerationRoot)}).ToString() << '\n';
        }
        std::cout << Call(runtime, garnet["detect_acceleration_json"]).ToString() << '\n';

        const auto modelRoot = Env("GARNET_MODEL_ROOT");
        if (modelRoot.empty()) {
            std::cout << "Set GARNET_MODEL_ROOT to run real Qwen3-1.7B inference.\n";
            return 0;
        }

        const auto xmodelRoot = Env("GARNET_XMODEL_ROOT", (fs::path(modelRoot) / "xmodel").string());
        const auto cacheRoot = Env("GARNET_CACHE_ROOT", (fs::path(modelRoot) / "compiled_cache").string());
        std::cout << Call(runtime, garnet["serve_model"], {
            X::Value(runtime, modelRoot), X::Value(runtime, xmodelRoot),
            X::Value(runtime, cacheRoot), X::Value(runtime, "{}"),
            X::Value(runtime, "Qwen3-1.7B")}).ToString() << '\n';
        try {
            const auto prompt = Env("GARNET_PROMPT", "Explain Garnet in one sentence.");
            std::cout << Call(runtime, garnet["infer_json"], {
                X::Value(runtime, prompt), X::Value(runtime, ""), X::Value(runtime, 64)
            }).ToString() << '\n';
        } catch (...) {
            Call(runtime, garnet["stop_serving"]);
            throw;
        }
        Call(runtime, garnet["stop_serving"]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
