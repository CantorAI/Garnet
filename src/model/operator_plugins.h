// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "nlohmann/json.hpp"
#include "xlang3/xlang3.h"
#include <string>
#include <vector>
namespace Garnet {
bool BindOperatorModule(const X::Value& module, std::string& error);
// An empty requirements array is a no-op and never opens the plugin registry.
bool ResolveOperatorPlugins(X3PackageHost* host, const nlohmann::json& requirements, const std::string& backend,
    nlohmann::json& resolved, std::string& error);
bool ValidateCachedOperatorPlugins(X3PackageHost* host, const nlohmann::json& resolved, const std::string& backend,
    bool& unchanged, std::string& error);
void* OperatorPluginSymbol(const std::string& id, const char* symbol, std::string& error);
bool ValidateOperatorPluginUse(const std::vector<std::string>& operations,
    const nlohmann::json& requirements, std::string& error);
}
