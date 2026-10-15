// SPDX-License-Identifier: Apache-2.0
#include "operator_plugins.h"
#include "operator_module_bridge.h"
#include "md5.h"
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <cstdlib>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace Garnet {
namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;
struct Library { const GarnetOperatorModuleBridge* bridge; X::Value module; X::Value owner; json manifest; std::string digest; };
std::mutex mutex;
// Backend creators can outlive runtime teardown. Keep native owners until process
// exit; destroying X::Value here after the XLang host is gone is unsafe.
// Native module unloading/rebinding across runtimes is not supported.
auto& loaded = *new std::unordered_map<std::string, Library>();
fs::path runtimeFolder() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&runtimeFolder), &module)) return {};
    wchar_t path[32768]; DWORD n = GetModuleFileNameW(module, path, 32768);
    return n && n < 32768 ? fs::path(path).parent_path() : fs::path();
#else
    Dl_info info{};
    return dladdr(reinterpret_cast<void*>(&runtimeFolder), &info) && info.dli_fname
        ? fs::path(info.dli_fname).parent_path() : fs::path();
#endif
}
std::string binaryDigest(const GarnetOperatorModuleBridge* bridge) {
    const char* path = bridge->binary_path();
    if (!path || !*path) throw std::runtime_error("operator module returned no binary path");
    std::ifstream binary(fs::u8path(path), std::ios::binary);
    if (!binary) throw std::runtime_error("operator module binary missing: " + std::string(path));
    std::string bytes((std::istreambuf_iterator<char>(binary)), {});
    return MD5(bytes).hexdigest();
}
Library& bind(const X::Value& module) {
    auto* host = module.host();
    if (!host || !host->instance_get_native_data) throw std::runtime_error("operator module requires a native host");
    auto method = module["operator_bridge"]; X::Value owner;
    if (!method.Call({}, owner)) throw std::runtime_error("module does not expose operator_bridge");
    auto* bridge = static_cast<const GarnetOperatorModuleBridge*>(host->instance_get_native_data(owner.raw(), GarnetOperatorBridgeType));
    if (!bridge || bridge->abi != 1 || bridge->size != sizeof(*bridge) || !bridge->manifest ||
        !bridge->register_backend || !bridge->resolve_callback || !bridge->binary_path)
        throw std::runtime_error("operator module bridge ABI mismatch");
    const char* description = bridge->manifest();
    if (!description) throw std::runtime_error("operator module returned no manifest");
    json manifest = json::parse(description);
    const std::string id = manifest.at("id");
    if (id.empty() || manifest.at("abi") != 1 || !manifest.at("module").is_string() ||
        manifest.at("module").get<std::string>().empty() || !manifest.at("operators").is_array() ||
        !manifest.at("version").is_string() || !manifest.at("backend").is_string())
        throw std::runtime_error("operator module manifest invalid");
    const auto digest = binaryDigest(bridge);
    auto found = loaded.find(id);
    if (found != loaded.end()) {
        if (found->second.bridge != bridge || found->second.digest != digest)
            throw std::runtime_error("operator module identity conflict or binary changed; restart process: " + id);
        return found->second;
    }
    if (!bridge->register_backend()) throw std::runtime_error("operator module backend registration failed: " + id);
    return loaded.emplace(id, Library{bridge, module, owner, manifest, digest}).first->second;
}
Library& load(X3PackageHost* host, const std::string& id, const std::string& moduleName = {}) {
    auto found = loaded.find(id);
    if (found != loaded.end()) {
        if (binaryDigest(found->second.bridge) != found->second.digest)
            throw std::runtime_error("operator module binary changed; restart process: " + id);
        return found->second;
    }
    if (!moduleName.empty()) {
        X::Module module(host, moduleName.c_str());
        auto& library = bind(module);
        if (library.manifest.at("id") != id || library.manifest.at("module") != moduleName)
            throw std::runtime_error("operator module identity mismatch: " + id);
        return library;
    }
    const char* custom = std::getenv("GARNET_OPERATOR_PLUGIN_REGISTRY");
    fs::path registryPath = custom ? fs::path(custom) : runtimeFolder() / "plugins" / "registry.json";
    std::ifstream stream(registryPath); json registry;
    if (!stream) throw std::runtime_error("operator plugin registry missing: " + registryPath.string());
    stream >> registry;
    if (registry.at("schema_version") != 1) throw std::runtime_error("unsupported operator plugin registry schema");
    if (!registry.at("plugins").contains(id)) throw std::runtime_error("operator plugin is not installed: " + id);
    const std::string name = registry.at("plugins").at(id).at("module");
    X::Module module(host, name.c_str());
    auto& library = bind(module);
    if (library.manifest.at("id") != id) throw std::runtime_error("operator module identity mismatch: " + id);
    return library;
}
}
bool BindOperatorModule(const X::Value& module, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex);
    try { bind(module); error.clear(); return true; }
    catch (const std::exception& e) { error = e.what(); return false; }
}
bool ResolveOperatorPlugins(X3PackageHost* host, const nlohmann::json& requirements, const std::string& backend,
    nlohmann::json& resolved, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex); resolved = json::array();
    try {
        if (!requirements.is_array()) throw std::runtime_error("requires.operator_plugins must be a list");
        std::set<std::string> ids;
        for (const auto& requirement : requirements) {
            if (!requirement.is_object() || !requirement.contains("id") || !requirement.at("id").is_string() ||
                !requirement.contains("abi") || !requirement.at("abi").is_number_integer() ||
                !requirement.contains("backend") || !requirement.at("backend").is_string() ||
                !requirement.contains("operators") || !requirement.at("operators").is_array())
                throw std::runtime_error("invalid operator plugin requirement");
            const std::string id = requirement.at("id");
            if (id.empty() || !ids.insert(id).second) throw std::runtime_error("empty/duplicate operator plugin id: " + id);
            if (requirement.at("abi") != 1 || requirement.at("backend") != backend)
                throw std::runtime_error("operator plugin requirement ABI/backend mismatch: " + id);
            if (requirement.contains("module") && !requirement.at("module").is_string())
                throw std::runtime_error("operator module name must be a string: " + id);
            const std::string moduleName = requirement.value("module", std::string());
            auto& library = load(host, id, moduleName);
            if (!moduleName.empty() && library.manifest.at("module") != moduleName)
                throw std::runtime_error("operator module name mismatch: " + id);
            if (library.manifest.at("backend") != backend) throw std::runtime_error("operator plugin does not support backend: " + id);
            const auto provided = library.manifest.at("operators").get<std::vector<std::string>>();
            for (const auto& op : requirement.at("operators")) {
                if (!op.is_string() || std::find(provided.begin(), provided.end(), op.get<std::string>()) == provided.end())
                    throw std::runtime_error("operator plugin is missing required operator: " + id + ": " + op.dump());
            }
            json snapshot = requirement;
            snapshot["module"] = library.manifest.at("module");
            snapshot["library_digest"] = library.digest;
            snapshot["plugin_version"] = library.manifest.at("version");
            resolved.push_back(snapshot);
        }
        error.clear(); return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
bool ValidateCachedOperatorPlugins(X3PackageHost* host, const nlohmann::json& cached, const std::string& backend,
    bool& unchanged, std::string& error) {
    json current;
    if (!ResolveOperatorPlugins(host, cached, backend, current, error)) return false;
    unchanged = current == cached;
    return true;
}
void* OperatorPluginSymbol(const std::string& id, const char* name, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = loaded.find(id);
    if (found == loaded.end()) { error = "operator plugin was not declared/loaded: " + id; return nullptr; }
    void* value = found->second.bridge->resolve_callback(name);
    if (!value) error = "operator plugin symbol missing: " + id + ": " + name;
    return value;
}
bool ValidateOperatorPluginUse(const std::vector<std::string>& operations,
    const nlohmann::json& requirements, std::string& error) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& library : loaded) {
        const auto provided = library.second.manifest.at("operators").get<std::vector<std::string>>();
        for (const auto& op : operations) {
            if (std::find(provided.begin(), provided.end(), op) == provided.end()) continue;
            bool declared = false;
            for (const auto& req : requirements) {
                if (req.at("id") != library.first) continue;
                const auto names = req.at("operators").get<std::vector<std::string>>();
                declared = std::find(names.begin(), names.end(), op) != names.end();
            }
            if (!declared) { error = "xModel uses an undeclared plugin operator: " + op; return false; }
        }
    }
    return true;
}
}
