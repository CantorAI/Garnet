#include "model_catalog.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <set>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Garnet
{
    namespace
    {
        using json = nlohmann::json;
        namespace fs = std::filesystem;

        fs::path AbsoluteNormalized(const fs::path& path)
        {
            std::error_code error;
            fs::path absolute = fs::absolute(path, error);
            return (error ? path : absolute).lexically_normal();
        }

        bool IsSafePackagePath(const fs::path& relative)
        {
            if (relative.empty() || relative.is_absolute()) return false;
            for (const auto& component : relative) {
                if (component == "..") return false;
            }
            return true;
        }

        bool ReadJsonFile(const fs::path& path, json& value, std::string& error)
        {
            try {
                std::ifstream stream(path, std::ios::binary);
                if (!stream) {
                    error = "cannot open " + path.string();
                    return false;
                }
                stream >> value;
                return true;
            }
            catch (const std::exception& exception) {
                error = exception.what();
                return false;
            }
        }

        bool ValidateReferencedFile(
            const fs::path& packageRoot,
            const std::string& relativeText,
            std::vector<std::string>& issues)
        {
            const fs::path relative(relativeText);
            if (!IsSafePackagePath(relative)) {
                issues.push_back("unsafe package-relative path: " + relativeText);
                return false;
            }
            if (!fs::is_regular_file(packageRoot / relative)) {
                issues.push_back("missing package file: " + relativeText);
                return false;
            }
            return true;
        }

        json InspectManifest(const fs::path& manifestPath)
        {
            const fs::path packageRoot = manifestPath.parent_path();
            std::vector<std::string> issues;
            json manifest;
            std::string readError;
            if (!ReadJsonFile(manifestPath, manifest, readError) ||
                !manifest.is_object()) {
                return {
                    {"id", ""},
                    {"available", false},
                    {"manifest_path", manifestPath.string()},
                    {"package_root", packageRoot.string()},
                    {"issues", json::array({"invalid manifest: " + readError})}
                };
            }

            if (manifest.value("schema_version", 0) != 1) {
                issues.push_back("unsupported or missing schema_version");
            }
            if (!manifest.contains("id") || !manifest["id"].is_string() ||
                manifest["id"].get<std::string>().empty()) {
                issues.push_back("id must be a non-empty string");
            }
            if (!manifest.contains("task") || !manifest["task"].is_string()) {
                issues.push_back("task must be a string");
            }
            if (!manifest.contains("entrypoints") ||
                !manifest["entrypoints"].is_object()) {
                issues.push_back("entrypoints must be an object");
            }
            else {
                for (const auto& item : manifest["entrypoints"].items()) {
                    const json& entrypoint = item.value();
                    if (!entrypoint.is_object() ||
                        !entrypoint.contains("file") ||
                        !entrypoint["file"].is_string() ||
                        !entrypoint.contains("function") ||
                        !entrypoint["function"].is_string()) {
                        issues.push_back("invalid entrypoint: " + item.key());
                        continue;
                    }
                    ValidateReferencedFile(
                        packageRoot,
                        entrypoint["file"].get<std::string>(),
                        issues);
                }
            }

            json resolvedProfiles = json::array();
            if (!manifest.contains("profile_files") ||
                !manifest["profile_files"].is_array()) {
                issues.push_back("profile_files must be an array");
            }
            else {
                for (const json& profileValue : manifest["profile_files"]) {
                    if (!profileValue.is_string()) {
                        issues.push_back("profile file path must be a string");
                        continue;
                    }
                    const std::string relativeText = profileValue.get<std::string>();
                    if (!ValidateReferencedFile(packageRoot, relativeText, issues)) {
                        continue;
                    }
                    json profile;
                    std::string profileError;
                    if (!ReadJsonFile(packageRoot / relativeText, profile, profileError) ||
                        !profile.is_object()) {
                        issues.push_back(
                            "invalid profile " + relativeText + ": " + profileError);
                        continue;
                    }
                    if (profile.value("schema_version", 0) != 1 ||
                        !profile.contains("id") || !profile["id"].is_string() ||
                        !profile.contains("backend") ||
                        !profile["backend"].is_string() ||
                        !profile.contains("precision") ||
                        !profile["precision"].is_string()) {
                        issues.push_back("profile has an invalid contract: " + relativeText);
                        continue;
                    }
                    profile["profile_path"] = (packageRoot / relativeText).string();
                    resolvedProfiles.push_back(std::move(profile));
                }
            }

            manifest["manifest_path"] = manifestPath.string();
            manifest["package_root"] = packageRoot.string();
            manifest["profiles"] = std::move(resolvedProfiles);
            manifest["available"] = issues.empty();
            manifest["issues"] = issues;
            return manifest;
        }
    }

    std::filesystem::path RuntimeModuleFolder()
    {
#if defined(_WIN32)
        HMODULE module = nullptr;
        if (!GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCSTR>(&RuntimeModuleFolder),
                &module)) {
            return {};
        }
        std::vector<char> path(1024);
        for (;;) {
            const DWORD length = GetModuleFileNameA(
                module, path.data(), static_cast<DWORD>(path.size()));
            if (length == 0) return {};
            if (length < path.size() - 1) {
                return fs::path(std::string(path.data(), length)).parent_path();
            }
            path.resize(path.size() * 2);
        }
#else
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&RuntimeModuleFolder), &info) == 0 ||
            !info.dli_fname) {
            return {};
        }
        return fs::path(info.dli_fname).parent_path();
#endif
    }

    std::filesystem::path ResolveModelCatalogRoot(
        const std::string& requestedRoot,
        const std::string& runtimeBaseFolder)
    {
        namespace fs = std::filesystem;
        if (!requestedRoot.empty()) return AbsoluteNormalized(requestedRoot);
        if (const char* configured = std::getenv("GARNET_MODEL_CATALOG");
            configured && *configured) {
            return AbsoluteNormalized(configured);
        }

        fs::path base = runtimeBaseFolder.empty()
            ? RuntimeModuleFolder()
            : fs::path(runtimeBaseFolder);
        if (base.empty()) base = fs::current_path();
        const fs::path sibling = AbsoluteNormalized(base / ".." / "models");
        if (fs::is_directory(sibling)) return sibling;
        const fs::path adjacent = AbsoluteNormalized(base / "models");
        if (fs::is_directory(adjacent)) return adjacent;
        return sibling;
    }

    std::string EnumerateAvailableModelsJson(
        const std::filesystem::path& requestedCatalogRoot)
    {
        namespace fs = std::filesystem;
        const fs::path catalogRoot = AbsoluteNormalized(requestedCatalogRoot);
        json response = {
            {"schema_version", 1},
            {"catalog_root", catalogRoot.string()},
            {"models", json::array()},
            {"errors", json::array()}
        };

        if (!fs::is_directory(catalogRoot)) {
            response["errors"].push_back({
                {"code", "catalog_not_found"},
                {"message", "model catalog directory was not found"}
            });
            return response.dump();
        }

        std::vector<fs::path> manifests;
        std::error_code walkError;
        for (fs::recursive_directory_iterator iterator(
                 catalogRoot,
                 fs::directory_options::skip_permission_denied,
                 walkError), end;
             !walkError && iterator != end;
             iterator.increment(walkError)) {
            if (iterator->is_regular_file() &&
                iterator->path().filename() == "model.json") {
                manifests.push_back(AbsoluteNormalized(iterator->path()));
            }
        }
        if (walkError) {
            response["errors"].push_back({
                {"code", "catalog_scan_failed"},
                {"message", walkError.message()}
            });
        }
        std::sort(manifests.begin(), manifests.end());

        std::set<std::string> ids;
        for (const fs::path& manifestPath : manifests) {
            json model = InspectManifest(manifestPath);
            const std::string id = model.value("id", "");
            if (!id.empty() && !ids.insert(id).second) {
                model["available"] = false;
                model["issues"].push_back("duplicate model id: " + id);
            }
            response["models"].push_back(std::move(model));
        }
        std::sort(
            response["models"].begin(),
            response["models"].end(),
            [](const json& left, const json& right) {
                return left.value("id", "") < right.value("id", "");
            });
        return response.dump();
    }
}
