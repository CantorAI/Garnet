#include "acceleration_detector.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>
#include <utility>

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

#if defined(_WIN32)
        using Library = HMODULE;
        constexpr Library MissingLibrary = nullptr;
        Library OpenDriver()
        {
            return LoadLibraryExW(
                L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        }
        void* Symbol(Library library, const char* name)
        {
            return reinterpret_cast<void*>(GetProcAddress(library, name));
        }
        void CloseDriver(Library library) { FreeLibrary(library); }
#define GARNET_CUDA_CALL __stdcall
#else
        using Library = void*;
        constexpr Library MissingLibrary = nullptr;
        Library OpenDriver() { return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL); }
        void* Symbol(Library library, const char* name) { return dlsym(library, name); }
        void CloseDriver(Library library) { dlclose(library); }
#define GARNET_CUDA_CALL
#endif

        using CuInit = int (GARNET_CUDA_CALL*)(unsigned int);
        using CuDriverGetVersion = int (GARNET_CUDA_CALL*)(int*);
        using CuDeviceGetCount = int (GARNET_CUDA_CALL*)(int*);
        using CuDeviceGet = int (GARNET_CUDA_CALL*)(int*, int);
        using CuDeviceGetName = int (GARNET_CUDA_CALL*)(char*, int, int);
        using CuDeviceGetAttribute = int (GARNET_CUDA_CALL*)(int*, int, int);

        constexpr int Success = 0;
        constexpr int ComputeCapabilityMajor = 75;
        constexpr int ComputeCapabilityMinor = 76;

        std::mutex g_activationMutex;
        std::unordered_set<std::string> g_activatedPackages;
#if defined(_WIN32)
        std::vector<HMODULE> g_accelerationLibraries;
        std::vector<DLL_DIRECTORY_COOKIE> g_accelerationDirectories;
#else
        std::vector<void*> g_accelerationLibraries;
#endif

        json LazyLoadContract()
        {
#if defined(_WIN32)
            const json libraries = {
                "cublas64_13.dll", "nvinfer_10.dll",
                "nppig64_13.dll", "nvjpeg64_13.dll"
            };
            return {
                {"enabled", true},
                {"driver_probe_only", true},
                {"libraries", libraries}
            };
#else
            const json libraries = {
                "libcublas.so.13", "libnvinfer.so.10",
                "libnppig.so.13", "libnvjpeg.so.13"
            };
            return {
                {"enabled", false},
                {"driver_probe_only", true},
                {"libraries", libraries},
                {"reason", "Linux Garnet must be split into a CPU base and optional GPU plugin before these libraries are dependency-free"}
            };
#endif
        }
    }

    std::string AccelerationDetector::PlatformId()
    {
#if defined(_WIN32)
        constexpr const char* os = "windows";
#elif defined(__linux__)
        constexpr const char* os = "linux";
#elif defined(__APPLE__)
        constexpr const char* os = "macos";
#else
        constexpr const char* os = "unknown";
#endif

#if defined(_M_X64) || defined(__x86_64__)
        constexpr const char* arch = "x64";
#elif defined(_M_ARM64) || defined(__aarch64__)
        constexpr const char* arch = "arm64";
#else
        constexpr const char* arch = "unknown";
#endif
        return std::string(os) + "-" + arch;
    }

    std::string AccelerationDetector::DetectJson(bool probeNvidia)
    {
        json response = {
            {"schema_version", 1},
            {"status", "ok"},
            {"platform", PlatformId()},
            {"cpu", {{"backend", "openvino"}, {"ready", true}}},
            {"lazy_loading", LazyLoadContract()},
            {"nvidia", {
                {"detected", false},
                {"driver_available", false},
                {"driver_api_version", 0},
                {"devices", json::array()}
            }}
        };

        if (!probeNvidia) {
            response["nvidia"]["disabled_by_policy"] = true;
            return response.dump();
        }

        Library library = OpenDriver();
        if (library == MissingLibrary) return response.dump();
        response["nvidia"]["driver_available"] = true;

        const auto cuInit = reinterpret_cast<CuInit>(Symbol(library, "cuInit"));
        const auto driverVersion = reinterpret_cast<CuDriverGetVersion>(
            Symbol(library, "cuDriverGetVersion"));
        const auto deviceCount = reinterpret_cast<CuDeviceGetCount>(
            Symbol(library, "cuDeviceGetCount"));
        const auto deviceGet = reinterpret_cast<CuDeviceGet>(
            Symbol(library, "cuDeviceGet"));
        const auto deviceName = reinterpret_cast<CuDeviceGetName>(
            Symbol(library, "cuDeviceGetName"));
        const auto deviceAttribute = reinterpret_cast<CuDeviceGetAttribute>(
            Symbol(library, "cuDeviceGetAttribute"));

        if (!cuInit || !driverVersion || !deviceCount || !deviceGet ||
            !deviceName || !deviceAttribute) {
            response["nvidia"]["error"] = "NVIDIA driver API is incomplete";
            CloseDriver(library);
            return response.dump();
        }

        int version = 0;
        if (driverVersion(&version) == Success) {
            response["nvidia"]["driver_api_version"] = version;
        }
        if (cuInit(0) != Success) {
            response["nvidia"]["error"] = "NVIDIA driver initialization failed";
            CloseDriver(library);
            return response.dump();
        }

        int count = 0;
        if (deviceCount(&count) != Success) count = 0;
        if (count < 0) count = 0;
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            int device = 0;
            if (deviceGet(&device, ordinal) != Success) continue;
            std::array<char, 256> name{};
            int major = 0;
            int minor = 0;
            deviceName(name.data(), static_cast<int>(name.size()), device);
            deviceAttribute(&major, ComputeCapabilityMajor, device);
            deviceAttribute(&minor, ComputeCapabilityMinor, device);
            response["nvidia"]["devices"].push_back({
                {"ordinal", ordinal},
                {"name", std::string(name.data())},
                {"compute_capability", std::to_string(major) + "." +
                    std::to_string(minor)}
            });
        }
        response["nvidia"]["detected"] =
            !response["nvidia"]["devices"].empty();
        CloseDriver(library);
        return response.dump();
    }

    std::string AccelerationDetector::ActivateJson(
        const std::filesystem::path& packageRoot)
    {
        namespace fs = std::filesystem;
        try {
            const fs::path root = fs::absolute(packageRoot).lexically_normal();
            const fs::path marker = root / ".garnet-model.json";
            std::ifstream stream(marker, std::ios::binary);
            if (!stream) throw std::runtime_error("acceleration manifest is missing");
            json manifest = json::parse(stream);
            const json libraries = manifest.value("runtime_libraries", json::array());
            if (!libraries.is_array() || libraries.empty()) {
                throw std::runtime_error("runtime_libraries is missing from the acceleration manifest");
            }

            std::vector<std::pair<fs::path, fs::path>> resolvedLibraries;
            for (const auto& item : libraries) {
                if (!item.is_string()) throw std::runtime_error("runtime library path is invalid");
                const fs::path relative(item.get<std::string>());
                if (relative.empty() || relative.is_absolute()) {
                    throw std::runtime_error("runtime library path must be relative");
                }
                const fs::path full = fs::absolute(root / relative).lexically_normal();
                const fs::path check = full.lexically_relative(root);
                if (check.empty() || check.is_absolute() ||
                    *check.begin() == ".." || !fs::is_regular_file(full)) {
                    throw std::runtime_error("runtime library escaped the acceleration package");
                }
                resolvedLibraries.emplace_back(relative, full);
            }

            std::lock_guard<std::mutex> guard(g_activationMutex);
            const std::string activationKey = root.generic_string();
            if (g_activatedPackages.count(activationKey) != 0) {
                return json({
                    {"schema_version", 1},
                    {"status", "ready"},
                    {"backend", "nvidia"},
                    {"package_root", root.string()},
                    {"already_active", true}
                }).dump();
            }
            json loaded = json::array();
            std::vector<Library> packageHandles;
#if defined(_WIN32)
            std::vector<DLL_DIRECTORY_COOKIE> packageDirectories;
            std::unordered_set<std::string> registeredDirectories;
#endif
            for (const auto& resolved : resolvedLibraries) {
                const fs::path& relative = resolved.first;
                const fs::path& full = resolved.second;
#if defined(_WIN32)
                const std::string directoryKey = full.parent_path().generic_string();
                if (registeredDirectories.insert(directoryKey).second) {
                    const DLL_DIRECTORY_COOKIE cookie = AddDllDirectory(
                        full.parent_path().wstring().c_str());
                    if (!cookie) {
                        for (auto directory = packageDirectories.rbegin();
                             directory != packageDirectories.rend(); ++directory) {
                            RemoveDllDirectory(*directory);
                        }
                        throw std::runtime_error(
                            "failed to register acceleration library directory: " +
                            full.parent_path().string());
                    }
                    packageDirectories.push_back(cookie);
                }
                HMODULE handle = LoadLibraryExW(
                    full.wstring().c_str(), nullptr,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
                void* handle = dlopen(full.string().c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
                if (!handle) {
#if defined(_WIN32)
                    for (auto loadedHandle = packageHandles.rbegin();
                         loadedHandle != packageHandles.rend(); ++loadedHandle) {
                        FreeLibrary(*loadedHandle);
                    }
                    for (auto directory = packageDirectories.rbegin();
                         directory != packageDirectories.rend(); ++directory) {
                        RemoveDllDirectory(*directory);
                    }
#else
                    for (auto loadedHandle = packageHandles.rbegin();
                         loadedHandle != packageHandles.rend(); ++loadedHandle) {
                        dlclose(*loadedHandle);
                    }
#endif
                    throw std::runtime_error(
                        "failed to load acceleration library: " + full.string());
                }
                packageHandles.push_back(handle);
                loaded.push_back(relative.generic_string());
            }
            g_accelerationLibraries.insert(
                g_accelerationLibraries.end(),
                packageHandles.begin(), packageHandles.end());
#if defined(_WIN32)
            g_accelerationDirectories.insert(
                g_accelerationDirectories.end(),
                packageDirectories.begin(), packageDirectories.end());
#endif
            g_activatedPackages.insert(activationKey);
            return json({
                {"schema_version", 1},
                {"status", "ready"},
                {"backend", "nvidia"},
                {"package_root", root.string()},
                {"loaded_libraries", loaded}
            }).dump();
        }
        catch (const std::exception& exception) {
            return json({
                {"schema_version", 1},
                {"status", "error"},
                {"error", "acceleration_activation_failed"},
                {"message", exception.what()}
            }).dump();
        }
    }
}
