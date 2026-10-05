// SPDX-License-Identifier: Apache-2.0
#include "xlang3/xlang3.h"
#include "operator_module_bridge.h"
#include "NvInfer.h"
#include <cstring>
#include <string>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
extern "C" const char* GarnetOperatorPluginManifest();
extern "C" int GarnetRegisterOperatorPlugin();
extern "C" nvinfer1::IPluginV2DynamicExt* GarnetCreateOperatorPlugin(const void*, size_t);
namespace {
void* Resolve(const char* name) {
    if (name && std::strcmp(name, "GarnetCreateOperatorPlugin") == 0)
        return reinterpret_cast<void*>(&GarnetCreateOperatorPlugin);
    return nullptr;
}
const char* BinaryPath() {
    static const std::string path = [] {
#if defined(_WIN32)
        HMODULE module = nullptr;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&BinaryPath), &module)) return std::string();
        char filename[32768]{};
        DWORD length = GetModuleFileNameA(module, filename, sizeof(filename));
        return length && length < sizeof(filename) ? std::string(filename, length) : std::string();
#else
        Dl_info info{};
        return dladdr(reinterpret_cast<void*>(&BinaryPath), &info) && info.dli_fname
            ? std::string(info.dli_fname) : std::string();
#endif
    }();
    return path.c_str();
}
GarnetOperatorModuleBridge bridge{1, sizeof(GarnetOperatorModuleBridge),
    &GarnetOperatorPluginManifest, &GarnetRegisterOperatorPlugin, &Resolve, &BinaryPath};
class GptOssModule {
public:
    BEGIN_PACKAGE(GptOssModule)
        APISET().AddFunc<0>("operator_bridge", &GptOssModule::OperatorBridge);
        APISET().AddFunc<0>("manifest_json", &GptOssModule::Manifest);
    END_PACKAGE
    std::string Manifest() { return GarnetOperatorPluginManifest(); }
    X::Value OperatorBridge() {
        auto* host = Host(); X3Value klass = x3_value_invalid();
        if (host->create_class(host, "OperatorBridge", nullptr, 0, &klass) != X3_STATUS_OK)
            throw X::Error("cannot create operator bridge");
        X::Value type(host, klass, false);
        X::Value instance(host, host->value_instance(host->runtime, klass), false);
        if (host->instance_set_native_data(instance.raw(), GarnetOperatorBridgeType, &bridge, nullptr) != X3_STATUS_OK)
            throw X::Error("cannot attach operator bridge");
        return instance;
    }
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version = X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* pointer, X3Value module) {
    auto* host = static_cast<X3PackageHost*>(pointer);
    if (!host || host->abi_version != X3_ABI_VERSION) return X3_STATUS_ERROR;
    try { GptOssModule::BuildAPI(); return GptOssModule::APISET().Create(host, "garnet_gpt_oss", module); }
    catch (...) { return X3_STATUS_ERROR; }
}
