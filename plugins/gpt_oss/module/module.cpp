// SPDX-License-Identifier: Apache-2.0
#include "xlang3/xlang3.h"
#include "operator_module_bridge.h"
#include "gpt_oss_peer_group.h"
#include "NvInfer.h"
#include <cstring>
#include <string>
#include <memory>
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
    if(name && std::strcmp(name,GarnetOperatorExecutionSymbol)==0)
        return const_cast<GarnetOperatorExecutionServices*>(Garnet::GptOssPeerExecutionServices());
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
    static void CleanupGroup(void* pointer){
        auto* p=static_cast<GarnetOperatorExecutionPayload*>(pointer);
        if(p->owner)p->services->release(p->owner);delete p;
    }
    static GarnetOperatorExecutionPayload* Group(const X::Value& value,bool allowReleased=false){
        auto* host=value.host();
        auto* p=host && host->instance_get_native_data?static_cast<GarnetOperatorExecutionPayload*>(
            host->instance_get_native_data(value.raw(),GarnetOperatorExecutionType)):nullptr;
        if(!p || p->abi!=1 || p->size!=sizeof(*p) || (!allowReleased && !p->owner) ||
           p->services!=Garnet::GptOssPeerExecutionServices())throw X::Error("GPT-OSS native execution group required");
        return p;
    }
public:
    BEGIN_PACKAGE(GptOssModule)
        APISET().AddFunc<0>("operator_bridge", &GptOssModule::OperatorBridge);
        APISET().AddFunc<0>("manifest_json", &GptOssModule::Manifest);
        APISET().AddVarFunc("peer_group", &GptOssModule::PeerGroup);
        APISET().AddVarFunc("peer_group_release", &GptOssModule::ReleaseGroup);
        APISET().AddVarFunc("peer_group_bind_phase", &GptOssModule::BindPhase);
        APISET().AddVarFunc("peer_group_status_json", &GptOssModule::GroupStatus);
    END_PACKAGE
    std::string Manifest() { return GarnetOperatorPluginManifest(); }
    X::Value PeerGroup(const X::ARGS& args,const X::KWARGS&){
        if(args.size()!=1 || !args[0].IsString())throw X::Error("peer_group(options_json) expected");
        const auto options=args[0].ToString();void* owner=nullptr;
        const auto status=Garnet::GptOssCreatePeerGroup(options.data(),options.size(),&owner);
        if(status!=cudaSuccess)throw X::Error(std::string("peer group creation failed: ")+cudaGetErrorString(status));
        auto* services=Garnet::GptOssPeerExecutionServices();
        std::unique_ptr<void,void(*)(void*)> hold(owner,services->release);
        auto payload=std::make_unique<GarnetOperatorExecutionPayload>(GarnetOperatorExecutionPayload{1,sizeof(GarnetOperatorExecutionPayload),services,owner});
        auto* host=Host();X3Value klass=x3_value_invalid();
        if(host->create_class(host,"OperatorExecutionGroup",nullptr,0,&klass)!=X3_STATUS_OK)throw X::Error("cannot create execution group type");
        X::Value type(host,klass,false),value(host,host->value_instance(host->runtime,klass),false);
        if(host->instance_set_native_data(value.raw(),GarnetOperatorExecutionType,payload.get(),&CleanupGroup)!=X3_STATUS_OK){
            throw X::Error("cannot attach execution group payload");}
        hold.release();payload.release();return value;
    }
    X::Value ReleaseGroup(const X::ARGS& args,const X::KWARGS&){
        if(args.size()!=1)throw X::Error("peer_group_release(group) expected");
        auto* p=Group(args[0],true);
        // Serialized with request submission. Scoped/cached native owners keep
        // their own references; VM-frame temporaries no longer own this resource.
        auto* owner=p->owner;p->owner=nullptr;
        if(owner)p->services->release(owner);
        return X::Value(true);
    }
    X::Value BindPhase(const X::ARGS& args,const X::KWARGS&){
        if(args.size()!=2 || !args[1].IsInt64())throw X::Error("peer_group_bind_phase(group, phase) expected");
        const auto phase=args[1].ToLongLong();if(phase<0 || phase>1)throw X::Error("peer group phase out of range");
        auto* p=Group(args[0]);const int status=p->services->bind_phase(p->owner,uint32_t(phase));
        if(status)throw X::Error(std::string("peer group phase binding failed: ")+cudaGetErrorString(cudaError_t(status)));
        return X::Value(true);
    }
    X::Value GroupStatus(const X::ARGS& args,const X::KWARGS&){
        if(args.size()!=1)throw X::Error("peer_group_status_json(group) expected");
        auto* p=Group(args[0]);return X::Value::String(Host(),p->services->status_json(p->owner));
    }
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
