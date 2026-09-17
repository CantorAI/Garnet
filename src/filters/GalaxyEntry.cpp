#include "xlang3/xlang3.h"
#include "native_filter.h"
#include "GarnetVLMFilter.h"
#include <iostream>

namespace {
bool ValidHost(X3PackageHost* host) {
    return host && host->runtime && host->abi_version == X3_ABI_VERSION && host->size >= sizeof(*host);
}
}

extern "C" GALAXY_FILTER_EXPORT const GalaxyFilterAbi galaxy_filter_abi = {
    sizeof(GalaxyFilterAbi), GALAXY_FILTER_ABI_VERSION, X3_ABI_VERSION};

extern "C" GALAXY_FILTER_EXPORT X3Status GLoad(const char* library, const char* filter,
    void* pointer, void* factoryPointer, X3Value* result) {
    if (!result) return X3_STATUS_ERROR;
    *result = x3_value_invalid();
    auto* host = static_cast<X3PackageHost*>(pointer);
    auto* factory = static_cast<Galaxy::IFactory*>(factoryPointer);
    if (!ValidHost(host) || !factory || !library || !filter || factory->Host() != host)
        return X3_STATUS_ERROR;
    try {
        if (std::string(filter) != "Garnet-VLM") return X3_STATUS_ERROR;
        *result = Galaxy::NativeFilter<Galaxy::GarnetVLMFilter>::Create(
            host, factory, library, filter).Detach();
        return X3_STATUS_OK;
    } catch (const std::exception& error) {
        std::cerr << "Garnet GLoad: " << error.what() << '\n';
        return X3_STATUS_ERROR;
    }
}
