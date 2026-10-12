// SPDX-License-Identifier: Apache-2.0
#include "xlang3/xlang3.h"
#include "garnet_tensor.h"
#include "../backends/metal/compiled_graph.h"
#include "model_tensor_io.h"
#include "tokenizer_api.h"
namespace Garnet {
class GarnetMetalAPI {
public:
    std::string BackendInfo() const {return "Garnet Metal/MPS "+MetalContext::AvailableDeviceName();}
    BEGIN_PACKAGE(GarnetMetalAPI)
        APISET().AddClass<0,GarnetTensor>("tensor");
        APISET().AddClass<0,MetalCompiledGraph>("compiled_graph");
        APISET().AddClass<0,ModelTensorIO>("tensor_io");
        APISET().AddClass<0,CPUTokenizer>("tokenizer");
        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);
    END_PACKAGE
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version=X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* rawHost,X3Value current) {
    auto* host=static_cast<X3PackageHost*>(rawHost);
    if(!host||host->abi_version!=X3_ABI_VERSION)return X3_STATUS_ERROR;
    try {Garnet::GarnetMetalAPI::BuildAPI();return Garnet::GarnetMetalAPI::APISET().Create(host,"garnet",current);}
    catch(...){return X3_STATUS_ERROR;}
}
