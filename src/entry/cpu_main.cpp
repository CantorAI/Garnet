// SPDX-License-Identifier: Apache-2.0
#if defined(GARNET_CPU_ONLY)
#include "xlang3/xlang3.h"
#include "garnet_tensor.h"
#include "compiled_graph.h"
#include "tensor_io.h"
#include "tokenizer_api.h"
#include "openvino_builder.h"
#include <openvino/openvino.hpp>
namespace Garnet {
class GarnetCPUAPI {
public:
    void Shutdown() { OpenVINOBuilder::ShutdownCPUBackend(); }
    std::string BackendInfo() const {
        return std::string("Garnet CPU-only OpenVINO ")+ov::get_openvino_version().buildNumber;
    }
    BEGIN_PACKAGE(GarnetCPUAPI)
        APISET().AddClass<0,GarnetTensor>("tensor");
        APISET().AddClass<0,CPUCompiledGraph>("compiled_graph");
        APISET().AddClass<0,CPUTensorIO>("tensor_io");
        APISET().AddClass<0,CPUTokenizer>("tokenizer");
        APISET().AddFunc<0>("backend_info",&GarnetCPUAPI::BackendInfo);
        APISET().AddFunc<0>("shutdown",&GarnetCPUAPI::Shutdown);
    END_PACKAGE
};
}
extern "C" XLANG3_PACKAGE_EXPORT const uint32_t xlang3_package_abi_version=X3_ABI_VERSION;
extern "C" XLANG3_PACKAGE_EXPORT X3Status Load(void* rawHost,X3Value current) {
    auto* host=static_cast<X3PackageHost*>(rawHost);
    if(!host||host->abi_version!=X3_ABI_VERSION)return X3_STATUS_ERROR;
    try {
        Garnet::GarnetCPUAPI::BuildAPI();
        return Garnet::GarnetCPUAPI::APISET().Create(host,"garnet",current);
    }catch(...){return X3_STATUS_ERROR;}
}
#endif
