// SPDX-License-Identifier: Apache-2.0
#include "xlang3/xlang3.h"
#include "garnet_tensor.h"
#include "../backends/metal/compiled_graph.h"
#include "../backends/metal/decoder_session.h"
#include "model_tensor_io.h"
#include "tokenizer_api.h"
namespace Garnet {
class GarnetMetalAPI {
public:
    std::string BackendInfo() const {
        auto info="Garnet Metal/MPS "+MetalContext::AvailableDeviceName();
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
        info+="; dense_key_split_k";
#endif
#ifdef GARNET_METAL_QWEN35_EXTENDED_PREFILL
        info+="; qwen35_extended_prefill";
#endif
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
        info+="; resident_kv_inplace";
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
        info+="; joined_mps_prefill";
#endif
#ifdef GARNET_METAL_ATTENTION_PREFIX
        info+="; attention_prefix";
#endif
#ifdef GARNET_METAL_DIRECT_STATE_ZERO
        info+="; direct_state_zero";
#endif
        return info;
    }
    // Graphs own synchronous queues and release them through close(). There
    // is no package-global executor to drain, unlike asynchronous backends.
    bool Shutdown() const {return true;}
    BEGIN_PACKAGE(GarnetMetalAPI)
        APISET().AddClass<0,GarnetTensor>("tensor");
        APISET().AddClass<0,MetalCompiledGraph>("compiled_graph");
        APISET().AddClass<0,MetalDecoderSession>("decoder_session");
        APISET().AddClass<0,ModelTensorIO>("tensor_io");
        APISET().AddClass<0,CPUTokenizer>("tokenizer");
        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);
        APISET().AddFunc<0>("shutdown",&GarnetMetalAPI::Shutdown);
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
