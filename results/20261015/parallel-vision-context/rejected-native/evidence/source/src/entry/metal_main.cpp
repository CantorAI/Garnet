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
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
        info+="; deferred_state_zero";
#endif
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
        info+="; reused_zero_source";
#endif
#ifdef GARNET_METAL_BODY_SHAPE
        info+="; body_shape";
#endif
#ifdef GARNET_METAL_PREFILL_STREAM
        info+="; prefill_stream";
#endif
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
        info+="; dirty_prefix_reset";
#endif
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
        info+="; constant_bf16_head";
#endif
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
        info+="; single_coder_head";
#endif
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
        info+="; fused_head_argmax";
#endif
#ifdef GARNET_METAL_FUSED_HEAD_DECODE_ONLY
        info+="; fused_head_decode_only";
#endif
#ifdef GARNET_METAL_RESIDUAL_NORM
        info+="; residual_norm";
#endif
#ifdef GARNET_METAL_PACKED_PREFILL_NORM
        info+="; packed_prefill_norm";
#endif
#ifdef GARNET_METAL_GLOBAL_HALF_ALPHA
        info+="; global_half_alpha";
#endif
#ifdef GARNET_METAL_VISION_GLOBAL_ALPHA
        info+="; vision_global_alpha";
#endif
#ifdef GARNET_METAL_PARALLEL_VISION_CONTEXT
        info+="; parallel_vision_context";
#endif
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
        info+="; joined_global_mlp";
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
