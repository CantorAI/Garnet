// SPDX-License-Identifier: Apache-2.0
#include "xlang3/xlang3.h"
#include "garnet_tensor.h"
#include "../backends/metal/compiled_graph.h"
#include "../backends/metal/decoder_session.h"
#include "model_tensor_io.h"
#include "tokenizer_api.h"
#include "../image/native_image.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <atomic>
#include <chrono>
namespace Garnet {
class GarnetMetalAPI {
    X::Module Serving(X3PackageHost* host) {
        const std::string root = GARNET_METAL_SOURCE_ROOT;
        if (x3_runtime_add_import_root(host->runtime, root.c_str()) != X3_STATUS_OK ||
                x3_runtime_add_import_root(host->runtime, (root + "/runtime").c_str()) != X3_STATUS_OK)
            throw X::Error(x3_runtime_last_error(host->runtime));
        return X::Module(host, "metal_serving");
    }
public:
    X::Value PrepareImage(X::Value path,X::Value options) {
        auto params=nlohmann::json::parse(options.ToString());
        Image::NativeImageConfig c;
        c.patch=params.at("patch_size").get<int>();c.merge=params.at("merge_size").get<int>();
        c.temporal=params.at("temporal_patch_size").get<int>();c.pixels=params.at("pixels").get<int>();
        auto mean=params.at("image_mean"),stddev=params.at("image_std");
        if(mean.size()!=3||stddev.size()!=3)throw X::Error("Native image normalization requires RGB");
        for(int i=0;i<3;++i){c.mean[i]=mean.at(i).get<float>();c.stddev[i]=stddev.at(i).get<float>();}
        auto image=Image::PrepareNativeImage(path.ToString(),c);
        std::vector<int64_t> shape{1,int64_t(image.gridH)*image.gridW,image.features};
        auto result=X::Value::List(path.host());result.Append(X::Tensor::Create(path.host(),X3_TENSOR_FLOAT32,shape,image.patches.data(),image.patches.size()*4));
        result.Append(X::Value(image.gridH));result.Append(X::Value(image.gridW));
        result.Append(X::Value(image.decodeMs));result.Append(X::Value(image.resizeMs));result.Append(X::Value(image.packMs));return result;
    }
    std::string BenchmarkPrefillPolicy(X::Value enabled) {
        auto module=Serving(enabled.host());
        return module["set_benchmark_prefill_policy"](enabled).ToString();
    }
    std::string ServeModel(X::Value root, X::Value xmodel, X::Value cache,
            X::Value profile, X::Value model) {
        auto module = Serving(root.host());
        return module["serve_model"](root, xmodel, cache, profile, model).ToString();
    }
    std::string ServeStatus(X::Value model) {
        auto module = Serving(model.host());
        return module["serve_status_json"](model).ToString();
    }
    std::string InferJson(X::Value prompt, X::Value image, X::Value limit, X::Value model) {
#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
        MetalCounterTraceRequest counterRequest(model.ToString());
#endif
        auto module = Serving(prompt.host());
        std::filesystem::path temporary;
        std::string imagePath;
        if (image.IsBin()) {
            uint64_t size = 0;
            const auto* data = static_cast<const char*>(image.BytesData(&size));
            if (data && size) {
                static std::atomic<uint64_t> sequence{0};
                const auto folder = std::filesystem::temp_directory_path() / "garnet-metal-images";
                std::filesystem::create_directories(folder);
                temporary = folder / (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                    + "-" + std::to_string(sequence++) + ".jpg");
                std::ofstream out(temporary, std::ios::binary);
                out.write(data, static_cast<std::streamsize>(size));
                if (!out) throw X::Error("cannot stage Garnet Metal image input");
                imagePath = temporary.string();
            }
        } else if (image.IsString()) imagePath = image.ToString();
        try {
            const auto result = module["infer_json"](prompt, imagePath, limit, model).ToString();
            if (!temporary.empty()) std::filesystem::remove(temporary);
            return result;
        } catch (...) {
            if (!temporary.empty()) std::filesystem::remove(temporary);
            throw;
        }
    }
#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
    bool CounterTraceStep(int64_t step){MetalContext::CounterTraceDecodeStep(step);return true;}
    int64_t CounterTraceId(){return int64_t(MetalContext::CounterTraceRequestId());}
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
    bool CounterTraceEmitFault(int64_t enabled){if(enabled!=0&&enabled!=1)throw X::Error("Counter fault flag must be zero or one");MetalContext::CounterTraceTestEmitFailure(enabled!=0);return true;}
#endif
#endif
    std::string BackendInfo() const {
        auto info="Garnet Metal/MPS "+MetalContext::AvailableDeviceName();
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
        info+="; dense_key_split_k";
#endif
#ifdef GARNET_METAL_QWEN35_EXTENDED_PREFILL
        info+="; qwen35_extended_prefill";
#endif
#ifdef GARNET_METAL_QWEN35_WIDE_BATCH_PREFILL
        info+="; qwen35_wide_batch_prefill";
#endif
#ifdef GARNET_METAL_INTERLEAVED_BF16_BODY
        info+="; interleaved_bf16_body";
#endif
#ifdef GARNET_METAL_COALESCED_BF16_BODY
        info+="; coalesced_bf16_body";
#endif
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
        info+="; mlp_prefill_graph";
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
#ifdef GARNET_METAL_QWEN35_4B_WIDE_PREFILL
        info+="; qwen35_4b_wide_prefill";
#endif
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
        info+="; dense_prefix_cache";
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
#ifdef GARNET_METAL_SCREENED_HEAD
        info+="; screened_head";
#endif
#ifdef GARNET_METAL_SMALL_SCREEN_SEED
        info+="; screen_seed_1024";
#endif
#ifdef GARNET_METAL_INTERLEAVED_SCREEN_INDEX
        info+="; interleaved_screen_index";
#endif
#ifdef GARNET_METAL_SERIAL_VISION_PREFETCH
        info+="; serial_vision_prefetch; vector_vision_context4";
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
#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
        info+="; gpu_pass_counter_profile";
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
        APISET().AddFunc<2>("prepare_image",&GarnetMetalAPI::PrepareImage);
        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);
#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
        APISET().AddFunc<1>("counter_trace_step",&GarnetMetalAPI::CounterTraceStep);
        APISET().AddFunc<0>("counter_trace_request_id",&GarnetMetalAPI::CounterTraceId);
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
        APISET().AddFunc<1>("_test_counter_emit_failure",&GarnetMetalAPI::CounterTraceEmitFault);
#endif
#endif

        APISET().AddFunc<5>("serve_model",&GarnetMetalAPI::ServeModel);
        APISET().AddFunc<1>("serve_status_json",&GarnetMetalAPI::ServeStatus);
        APISET().AddFunc<4>("infer_json",&GarnetMetalAPI::InferJson);
        APISET().AddFunc<1>("set_benchmark_prefill_policy",&GarnetMetalAPI::BenchmarkPrefillPolicy);
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
