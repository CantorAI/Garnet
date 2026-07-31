#pragma once
#include "singleton.h"
#include "xpackage.h"
#include "xlang.h"
#include "garnet_tensor.h"
#include "model.h"
#include "log.h"
#include <string>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Garnet
{
	class QwenVLRequestContext
	{
	public:
		X::Value inputIds;
		X::Value mmTokenTypeIds;
		X::Value pixelValues;
		X::Value imageGridTHW;
		X::Value kvCache;
		long long kvHandle = 0;
		int kvMaxTokens = 0;
		int kvLogicalLength = 0;
		int kvPageSize = 0;
		int kvLogicalPages = 0;
		int kvPhysicalPages = 0;
		int kvQHeads = 0;
		int kvHeads = 0;
		int kvHeadDim = 0;
		std::string modelDir;
		std::string imagePath;
		std::string prompt;
		int sourceHeight = 0;
		int sourceWidth = 0;
		int resizedHeight = 0;
		int resizedWidth = 0;
		int promptTokenCount = 0;
		int visualTokenCount = 0;
		int pixelValueCount = 0;
		int patchSize = 16;
		int temporalPatchSize = 2;
		int mergeSize = 2;
		long long imagePreprocessUs = 0;
		long long tokenizeUs = 0;
		long long tensorUploadUs = 0;
		long long totalUs = 0;

		BEGIN_PACKAGE(QwenVLRequestContext)
			APISET().AddProp0("input_ids", &QwenVLRequestContext::inputIds);
			APISET().AddProp0("mm_token_type_ids", &QwenVLRequestContext::mmTokenTypeIds);
			APISET().AddProp0("pixel_values", &QwenVLRequestContext::pixelValues);
			APISET().AddProp0("image_grid_thw", &QwenVLRequestContext::imageGridTHW);
			APISET().AddProp0("kv_cache", &QwenVLRequestContext::kvCache);
			APISET().AddPropWithType<long long>("kv_handle", &QwenVLRequestContext::kvHandle);
			APISET().AddPropWithType<int>("kv_max_tokens", &QwenVLRequestContext::kvMaxTokens);
			APISET().AddPropWithType<int>("kv_logical_length", &QwenVLRequestContext::kvLogicalLength);
			APISET().AddPropWithType<int>("kv_page_size", &QwenVLRequestContext::kvPageSize);
			APISET().AddPropWithType<int>("kv_logical_pages", &QwenVLRequestContext::kvLogicalPages);
			APISET().AddPropWithType<int>("kv_physical_pages", &QwenVLRequestContext::kvPhysicalPages);
			APISET().AddPropWithType<int>("kv_q_heads", &QwenVLRequestContext::kvQHeads);
			APISET().AddPropWithType<int>("kv_heads", &QwenVLRequestContext::kvHeads);
			APISET().AddPropWithType<int>("kv_head_dim", &QwenVLRequestContext::kvHeadDim);
			APISET().AddPropWithType<std::string>("model_dir", &QwenVLRequestContext::modelDir);
			APISET().AddPropWithType<std::string>("image_path", &QwenVLRequestContext::imagePath);
			APISET().AddPropWithType<std::string>("prompt", &QwenVLRequestContext::prompt);
			APISET().AddVarFunc("stats", &QwenVLRequestContext::Stats);
		END_PACKAGE

		void Stats(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};

	class KVCacheManager
	{
		struct SequenceState
		{
			std::vector<int> pages;
			long long logicalLength = 0;
		};
		int m_maxNumPages = 0;
		int m_pageSize = 0;
		int m_headDim = 0;
		int m_numKVHeads = 0;
		int m_numLayers = 0;
		int m_dtypeBytes = 2;
		int m_deviceId = 0;
		size_t m_bytesPerPagePerLayer = 0;
		size_t m_totalBytes = 0;
		void* m_keyArena = nullptr;
		void* m_valueArena = nullptr;
		std::deque<int> m_freePages;
		std::unordered_map<long long, SequenceState> m_sequences;

		int PagesForTokens(long long tokenCount) const;
		bool EnsurePages(long long sequenceId, long long tokenCount);
		X::Value MakePageList(const std::vector<int>& pages) const;
	public:
		~KVCacheManager();
		BEGIN_PACKAGE(KVCacheManager)
			APISET().AddVarFunc("allocate", &KVCacheManager::Allocate);
			APISET().AddVarFunc("append", &KVCacheManager::Append);
			APISET().AddVarFunc("free", &KVCacheManager::Free);
			APISET().AddVarFunc("stats", &KVCacheManager::Stats);
		END_PACKAGE

		void Configure(int maxNumPages, int pageSize, int headDim, int numKVHeads,
			int numLayers = 1, int dtypeBytes = 2, int deviceId = 0);
		void Allocate(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Append(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Free(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Stats(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};

	class GarnetAPI :
		public Singleton<GarnetAPI>
	{
		X::Value m_cantor;
		X::Value m_log;
		X::Value m_curModule;
		std::string m_baseFolder;//like cantor's folder
		X::Value m_current_weights;
		X::Value m_compiledEngine;
		X::Value m_servingModel;
		std::string m_servingModelRoot;
		std::string m_servingCacheRoot;
		std::string m_servingModelId;
		std::string m_servingInputCapability;
		std::string m_servingError;
		int m_servingMinPixels = 256 * 28 * 28;
		int m_servingMaxPixels = 1280 * 28 * 28;
		int m_servingMaxOutputTokens = 256;
		mutable std::mutex m_servingMutex;
		bool LoadModelFromFile(std::string modelPath, X::Dict& model);
	public:
		BEGIN_PACKAGE(GarnetAPI)
			APISET().AddPropL("cantor",
				[](auto* pThis, X::Value v)
				{
					pThis->SetCantor(v);
				},
				[](auto* pThis) {return pThis->m_cantor; });
			APISET().AddVarFunc("load_model", &GarnetAPI::LoadModelEx);
			APISET().AddVarFunc("qwen_vl_smart_resize", &GarnetAPI::QwenVLSmartResize);
			APISET().AddVarFunc("qwen_vl_create_request", &GarnetAPI::QwenVLCreateRequest);
			APISET().AddVarFunc("qwen_vl_prepare_request", &GarnetAPI::QwenVLPrepareRequest);
			APISET().AddVarFunc("qwen_vl_preprocess_image", &GarnetAPI::QwenVLPreprocessImage);
			APISET().AddVarFunc("qwen_vl_preprocess_jpeg_file", &GarnetAPI::QwenVLPreprocessJpegFile);
			APISET().AddVarFunc("KVCacheManager", &GarnetAPI::CreateKVCacheManager);
			APISET().AddVarFunc("device_paged_kv_write", &GarnetAPI::DevicePagedKVWriteTensor);
			APISET().AddVarFunc("device_paged_kv_attention", &GarnetAPI::DevicePagedKVAttentionTensor);
			APISET().AddVarFunc("tensor_add", &GarnetAPI::TensorAdd);
			APISET().AddVarFunc("embedding", &GarnetAPI::Embedding);
			APISET().AddVarFunc("replace_rows_by_mask", &GarnetAPI::ReplaceRowsByMask);
			APISET().AddVarFunc("tensor_last_row", &GarnetAPI::TensorLastRow);
			APISET().AddVarFunc("gelu_tanh", &GarnetAPI::GeluTanh);
			APISET().AddVarFunc("vision_rope", &GarnetAPI::VisionRoPE);
			APISET().AddVarFunc("tensor_to_gpu", &GarnetAPI::TensorToGPU);
			APISET().AddVarFunc("tensor_to_bfloat16", &GarnetAPI::TensorToBFloat16);
			APISET().AddVarFunc("tensor_from_bfloat16_bits", &GarnetAPI::TensorFromBFloat16Bits);
			APISET().AddVarFunc("tensor_from_host", &GarnetAPI::TensorFromHost);
			APISET().AddVarFunc("tensor_update_from_host", &GarnetAPI::TensorUpdateFromHost);
			APISET().AddVarFunc("tensor_to_cpu", &GarnetAPI::TensorToCPU);
			APISET().AddVarFunc("serve_model", &GarnetAPI::ServeModel);
			APISET().AddVarFunc("serve_status_json", &GarnetAPI::ServeStatusJson);
			APISET().AddVarFunc("infer_json", &GarnetAPI::InferJson);
			APISET().AddVarFunc("stop_serving", &GarnetAPI::StopServing);
			APISET().AddVarFunc("runTest", &GarnetAPI::RunTest);
			APISET().AddClass<0, KVCacheManager>("KVCacheManagerClass");
			APISET().AddClass<0, QwenVLRequestContext>("QwenVLRequestContext");
			APISET().AddClass<0, Model>("model");
			APISET().AddClass<0, GarnetTensor>("tensor");
		END_PACKAGE

		void SetBaseFolder(std::string folder)
		{
			m_baseFolder = folder;
		}
		std::string GetBaseFolder()
		{
			return m_baseFolder;
		}
		void SetModule(X::Value curModule)
		{
			m_curModule = curModule;
		}

		void SetCurrentWeights(X::Value weights) { m_current_weights = weights; }
		X::Value GetCurrentWeights() { return m_current_weights; }

		void SetCompiledEngine(X::Value engine) { m_compiledEngine = engine; }
		X::Value GetCompiledEngine() { return m_compiledEngine; }
		bool SetCantor(X::Value cantor)
		{
			if (m_cantor.IsValid())
			{
				return true;
			}
			m_cantor = cantor;
			m_log = cantor["log_nolineend"];
			InitLog(m_log);

			return true;
		}
		X::Value LoadModel(std::string modelPath);
		void CreateKVCacheManager(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void LoadModelEx(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLSmartResize(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLCreateRequest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLPrepareRequest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLPreprocessImage(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void QwenVLPreprocessJpegFile(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void DevicePagedKVWriteTensor(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void DevicePagedKVAttentionTensor(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorAdd(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void Embedding(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void ReplaceRowsByMask(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorLastRow(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void GeluTanh(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void VisionRoPE(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorToGPU(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorToBFloat16(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorFromBFloat16Bits(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorFromHost(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorUpdateFromHost(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void TensorToCPU(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void ServeModel(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void ServeStatusJson(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void InferJson(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void StopServing(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
		void RunTest(X::XRuntime* rt, X::XObj* pContext,
			X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
	};
}
