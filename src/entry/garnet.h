#pragma once
#include "xlang3/xlang3.h"
#include "garnet_tensor.h"
#include "model.h"
#include "model_manager.h"
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

		X::Value Stats(const X::ARGS& params, const X::KWARGS& kwParams);
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
		X::Value Allocate(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value Append(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value Free(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value Stats(const X::ARGS& params, const X::KWARGS& kwParams);
	};

	class GarnetAPI
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
		ModelManager m_modelManager;
		ModelManager m_accelerationManager;
		bool LoadModelFromFile(std::string modelPath, X::Value& model);
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
			APISET().AddVarFunc("list_available_models_json", &GarnetAPI::ListAvailableModelsJson);
			APISET().AddVarFunc("list_loaded_models_json", &GarnetAPI::ListLoadedModelsJson);
			APISET().AddVarFunc("serve_status_json", &GarnetAPI::ServeStatusJson);
			APISET().AddVarFunc("infer_json", &GarnetAPI::InferJson);
			APISET().AddVarFunc("transcribe_json", &GarnetAPI::TranscribeJson);
			APISET().AddVarFunc("synthesize_json", &GarnetAPI::SynthesizeJson);
			APISET().AddVarFunc("stop_serving", &GarnetAPI::StopServing);
			APISET().AddVarFunc("configure_model_manager_json", &GarnetAPI::ConfigureModelManagerJson);
			APISET().AddVarFunc("list_remote_models_json", &GarnetAPI::ListRemoteModelsJson);
			APISET().AddVarFunc("list_installed_models_json", &GarnetAPI::ListInstalledModelsJson);
			APISET().AddVarFunc("install_model_json", &GarnetAPI::InstallModelJson);
			APISET().AddVarFunc("model_install_status_json", &GarnetAPI::ModelInstallStatusJson);
			APISET().AddVarFunc("cancel_model_install_json", &GarnetAPI::CancelModelInstallJson);
			APISET().AddVarFunc("verify_installed_model_json", &GarnetAPI::VerifyInstalledModelJson);
			APISET().AddVarFunc("remove_installed_model_json", &GarnetAPI::RemoveInstalledModelJson);
			APISET().AddVarFunc("serve_installed_model_json", &GarnetAPI::ServeInstalledModelJson);
			APISET().AddVarFunc("_run_model_install_job", &GarnetAPI::RunModelInstallJob);
			APISET().AddVarFunc("detect_acceleration_json", &GarnetAPI::DetectAccelerationJson);
			APISET().AddVarFunc("configure_acceleration_manager_json", &GarnetAPI::ConfigureAccelerationManagerJson);
			APISET().AddVarFunc("list_acceleration_packages_json", &GarnetAPI::ListAccelerationPackagesJson);
			APISET().AddVarFunc("prepare_acceleration_json", &GarnetAPI::PrepareAccelerationJson);
			APISET().AddVarFunc("list_installed_accelerations_json", &GarnetAPI::ListInstalledAccelerationsJson);
			APISET().AddVarFunc("activate_acceleration_json", &GarnetAPI::ActivateAccelerationJson);
			APISET().AddVarFunc("activate_acceleration_path_json", &GarnetAPI::ActivateAccelerationPathJson);
			APISET().AddVarFunc("acceleration_install_status_json", &GarnetAPI::AccelerationInstallStatusJson);
			APISET().AddVarFunc("cancel_acceleration_install_json", &GarnetAPI::CancelAccelerationInstallJson);
			APISET().AddVarFunc("runTest", &GarnetAPI::RunTest);
			APISET().AddClass<0, KVCacheManager>("KVCacheManagerClass");
			APISET().AddClass<0, QwenVLRequestContext>("QwenVLRequestContext");
			APISET().AddClass<0, Model>("model");
			APISET().AddClass<0, GarnetTensor>("tensor");
		END_PACKAGE

		GarnetAPI();
        ~GarnetAPI();
        void OnPackageCreated(X::Package<GarnetAPI>* package);

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
		X::Value CreateKVCacheManager(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value LoadModelEx(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value QwenVLSmartResize(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value QwenVLCreateRequest(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value QwenVLPrepareRequest(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value QwenVLPreprocessImage(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value QwenVLPreprocessJpegFile(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value DevicePagedKVWriteTensor(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value DevicePagedKVAttentionTensor(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorAdd(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value Embedding(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ReplaceRowsByMask(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorLastRow(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value GeluTanh(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value VisionRoPE(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorToGPU(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorToBFloat16(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorFromBFloat16Bits(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorFromHost(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorUpdateFromHost(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TensorToCPU(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ServeModel(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ListAvailableModelsJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ListLoadedModelsJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ServeStatusJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value InferJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value TranscribeJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value SynthesizeJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value StopServing(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ConfigureModelManagerJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ListRemoteModelsJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ListInstalledModelsJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value InstallModelJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ModelInstallStatusJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value CancelModelInstallJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value VerifyInstalledModelJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value RemoveInstalledModelJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ServeInstalledModelJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value RunModelInstallJob(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value DetectAccelerationJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ConfigureAccelerationManagerJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ListAccelerationPackagesJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value PrepareAccelerationJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ListInstalledAccelerationsJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ActivateAccelerationJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value ActivateAccelerationPathJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value AccelerationInstallStatusJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value CancelAccelerationInstallJson(const X::ARGS& params, const X::KWARGS& kwParams);
		X::Value RunTest(const X::ARGS& params, const X::KWARGS& kwParams);

		std::string AvailableModelsJson(const std::string& catalogRoot) const;
		std::string LoadedModelsJson();
	};
}
