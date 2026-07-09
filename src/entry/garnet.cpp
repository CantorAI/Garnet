#include "garnet.h"
#include "../trt/trt_builder.h"
#include "../image/qwen_vl/qwen_vl_image_preprocessor.h"
#include "xpackage.h"
#include "xlang.h"
#include <fstream> 
#include <numeric> 
#include <filesystem>
#include <regex>
#include <iostream>
#include <vector>

namespace Garnet
{
    namespace
    {
        std::vector<int> ReadIntList(X::Value value)
        {
            std::vector<int> result;
            if (!value.IsList()) return result;
            X::List list(value);
            long long size = list->Size();
            result.reserve(static_cast<size_t>(size));
            for (long long i = 0; i < size; ++i) {
                result.push_back(static_cast<int>(list->Get(i).ToLongLong()));
            }
            return result;
        }

        std::vector<int> TensorShape(X::Value value)
        {
            std::vector<int> result;
            if (!value.IsTensor()) return result;
            X::Tensor tensor(value);
            int dimCount = tensor->GetDimCount();
            result.reserve(static_cast<size_t>(dimCount));
            for (int i = 0; i < dimCount; ++i) {
                result.push_back(static_cast<int>(tensor->GetDimSize(i)));
            }
            return result;
        }

        X::Value GetKwarg(X::KWARGS& kwParams, const char* name)
        {
            if (kwParams.Has(name)) {
                auto it = kwParams.find(name);
                return it->val;
            }
            return X::Value();
        }

        int GetIntArg(X::ARGS& params, X::KWARGS& kwParams, size_t index, const char* name, int defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return static_cast<int>(value.ToLongLong());
            }
            if (params.size() > index) {
                return static_cast<int>(params[index].ToLongLong());
            }
            return defaultValue;
        }

        double GetDoubleArg(X::ARGS& params, X::KWARGS& kwParams, size_t index, const char* name, double defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return value.ToDouble();
            }
            if (params.size() > index) {
                return params[index].ToDouble();
            }
            return defaultValue;
        }

        std::string GetStringArg(X::ARGS& params, X::KWARGS& kwParams, size_t index, const char* name, const std::string& defaultValue)
        {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) {
                return value.ToString();
            }
            if (params.size() > index) {
                return params[index].ToString();
            }
            return defaultValue;
        }
    }

    void KVCacheManager::Configure(int maxNumPages, int pageSize, int headDim, int numKVHeads)
    {
        m_maxNumPages = maxNumPages > 0 ? maxNumPages : 0;
        m_pageSize = pageSize > 0 ? pageSize : 1;
        m_headDim = headDim > 0 ? headDim : 1;
        m_numKVHeads = numKVHeads > 0 ? numKVHeads : 1;
        m_freePages.clear();
        m_sequencePages.clear();
        for (int page = 0; page < m_maxNumPages; ++page) {
            m_freePages.push_back(page);
        }
    }

    void KVCacheManager::Allocate(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        X::Value sequenceLengthValue = GetKwarg(kwParams, "sequence_length");
        long long sequenceLength = sequenceLengthValue.IsValid()
            ? sequenceLengthValue.ToLongLong()
            : (params.size() > 1 ? params[1].ToLongLong() : 0);
        if (sequenceLength < 0) sequenceLength = 0;
        int pagesNeeded = static_cast<int>((sequenceLength + m_pageSize - 1) / m_pageSize);
        if (pagesNeeded > static_cast<int>(m_freePages.size())) {
            std::cout << "[KVCacheManager] Not enough free pages: need " << pagesNeeded
                << ", have " << m_freePages.size() << std::endl;
            retValue = X::Value();
            return;
        }
        auto existing = m_sequencePages.find(seqId);
        if (existing != m_sequencePages.end()) {
            for (int page : existing->second) {
                m_freePages.push_front(page);
            }
            m_sequencePages.erase(existing);
        }

        std::vector<int> pages;
        pages.reserve(static_cast<size_t>(pagesNeeded));
        X::V<X::XList> retList;
        for (int i = 0; i < pagesNeeded; ++i) {
            int page = m_freePages.front();
            m_freePages.pop_front();
            pages.push_back(page);
            X::Value pageValue(page);
            retList->AddItem(pageValue);
        }
        m_sequencePages[seqId] = std::move(pages);
        retValue = retList;
    }

    void KVCacheManager::Free(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        long long seqId = params.size() > 0 ? params[0].ToLongLong() : 0;
        auto existing = m_sequencePages.find(seqId);
        bool released = existing != m_sequencePages.end();
        if (released) {
            for (int page : existing->second) {
                m_freePages.push_back(page);
            }
            m_sequencePages.erase(existing);
        }
        retValue = X::Value(released);
    }

    void KVCacheManager::Stats(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        X::Dict stats;
        stats->Set("max_num_pages", X::Value(m_maxNumPages));
        stats->Set("free_pages", X::Value(static_cast<int>(m_freePages.size())));
        stats->Set("used_pages", X::Value(m_maxNumPages - static_cast<int>(m_freePages.size())));
        stats->Set("page_size", X::Value(m_pageSize));
        stats->Set("head_dim", X::Value(m_headDim));
        stats->Set("num_kv_heads", X::Value(m_numKVHeads));
        stats->Set("sequence_count", X::Value(static_cast<int>(m_sequencePages.size())));
        retValue = stats;
    }

    bool GarnetAPI::LoadModelFromFile(std::string modelPath, X::Dict& model)
    {
        std::ifstream file(modelPath, std::ios::binary);

        if (!file.is_open()) {
            return false;
        }
        LOG << "LoadModelFromFile at: " << modelPath << LINE_END;

        // Read metadata
        std::string line;
        std::getline(file, line, '\0');  // Skip metadata

        while (!file.eof()) {
            std::string key;
            std::getline(file, key, '\0');  // Read key
            if (key.empty() || file.eof()) break;

            std::string dtype;
            std::getline(file, dtype, '\0');  // Read dtype

            // Determine the data type
            X::TensorDataType tensor_data_type = X::TensorDataType::FLOAT32;  // Default to float32
            if (dtype == "torch.float32") {
                tensor_data_type = X::TensorDataType::FLOAT32;
            }
            else if (dtype == "torch.float64") {
                tensor_data_type = X::TensorDataType::FLOAT64;
            }
            else if (dtype == "torch.int32") {
                tensor_data_type = X::TensorDataType::INT32;
            }
            else if (dtype == "torch.int64") {
                tensor_data_type = X::TensorDataType::INT64;
            }
            else if (dtype == "torch.bfloat16") {
                tensor_data_type = X::TensorDataType::BFLOAT16;
            }
            else if (dtype.find("torch.float8_e4m3fn") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E4M3FN;
            }
            else if (dtype.find("torch.float8_e4m3fnuz") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E4M3FNUZ;
            }
            else if (dtype.find("torch.float8_e5m2") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E5M2;
            }
            else if (dtype.find("torch.float8_e5m2fnuz") != std::string::npos)
            {
                tensor_data_type = X::TensorDataType::FLOAT8_E5M2FNUZ;
            }
            // Read shape
            uint64_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint64_t));
            X::Port::vector<int> shape(num_dims);
            for (uint64_t i = 0; i < num_dims;i++) {
                int64_t d;
                file.read((char*)&d, sizeof(int64_t));
				shape.push_back((int)d);
            }
            X::Tensor tensor;
            tensor->SetShape(shape);
            tensor->SetDataType(tensor_data_type);

            int64_t num_elements = std::accumulate(shape.begin(), shape.end(), 
                1, std::multiplies<int64_t>());
            int64_t num_bytes = num_elements * tensor->GetItemSize();
            std::vector<char> buffer(num_bytes);
            file.read(buffer.data(), num_bytes);

            // Create and populate X::Tensor
            X::Value dummy;
            tensor->Create(dummy);

            // Copy data into tensor
            memcpy(tensor->GetData(), buffer.data(), num_bytes);

            // Store in dictionary
            model->Set(key, tensor);
        }

        file.close();
        return true;
    }


    X::Value GarnetAPI::LoadModel(std::string modelPath)
    {
        X::Dict dictModel;
        namespace fs = std::filesystem;

        std::string tokenizerJsonPath;
        std::string tokenizerConfigJsonPath;
        std::string strModelPath;
        // Check if the path contains a wildcard ('*' or '?')
        if (modelPath.find('*') != std::string::npos || modelPath.find('?') != std::string::npos)
        {
            // Split the path into directory and pattern parts.
            fs::path pathPattern(modelPath);
            fs::path directory = pathPattern.parent_path();
            if (directory.empty()) {
                directory = fs::current_path();
            }
			strModelPath = directory.string();
            std::string pattern = pathPattern.filename().string();

            // Convert wildcard pattern to a regular expression.
            // For example, "*.bin" becomes ".*\.bin"
            std::string regexPattern;
            for (char c : pattern)
            {
                if (c == '*')
                    regexPattern += ".*";
                else if (c == '?')
                    regexPattern += ".";
                // Escape regex special characters, except for alphanumerics
                else if (std::isalnum(c) || c == '_' || c == '-')
                    regexPattern += c;
                else
                    regexPattern += "\\" + std::string(1, c);
            }
            std::regex fileRegex(regexPattern, std::regex::icase);

            // Iterate over files in the target directory.
            for (const auto& entry : fs::directory_iterator(directory))
            {
                if (entry.is_regular_file())
                {
                    std::string filename = entry.path().filename().string();
                    if (std::regex_match(filename, fileRegex))
                    {
                        // Append the model data from each matching file.
                        LoadModelFromFile(entry.path().string(), dictModel);
                    }
                }
            }
        }
        else
        {
            // No wildcard found; check if it's a directory or file
            fs::path path(modelPath);

            if (fs::is_directory(path))
            {
				strModelPath = path.string();
                // Reset tokenizer paths
                tokenizerJsonPath = "";
                tokenizerConfigJsonPath = "";

                // Scan for *.bin files and tokenizer files
                for (const auto& entry : fs::directory_iterator(path))
                {
                    if (!entry.is_regular_file()) continue;

                    fs::path entryPath = entry.path();
                    std::string filename = entryPath.filename().string();
                    std::string extension = entryPath.extension().string();

                    if (extension == ".bin")
                    {
                        // Load .bin file
                        LoadModelFromFile(entryPath.string(), dictModel);
                    }
                    else if (filename == "tokenizer.json")
                    {
                        tokenizerJsonPath = fs::absolute(entryPath).string();
                        LOG << "Found tokenizer.json at: " << tokenizerJsonPath << LINE_END;
                    }
                    else if (filename == "tokenizer_config.json")
                    {
                        tokenizerConfigJsonPath = fs::absolute(entryPath).string();
                        LOG << "Found tokenizer_config.json at: " << tokenizerConfigJsonPath << LINE_END;
                    }
                }
            }
            else
            {
                // Single file
                fs::path fsPath(modelPath);
                fs::path directory = fsPath.parent_path();
                strModelPath = directory.string();
                LoadModelFromFile(modelPath, dictModel);
            }
        }

        X::XPackageValue<Model> varModel;
        Model& model = *varModel;
		model.SetInfo(strModelPath, tokenizerJsonPath, tokenizerConfigJsonPath, dictModel);
        return varModel;
    }

    void GarnetAPI::CreateKVCacheManager(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        auto getInt = [&](const char* name, size_t pos, int defaultValue) -> int {
            X::Value value = GetKwarg(kwParams, name);
            if (value.IsValid()) return static_cast<int>(value.ToLongLong());
            if (params.size() > pos) return static_cast<int>(params[pos].ToLongLong());
            return defaultValue;
        };
        int maxNumPages = getInt("max_num_pages", 0, 0);
        int pageSize = getInt("page_size", 1, 16);
        int headDim = getInt("head_dim", 2, 128);
        int numKVHeads = getInt("num_kv_heads", 3, 1);
        X::XPackageValue<KVCacheManager> manager;
        (*manager).Configure(maxNumPages, pageSize, headDim, numKVHeads);
        retValue = manager;
    }

    void GarnetAPI::LoadModelEx(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        if (params.size() == 0) return;
        std::string modelPath = params[0].ToString();
        
        namespace fs = std::filesystem;
        fs::path path(modelPath);
        
        X::Value modelVal;
        
        // Check if the path is a .x file
        if (!fs::is_directory(path) && path.extension() == ".x") {
            // Load an empty model object
            X::XPackageValue<Model> varModel;
            Model& model = *varModel;
            X::Dict dictModel;
            std::string dir = path.parent_path().string();
            std::string emptyStr = "";
            model.SetInfo(dir, emptyStr, emptyStr, dictModel);
            modelVal = varModel;
            
            // Read the script file
            std::cout << "[Garnet] Loading .x module from " << modelPath << std::endl;
            std::ifstream file(modelPath);
            std::string code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (file.fail() && code.empty()) {
                std::cout << "[Garnet] Failed to read file " << modelPath << std::endl;
            }

            X::Value moduleVal;
            bool bOK = X::g_pXHost->LoadModule(modelPath.c_str(), code.c_str(), (int)code.size(), moduleVal);
            std::cout << "[Garnet] LoadModule returned " << bOK << std::endl;
            if (bOK && moduleVal.IsObject()) {
                X::Value weightsDict;
                X::Value inputShapes;
                X::Value weightShape;
                std::string subgraph;
                std::string cacheDir = (path.parent_path() / "cache").string();
                for (auto& it : kwParams) {
                    if (std::string(it.key) == "weights") {
                        weightsDict = it.val;
                    }
                    else if (std::string(it.key) == "input_shapes") {
                        inputShapes = it.val;
                    }
                    else if (std::string(it.key) == "weight_shape") {
                        weightShape = it.val;
                    }
                    else if (std::string(it.key) == "cache_dir") {
                        cacheDir = it.val.ToString();
                    }
                    else if (std::string(it.key) == "subgraph") {
                        subgraph = it.val.ToString();
                    }
                }

                model.SetInfo(dir, emptyStr, emptyStr, weightsDict);
                model.SetSubgraph(subgraph);
                
                // Store weights in GarnetAPI singleton before running script
                GarnetAPI::I().SetCurrentWeights(weightsDict);

                X::Value retVal;
                X::g_pXHost->RunModule(moduleVal, retVal, true);

                if (subgraph == "qwen3_text_mlp" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> gateShape = TensorShape(weights["language_model.layers.0.mlp.gate_proj.weight"]);
                        std::vector<int> upShape = TensorShape(weights["language_model.layers.0.mlp.up_proj.weight"]);
                        std::vector<int> downShape = TensorShape(weights["language_model.layers.0.mlp.down_proj.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaTextMlp = inputShape.size() == 2 && inputShape[0] > 64;
                        if (useCudaTextMlp) {
                            model.SetEngine(X::Value("cuda_text_mlp"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextMLPEngine(enginePath.string(), inputShape, gateShape, upShape, downShape);
                        }
                        if (!useCudaTextMlp && std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_qkv_proj" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> qShape = TensorShape(weights["language_model.layers.0.self_attn.q_proj.weight"]);
                        std::vector<int> kShape = TensorShape(weights["language_model.layers.0.self_attn.k_proj.weight"]);
                        std::vector<int> vShape = TensorShape(weights["language_model.layers.0.self_attn.v_proj.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextQKVEngine(enginePath.string(), inputShape, qShape, kShape, vShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_qkv_head_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> qShape = TensorShape(weights["language_model.layers.0.self_attn.q_proj.weight"]);
                        std::vector<int> kShape = TensorShape(weights["language_model.layers.0.self_attn.k_proj.weight"]);
                        std::vector<int> vShape = TensorShape(weights["language_model.layers.0.self_attn.v_proj.weight"]);
                        std::vector<int> qNormShape = TensorShape(weights["language_model.layers.0.self_attn.q_norm.weight"]);
                        std::vector<int> kNormShape = TensorShape(weights["language_model.layers.0.self_attn.k_norm.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextQKVHeadNormEngine(enginePath.string(), inputShape, qShape, kShape, vShape, qNormShape, kNormShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_rope_apply" && inputShapes.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() >= 3) {
                        std::vector<int> qkvShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> cosShape = ReadIntList(shapeList->Get(1));
                        std::vector<int> sinShape = ReadIntList(shapeList->Get(2));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextRoPEEngine(enginePath.string(), qkvShape, cosShape, sinShape, 16, 8, 128);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_attention_core" && inputShapes.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        std::vector<int> qkvShape = ReadIntList(shapeList->Get(0));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportTextAttentionEngine(enginePath.string(), qkvShape, 16, 8, 128);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_attention_core" && inputShapes.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        std::vector<int> qkvShape = ReadIntList(shapeList->Get(0));
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (qkvShape.size() == 2 && qkvShape[0] > 512) {
                            model.SetEngine(X::Value("cuda_exact_vision_attention"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportVisionAttentionEngine(enginePath.string(), qkvShape, 16, 64);
                        }
                        if (qkvShape.size() == 2 && qkvShape[0] > 512) {
                            model.SetEngine(X::Value("cuda_exact_vision_attention"));
                        }
                        else if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_o_proj" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> oShape = TensorShape(weights["language_model.layers.0.self_attn.o_proj.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearTransposeEngine(enginePath.string(), inputShape, oShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_lm_head" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> embedShape = TensorShape(weights["language_model.embed_tokens.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaLinear = inputShape.size() == 2
                            && embedShape.size() == 2
                            && (embedShape[0] > 65536 || (static_cast<long long>(inputShape[0]) * static_cast<long long>(embedShape[0]) > 8LL * 1024LL * 1024LL));
                        if (useCudaLinear) {
                            model.SetEngine(X::Value("cuda_linear_transpose"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearTransposeEngine(enginePath.string(), inputShape, embedShape);
                        }
                        if (!useCudaLinear && std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_patch_embed" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["visual.patch_embed.proj.weight"]);
                        std::vector<int> biasShape = TensorShape(weights["visual.patch_embed.proj.bias"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearBiasTransposeEngine(enginePath.string(), inputShape, weightShape, biasShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "linear_bias" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["W"]);
                        std::vector<int> biasShape = TensorShape(weights["B"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        bool useCudaLinear = inputShape.size() == 2 && weightShape.size() == 2
                            && (inputShape[0] > 2048 || (static_cast<long long>(inputShape[0]) * static_cast<long long>(weightShape[0]) > 8LL * 1024LL * 1024LL));
                        if (useCudaLinear) {
                            model.SetEngine(X::Value("cuda_linear_bias_transpose"));
                        }
                        else if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLinearBiasTransposeEngine(enginePath.string(), inputShape, weightShape, biasShape);
                        }
                        if (useCudaLinear) {
                            model.SetEngine(X::Value("cuda_linear_bias_transpose"));
                        }
                        else if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "vision_mlp" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> fc1Shape = TensorShape(weights["visual.blocks.0.mlp.linear_fc1.weight"]);
                        std::vector<int> fc2Shape = TensorShape(weights["visual.blocks.0.mlp.linear_fc2.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportVisionMLPEngine(enginePath.string(), inputShape, fc1Shape, fc2Shape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "rms_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        model.SetRMSNormWeight(weights["language_model.layers.0.input_layernorm.weight"]);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["language_model.layers.0.input_layernorm.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportRMSNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "text_post_attention_rms_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        model.SetRMSNormWeight(weights["language_model.layers.0.post_attention_layernorm.weight"]);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["language_model.layers.0.post_attention_layernorm.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportRMSNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (subgraph == "layer_norm" && inputShapes.IsList() && weightsDict.IsObject()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        X::Dict weights(weightsDict);
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> weightShape = TensorShape(weights["visual.blocks.0.norm1.weight"]);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportLayerNormEngine(enginePath.string(), inputShape, weightShape, 1.0e-6f);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }
                else if (inputShapes.IsList() && weightShape.IsList()) {
                    X::List shapeList(inputShapes);
                    if (shapeList->Size() > 0) {
                        std::vector<int> inputShape = ReadIntList(shapeList->Get(0));
                        std::vector<int> wShape = ReadIntList(weightShape);
                        std::filesystem::path enginePath = std::filesystem::path(cacheDir) / (path.stem().string() + ".engine");
                        if (!std::filesystem::exists(enginePath)) {
                            TRTBuilder builder;
                            builder.ExportMatmulEngine(enginePath.string(), inputShape, wShape);
                        }
                        if (std::filesystem::exists(enginePath)) {
                            model.SetEngine(X::Value(enginePath.string()));
                        }
                    }
                }

                // Extract compiled engine that was set during script execution
                X::Value compiledEngine = GarnetAPI::I().GetCompiledEngine();
                if (!model.m_engine.IsValid() && compiledEngine.IsValid()) {
                    model.SetEngine(compiledEngine);
                } else if (!model.m_engine.IsValid()) {
                    std::cout << "[Garnet] Warning: Script finished but no engine was compiled!" << std::endl;
                }
            }
        }
        else {
            modelVal = LoadModel(modelPath);
        }
        
        retValue = modelVal;
    }

    void GarnetAPI::QwenVLSmartResize(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        try {
            int height = GetIntArg(params, kwParams, 0, "height", 0);
            int width = GetIntArg(params, kwParams, 1, "width", 0);
            int patchSize = GetIntArg(params, kwParams, 2, "patch_size", 16);
            int mergeSize = GetIntArg(params, kwParams, 3, "merge_size", 2);
            int minPixels = GetIntArg(params, kwParams, 4, "min_pixels", 65536);
            int maxPixels = GetIntArg(params, kwParams, 5, "max_pixels", 65536);
            auto resized = Image::QwenVL::SmartResize(
                height,
                width,
                patchSize * mergeSize,
                minPixels,
                maxPixels);

            X::Dict result;
            int gridH = resized.height / patchSize;
            int gridW = resized.width / patchSize;
            result->Set("height", X::Value(resized.height));
            result->Set("width", X::Value(resized.width));
            result->Set("patch_size", X::Value(patchSize));
            result->Set("merge_size", X::Value(mergeSize));
            result->Set("grid_h", X::Value(gridH));
            result->Set("grid_w", X::Value(gridW));
            result->Set("visual_tokens", X::Value(gridH * gridW / (mergeSize * mergeSize)));
            retValue = result;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_smart_resize failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::QwenVLPreprocessImage(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
        try {
            X::Value image = GetKwarg(kwParams, "image");
            if (!image.IsValid() && params.size() > 0) {
                image = params[0];
            }
            if (!image.IsValid()) {
                retValue = X::Value();
                return;
            }

            Image::QwenVL::QwenVLImagePreprocessConfig config;
            int height = GetIntArg(params, kwParams, 1, "height", 0);
            int width = GetIntArg(params, kwParams, 2, "width", 0);
            config.patchSize = GetIntArg(params, kwParams, 3, "patch_size", 16);
            config.temporalPatchSize = GetIntArg(params, kwParams, 4, "temporal_patch_size", 2);
            config.mergeSize = GetIntArg(params, kwParams, 5, "merge_size", 2);
            config.inputScale = static_cast<float>(GetDoubleArg(params, kwParams, 6, "input_scale", 255.0));
            std::string inputFormat = GetStringArg(params, kwParams, 7, "format", "rgb");
            config.pixelFormat = Image::PixelFormatFromString(inputFormat);

            auto result = Image::QwenVL::PreprocessRawImageTensor(image, height, width, config);
            X::Dict dict;
            dict->Set("pixel_values", result.pixelValues);
            dict->Set("image_grid_thw", result.imageGridTHW);
            dict->Set("height", X::Value(result.resizedHeight));
            dict->Set("width", X::Value(result.resizedWidth));
            dict->Set("patch_size", X::Value(result.patchSize));
            dict->Set("temporal_patch_size", X::Value(result.temporalPatchSize));
            dict->Set("merge_size", X::Value(result.mergeSize));
            dict->Set("input_format", X::Value(inputFormat));
            dict->Set("backend", X::Value("cuda_raw_tensor"));
            retValue = dict;
        }
        catch (const std::exception& exc) {
            std::cout << "[GarnetAPI] qwen_vl_preprocess_image failed: " << exc.what() << std::endl;
            retValue = X::Value();
        }
    }

    void GarnetAPI::RunTest(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
    }

}
