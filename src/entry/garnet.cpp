#include "garnet.h"
#include "xpackage.h"
#include "xlang.h"
#include <fstream> 
#include <numeric> 
#include <filesystem>
#include <regex>
#include <iostream>

namespace Garnet
{
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
                for (auto& it : kwParams) {
                    if (std::string(it.key) == "weights") {
                        weightsDict = it.val;
                    }
                }
                
                // Store weights in GarnetAPI singleton before running script
                GarnetAPI::I().SetCurrentWeights(weightsDict);

                X::Value retVal;
                X::g_pXHost->RunModule(moduleVal, retVal, true);

                // Extract compiled engine that was set during script execution
                X::Value compiledEngine = GarnetAPI::I().GetCompiledEngine();
                if (compiledEngine.IsValid()) {
                    model.SetEngine(compiledEngine);
                } else {
                    std::cout << "[Garnet] Warning: Script finished but no engine was compiled!" << std::endl;
                }
            }
        }
        else {
            modelVal = LoadModel(modelPath);
        }
        
        retValue = modelVal;
    }

    void GarnetAPI::RunTest(X::XRuntime* rt, X::XObj* pContext,
        X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue)
    {
    }

}
