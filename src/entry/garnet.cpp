#include "garnet.h"
#include "xpackage.h"
#include "xlang.h"
#include <fstream> 
#include <numeric> 

namespace Garnet
{
    X::Value GarnetAPI::LoadModel(std::string modelPath)
    {
        X::Dict model;
        std::ifstream file(modelPath, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open model file: " + modelPath);
        }

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
        return model;
    }

}
