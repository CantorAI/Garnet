#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <string>

std::unordered_map<std::string, torch::Tensor> load_weights_from_binary(
    const std::string& filename) {
    std::unordered_map<std::string, torch::Tensor> model_weights;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file!");
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
        torch::ScalarType scalar_type = torch::kFloat32;  // Default to float32
        if (dtype == "torch.float32") {
            scalar_type = torch::kFloat32;
        }
        else if (dtype == "torch.float64") {
            scalar_type = torch::kFloat64;
        }
        else if (dtype == "torch.int32") {
            scalar_type = torch::kInt32;
        }
        else if (dtype == "torch.int64") {
            scalar_type = torch::kInt64;
        }
        else if (dtype == "torch.bfloat16") {
            scalar_type = torch::kBFloat16;
        }

        // Read shape
        uint64_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint64_t));  // Read the number of dimensions if written
        std::vector<int64_t> shape(num_dims);
        for (auto& dim : shape) {
            file.read(reinterpret_cast<char*>(&dim), sizeof(int64_t));
        }

        int64_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
        int64_t num_bytes = num_elements * torch::elementSize(scalar_type);
        std::vector<char> buffer(num_bytes);
        file.read(buffer.data(), num_bytes);

        // Create a tensor from the buffer
        torch::Tensor tensor = torch::from_blob(buffer.data(), shape, scalar_type).clone();
        model_weights[key] = tensor;
    }

    file.close();
    return model_weights;
}

int main() {
    try {
        auto weights = load_weights_from_binary("D:\\ToGithub\\CantorAI\\Garnet\\model_weights.bin");
        for (const auto& pair : weights) {
            std::cout << "Key: " << pair.first << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
