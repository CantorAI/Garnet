#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };

    Net() {
        // Initialize layers
        fc1 = register_module("fc1", torch::nn::Linear(1, 10)); // Input layer
        fc2 = register_module("fc2", torch::nn::Linear(10, 1)); // Output layer
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};

auto generate_data(long long num_samples) {
    // Generate random x values
    auto x = torch::randn({ num_samples, 1 });
    // Generate corresponding y values
    auto y = 3 * x + 2 + torch::randn({ num_samples, 1 }) * 0.1; // Adding some noise
    return std::make_pair(x, y);
}

int main2() {
    torch::Device device(torch::kCPU); // Default to CPU
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA); // Use GPU if available
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    }
    else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }

    auto net = std::make_shared<Net>();
    net->to(device); // Move the model to the appropriate device
    auto data = generate_data(100); // Generate 100 data points

    auto inputs = data.first.to(device);
    auto targets = data.second.to(device);

    auto optimizer = torch::optim::SGD(net->parameters(), 0.01); // Learning rate
    auto criterion = torch::nn::MSELoss();

    for (size_t epoch = 0; epoch < 100; ++epoch) { // Train for 100 epochs
        optimizer.zero_grad(); // Clear gradients
        auto outputs = net->forward(inputs); // Compute model output
        auto loss = criterion(outputs, targets); // Compute loss
        loss.backward(); // Backpropagation
        optimizer.step(); // Update weights

        if (epoch % 10 == 0) { // Print loss every 10 epochs
            std::cout << "Epoch " << epoch << ": Loss: " << loss.item<float>() << std::endl;
        }
    }

    return 0;
}
