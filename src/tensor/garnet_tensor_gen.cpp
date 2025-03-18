// garnet_tensor_gen.cpp
#include "garnet_tensor.h"
#include <cuda_runtime.h>
#include "tensor_helper.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <string>
#include <sstream>

namespace Garnet {

    // Generate CUDA file header code
    X::Value GarnetTensor::Header(X::Value& graph, X::ARGS& params) {
        std::string headerCode =
            "#include <cuda_runtime.h>\n"
            "#include <cuda_fp16.h>\n"
            "#include <cuda_bf16.h>\n"
            "#include <cuda_fp8.h>\n\n"

            "// Include CUDA function declarations\n"
            "#include \"cuda_templates/cuda_function_declarations.h\"\n\n"

            "// Main function for executing the CUDA operations\n"
            "extern \"C\" void executeGarnetTensorOperations() {\n";

        return X::Value(headerCode);
    }

    // Generate CUDA file trailer code
    X::Value GarnetTensor::Trailer(X::Value& graph, X::ARGS& params) {
        std::string trailerCode =
            "// End of operations\n"
            "cudaDeviceSynchronize();\n"
            "}\n";

        return X::Value(trailerCode);
    }

    // Generate CUDA code for tensor multiplication
    X::Value GarnetTensor::Multiply(X::Value& graph, X::ARGS& params, 
        X::KWARGS& kwParams,X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        if (isTensor1 && isTensor2)
        {
            // Tensor-tensor multiplication
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);

            // Get tensor information
            auto tensor1_type = tensor1->GetDataType();
            auto tensor2_type = tensor2->GetDataType();
            int dimCount1 = tensor1->GetDimCount();
            int dimCount2 = tensor2->GetDimCount();

            // Validate dimensions
            if (dimCount1 > 2 || dimCount2 > 2)
            {
                return X::Value("// Error: Tensor dimensions > 2 not supported\n");
            }

            int m = tensor1->GetDimSize(0);
            int n = (dimCount1 > 1) ? tensor1->GetDimSize(1) : 1;
            int k = (dimCount2 > 1) ? tensor2->GetDimSize(1) : 1;

            // Check if dimensions match for matrix multiplication
            if (dimCount2 == 1)
            {
                // Vector case
                if (n != tensor2->GetDimSize(0))
                {
                    return X::Value("// Error: Dimension mismatch for vector multiplication\n");
                }
            }
            else if (n != tensor2->GetDimSize(0))
            {
                // Matrix case
                return X::Value("// Error: Inner dimensions must match for matrix multiplication\n");
            }

            // Ensure GPU memory is allocated for input tensors
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor1);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor1\n");
            }

            status = TensorHelper::EnsureGPUMemory(tensor2);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor2\n");
            }

            // Create result tensor with proper dimensions
            int dimNum = 1;
            if (tensor2->GetDimCount() > 1)
            {
                dimNum = 2;
            }
            X::Port::vector<int> resultDims(dimNum);
            resultDims.push_back(m);
            if (tensor2->GetDimCount() > 1) {
                resultDims.push_back(k);
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensor1_type);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers using TensorHelper
            void* gpuData1 = TensorHelper::GetGPUMemory(tensor1);
            void* gpuData2 = TensorHelper::GetGPUMemory(tensor2);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names based on tensor IDs
            std::string input1Var = mCodeGen.GetTensorName(tensor1);
            std::string input2Var = mCodeGen.GetTensorName(tensor2);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code using helper functions
            cudaCodeString = mCodeGen.GenerateCommentHeader("Matrix multiplication", tensor1, tensor2);

            // Add variable declarations
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input1Var, tensor1_type, gpuData1);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input2Var, tensor2_type, gpuData2);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensor1_type, gpuResultData);

            // Generate the kernel call based on data types - specific to the Multiply operation
            if (tensor1_type == X::TensorDataType::FLOAT32 && tensor2_type == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runGemmFP32(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                    std::to_string(n) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT16 && tensor2_type == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runGemmFP16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                    std::to_string(n) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::BFLOAT16 && tensor2_type == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runGemmBF16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                    std::to_string(n) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E4M3FN && tensor2_type == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runGemmFP8E4M3(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                    std::to_string(n) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E5M2 && tensor2_type == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runGemmFP8E5M2(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                    std::to_string(n) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type combination for matrix multiplication\n";
                return X::Value("// Error: Unsupported data type combination for matrix multiplication\n");
            }

            return X::Value(cudaCodeString);
        }
        else if (isTensor1)
        {
            // Tensor-scalar multiplication
            X::Tensor tensor(input1);
            auto tensorType = tensor->GetDataType();
            float scalar = (float)input2.ToDouble();

            // We only support 1D or 2D tensors
            int dimCount = tensor->GetDimCount();
            if (dimCount > 2)
            {
                return X::Value("// Error: Tensor dimensions > 2 not supported for scalar multiplication\n");
            }

            // Calculate total elements
            long long totalElements = 1;
            for (int i = 0; i < dimCount; i++)
            {
                totalElements *= tensor->GetDimSize(i);
            }

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor\n");
            }

            // Create result tensor with same shape as input
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                resultDims.push_back(tensor->GetDimSize(i));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get GPU memory pointers
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code for scalar multiplication
            cudaCodeString = mCodeGen.GenerateCommentHeader("Scalar multiplication", tensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(inputVar, tensorType, gpuData);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensorType, gpuResultData);

            // Add scalar declaration
            cudaCodeString += "float scalar_value = " + std::to_string(scalar) + ";\n";

            // Call the appropriate scalar multiplication function based on data type
            if (tensorType == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runScalarMultiplyFP32(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runScalarMultiplyFP16(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runScalarMultiplyBF16(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runScalarMultiplyFP8E4M3(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runScalarMultiplyFP8E5M2(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type for scalar multiplication\n";
                return X::Value("// Error: Unsupported data type for scalar multiplication\n");
            }

            return X::Value(cudaCodeString);
        }
        else if (isTensor2)
        {
            // Scalar-tensor multiplication (commutative)
            // Generate code by swapping the inputs
            return Multiply(graph, params, kwParams, input2, input1, output);
        }
        else
        {
            // Scalar-scalar multiplication
            float val1 = (float)input1.ToDouble();
            float val2 = (float)input2.ToDouble();
            cudaCodeString = "// Scalar-scalar multiplication\n";
            cudaCodeString += "float scalar_result = " + std::to_string(val1) + " * " + std::to_string(val2) + ";\n";
            return X::Value(cudaCodeString);
        }
    }


    // Implementation of Add function
    X::Value GarnetTensor::Add(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        if (isTensor1 && isTensor2)
        {
            // Tensor-tensor addition
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);

            // Get tensor information
            auto tensor1_type = tensor1->GetDataType();
            auto tensor2_type = tensor2->GetDataType();
            int dimCount1 = tensor1->GetDimCount();
            int dimCount2 = tensor2->GetDimCount();

            // Validate dimensions - tensors must have compatible shapes for addition
            if (dimCount1 != dimCount2)
            {
                return X::Value("// Error: Tensors must have same number of dimensions for addition\n");
            }

            for (int i = 0; i < dimCount1; i++)
            {
                if (tensor1->GetDimSize(i) != tensor2->GetDimSize(i))
                {
                    return X::Value("// Error: Tensor dimensions must match for addition\n");
                }
            }

            // Calculate total elements
            long long totalElements = 1;
            for (int i = 0; i < dimCount1; i++)
            {
                totalElements *= tensor1->GetDimSize(i);
            }

            // Ensure GPU memory is allocated for input tensors
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor1);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor1\n");
            }

            status = TensorHelper::EnsureGPUMemory(tensor2);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor2\n");
            }

            // Create result tensor with same shape as inputs
            X::Port::vector<int> resultDims(dimCount1);
            for (int i = 0; i < dimCount1; i++) {
                resultDims.push_back(tensor1->GetDimSize(i));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensor1_type);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers using TensorHelper
            void* gpuData1 = TensorHelper::GetGPUMemory(tensor1);
            void* gpuData2 = TensorHelper::GetGPUMemory(tensor2);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names based on tensor IDs
            std::string input1Var = mCodeGen.GetTensorName(tensor1);
            std::string input2Var = mCodeGen.GetTensorName(tensor2);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code using helper functions
            cudaCodeString = mCodeGen.GenerateCommentHeader("Tensor addition", tensor1, tensor2);

            // Add variable declarations
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input1Var, tensor1_type, gpuData1);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input2Var, tensor2_type, gpuData2);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensor1_type, gpuResultData);

            // Generate the kernel call based on data types - specific to the Add operation
            if (tensor1_type == X::TensorDataType::FLOAT32 && tensor2_type == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runAddFP32(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT16 && tensor2_type == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runAddFP16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::BFLOAT16 && tensor2_type == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runAddBF16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E4M3FN && tensor2_type == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runAddFP8E4M3(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E5M2 && tensor2_type == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runAddFP8E5M2(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type combination for tensor addition\n";
                return X::Value("// Error: Unsupported data type combination for tensor addition\n");
            }

            return X::Value(cudaCodeString);
        }
        else if (isTensor1)
        {
            // Tensor-scalar addition
            X::Tensor tensor(input1);
            auto tensorType = tensor->GetDataType();
            float scalar = (float)input2.ToDouble();

            // Calculate total elements
            int dimCount = tensor->GetDimCount();
            long long totalElements = 1;
            for (int i = 0; i < dimCount; i++)
            {
                totalElements *= tensor->GetDimSize(i);
            }

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor\n");
            }

            // Create result tensor with same shape as input
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                resultDims.push_back(tensor->GetDimSize(i));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get GPU memory pointers
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code for scalar addition
            cudaCodeString = mCodeGen.GenerateCommentHeader("Scalar addition", tensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(inputVar, tensorType, gpuData);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensorType, gpuResultData);

            // Add scalar declaration
            cudaCodeString += "float scalar_value = " + std::to_string(scalar) + ";\n";

            // Call the appropriate scalar addition function based on data type
            if (tensorType == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runScalarAddFP32(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runScalarAddFP16(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runScalarAddBF16(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runScalarAddFP8E4M3(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runScalarAddFP8E5M2(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type for scalar addition\n";
                return X::Value("// Error: Unsupported data type for scalar addition\n");
            }

            return X::Value(cudaCodeString);
        }
        else if (isTensor2)
        {
            // Scalar-tensor addition (commutative)
            // Generate code by swapping the inputs
            return Add(graph, params, kwParams, input2, input1, output);
        }
        else
        {
            // Scalar-scalar addition
            float val1 = (float)input1.ToDouble();
            float val2 = (float)input2.ToDouble();
            cudaCodeString = "// Scalar-scalar addition\n";
            cudaCodeString += "float scalar_result = " + std::to_string(val1) + " + " + std::to_string(val2) + ";\n";
            return X::Value(cudaCodeString);
        }
    }

    // Implementation of Minus function
    X::Value GarnetTensor::Minus(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        if (isTensor1 && isTensor2)
        {
            // Tensor-tensor subtraction
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);

            // Get tensor information
            auto tensor1_type = tensor1->GetDataType();
            auto tensor2_type = tensor2->GetDataType();
            int dimCount1 = tensor1->GetDimCount();
            int dimCount2 = tensor2->GetDimCount();

            // Validate dimensions - tensors must have compatible shapes for subtraction
            if (dimCount1 != dimCount2)
            {
                return X::Value("// Error: Tensors must have same number of dimensions for subtraction\n");
            }

            for (int i = 0; i < dimCount1; i++)
            {
                if (tensor1->GetDimSize(i) != tensor2->GetDimSize(i))
                {
                    return X::Value("// Error: Tensor dimensions must match for subtraction\n");
                }
            }

            // Calculate total elements
            long long totalElements = 1;
            for (int i = 0; i < dimCount1; i++)
            {
                totalElements *= tensor1->GetDimSize(i);
            }

            // Ensure GPU memory is allocated for input tensors
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor1);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor1\n");
            }

            status = TensorHelper::EnsureGPUMemory(tensor2);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor2\n");
            }

            // Create result tensor with same shape as inputs
            X::Port::vector<int> resultDims(dimCount1);
            for (int i = 0; i < dimCount1; i++) {
                resultDims.push_back(tensor1->GetDimSize(i));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensor1_type);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers using TensorHelper
            void* gpuData1 = TensorHelper::GetGPUMemory(tensor1);
            void* gpuData2 = TensorHelper::GetGPUMemory(tensor2);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names based on tensor IDs
            std::string input1Var = mCodeGen.GetTensorName(tensor1);
            std::string input2Var = mCodeGen.GetTensorName(tensor2);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code using helper functions
            cudaCodeString = mCodeGen.GenerateCommentHeader("Tensor subtraction", tensor1, tensor2);

            // Add variable declarations
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input1Var, tensor1_type, gpuData1);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input2Var, tensor2_type, gpuData2);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensor1_type, gpuResultData);

            // Generate the kernel call based on data types
            if (tensor1_type == X::TensorDataType::FLOAT32 && tensor2_type == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runSubtractFP32(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT16 && tensor2_type == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runSubtractFP16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::BFLOAT16 && tensor2_type == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runSubtractBF16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E4M3FN && tensor2_type == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runSubtractFP8E4M3(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E5M2 && tensor2_type == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runSubtractFP8E5M2(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type combination for tensor subtraction\n";
                return X::Value("// Error: Unsupported data type combination for tensor subtraction\n");
            }

            return X::Value(cudaCodeString);
        }
        else if (isTensor1)
        {
            // Tensor-scalar subtraction
            X::Tensor tensor(input1);
            auto tensorType = tensor->GetDataType();
            float scalar = (float)input2.ToDouble();

            // Calculate total elements
            int dimCount = tensor->GetDimCount();
            long long totalElements = 1;
            for (int i = 0; i < dimCount; i++)
            {
                totalElements *= tensor->GetDimSize(i);
            }

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor\n");
            }

            // Create result tensor with same shape as input
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                resultDims.push_back(tensor->GetDimSize(i));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get GPU memory pointers
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code for scalar subtraction
            cudaCodeString = mCodeGen.GenerateCommentHeader("Scalar subtraction", tensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(inputVar, tensorType, gpuData);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensorType, gpuResultData);

            // Add scalar declaration
            cudaCodeString += "float scalar_value = " + std::to_string(scalar) + ";\n";

            // Call the appropriate scalar subtraction function based on data type
            if (tensorType == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runScalarSubtractFP32(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runScalarSubtractFP16(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runScalarSubtractBF16(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runScalarSubtractFP8E4M3(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runScalarSubtractFP8E5M2(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type for scalar subtraction\n";
                return X::Value("// Error: Unsupported data type for scalar subtraction\n");
            }

            return X::Value(cudaCodeString);
        }
        else if (isTensor2)
        {
            // Scalar-tensor subtraction (NOT commutative)
            X::Tensor tensor(input2);
            auto tensorType = tensor->GetDataType();
            float scalar = (float)input1.ToDouble();

            // Calculate total elements
            int dimCount = tensor->GetDimCount();
            long long totalElements = 1;
            for (int i = 0; i < dimCount; i++)
            {
                totalElements *= tensor->GetDimSize(i);
            }

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor\n");
            }

            // Create result tensor with same shape as input
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                resultDims.push_back(tensor->GetDimSize(i));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get GPU memory pointers
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code for scalar-tensor subtraction
            cudaCodeString = mCodeGen.GenerateCommentHeader("Scalar-tensor subtraction", tensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(inputVar, tensorType, gpuData);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensorType, gpuResultData);

            // Add scalar declaration
            cudaCodeString += "float scalar_value = " + std::to_string(scalar) + ";\n";

            // Call the appropriate scalar-tensor subtraction function based on data type
            if (tensorType == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runScalarMinusTensorFP32(scalar_value, " +
                    inputVar + ", " + resultVar + ", " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runScalarMinusTensorFP16(scalar_value, " +
                    inputVar + ", " + resultVar + ", " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runScalarMinusTensorBF16(scalar_value, " +
                    inputVar + ", " + resultVar + ", " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runScalarMinusTensorFP8E4M3(scalar_value, " +
                    inputVar + ", " + resultVar + ", " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runScalarMinusTensorFP8E5M2(scalar_value, " +
                    inputVar + ", " + resultVar + ", " +
                    std::to_string(totalElements) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type for scalar-tensor subtraction\n";
                return X::Value("// Error: Unsupported data type for scalar-tensor subtraction\n");
            }

            return X::Value(cudaCodeString);
        }
        else
        {
            // Scalar-scalar subtraction
            float val1 = (float)input1.ToDouble();
            float val2 = (float)input2.ToDouble();
            cudaCodeString = "// Scalar-scalar subtraction\n";
            cudaCodeString += "float scalar_result = " + std::to_string(val1) + " - " + std::to_string(val2) + ";\n";
            return X::Value(cudaCodeString);
        }
    }

    // Implementation of Matmul function
    X::Value GarnetTensor::Matmul(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        if (isTensor1 && isTensor2)
        {
            // Matrix multiplication between tensors
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);

            // Get tensor information
            auto tensor1_type = tensor1->GetDataType();
            auto tensor2_type = tensor2->GetDataType();
            int dimCount1 = tensor1->GetDimCount();
            int dimCount2 = tensor2->GetDimCount();

            // Validate dimensions for matrix multiplication
            if (dimCount1 != 2 || dimCount2 != 2)
            {
                return X::Value("// Error: Matmul requires 2D tensors (matrices)\n");
            }

            // Check if inner dimensions match for matrix multiplication
            int m = tensor1->GetDimSize(0); // rows of first matrix
            int n = tensor1->GetDimSize(1); // cols of first matrix
            int k = tensor2->GetDimSize(1); // cols of second matrix

            if (n != tensor2->GetDimSize(0))
            {
                return X::Value("// Error: Inner dimensions must match for matrix multiplication\n");
            }

            // Ensure GPU memory is allocated for input tensors
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor1);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor1\n");
            }

            status = TensorHelper::EnsureGPUMemory(tensor2);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for tensor2\n");
            }

            // Create result tensor with shape [m, k]
            X::Port::vector<int> resultDims(2);
            resultDims.push_back(m);
            resultDims.push_back(k);

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensor1_type);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers using TensorHelper
            void* gpuData1 = TensorHelper::GetGPUMemory(tensor1);
            void* gpuData2 = TensorHelper::GetGPUMemory(tensor2);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names based on tensor IDs
            std::string input1Var = mCodeGen.GetTensorName(tensor1);
            std::string input2Var = mCodeGen.GetTensorName(tensor2);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code using helper functions
            cudaCodeString = mCodeGen.GenerateCommentHeader("Matrix multiplication", tensor1, tensor2);

            // Add variable declarations
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input1Var, tensor1_type, gpuData1);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(input2Var, tensor2_type, gpuData2);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensor1_type, gpuResultData);

            // Generate the kernel call based on data types
            if (tensor1_type == X::TensorDataType::FLOAT32 && tensor2_type == X::TensorDataType::FLOAT32) {
                cudaCodeString += "runMatmulFP32(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(n) + ", " +
                    std::to_string(k) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT16 && tensor2_type == X::TensorDataType::FLOAT16) {
                cudaCodeString += "runMatmulFP16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(n) + ", " +
                    std::to_string(k) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::BFLOAT16 && tensor2_type == X::TensorDataType::BFLOAT16) {
                cudaCodeString += "runMatmulBF16(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(n) + ", " +
                    std::to_string(k) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E4M3FN && tensor2_type == X::TensorDataType::FLOAT8_E4M3FN) {
                cudaCodeString += "runMatmulFP8E4M3(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(n) + ", " +
                    std::to_string(k) + ");\n";
            }
            else if (tensor1_type == X::TensorDataType::FLOAT8_E5M2 && tensor2_type == X::TensorDataType::FLOAT8_E5M2) {
                cudaCodeString += "runMatmulFP8E5M2(" + input1Var + ", " + input2Var + ", " + resultVar +
                    ", " + std::to_string(m) + ", " + std::to_string(n) + ", " +
                    std::to_string(k) + ");\n";
            }
            else {
                cudaCodeString += "// Error: Unsupported data type combination for matrix multiplication\n";
                return X::Value("// Error: Unsupported data type combination for matrix multiplication\n");
            }

            return X::Value(cudaCodeString);
        }
        else
        {
            // Matrix multiplication doesn't support scalar-tensor operations
            return X::Value("// Error: Matrix multiplication requires two tensors\n");
        }
    }

    // Implementation of Permute function
    X::Value GarnetTensor::Permute(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
        X::Value input, X::Value& output)
    {
        std::string cudaCodeString;

        if (input.IsTensor())
        {
            X::Tensor tensor(input);
            auto tensorType = tensor->GetDataType();
            int dimCount = tensor->GetDimCount();

            // Permute requires a permutation order as parameter
            if (params.size() == 0)
            {
                return X::Value("// Error: Permute requires a permutation order parameter\n");
            }

            // Extract permutation order from params
            X::Port::vector<int> permOrder;
            for (size_t i = 0; i < params.size(); i++)
            {
                if (params[i].IsNumber())
                {
                    int idx = params[i].ToInt();
                    if (idx < 0 || idx >= dimCount)
                    {
                        return X::Value("// Error: Invalid permutation index\n");
                    }
                    permOrder.push_back(idx);
                }
                else
                {
                    return X::Value("// Error: Permutation indices must be integers\n");
                }
            }

            // Validate permutation order
            if (permOrder.size() != dimCount)
            {
                return X::Value("// Error: Permutation order must match tensor dimensions\n");
            }

            // Check for duplicates in permutation order
            for (int i = 0; i < dimCount; i++)
            {
                for (int j = i + 1; j < dimCount; j++)
                {
                    if (permOrder[i] == permOrder[j])
                    {
                        return X::Value("// Error: Duplicate indices in permutation order\n");
                    }
                }
            }

            // Ensure GPU memory is allocated for input tensor
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for input tensor\n");
            }

            // Create result tensor with permuted shape
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++)
            {
                resultDims.push_back(tensor->GetDimSize(permOrder[i]));
            }

            // Set up the return tensor and register it with the tensor graph
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code for permutation
            cudaCodeString = mCodeGen.GenerateCommentHeader("Tensor permutation", tensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(inputVar, tensorType, gpuData);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, tensorType, gpuResultData);

            // Generate code for permutation order array
            cudaCodeString += "int permOrder[" + std::to_string(dimCount) + "] = {";
            for (int i = 0; i < dimCount; i++)
            {
                cudaCodeString += std::to_string(permOrder[i]);
                if (i < dimCount - 1)
                {
                    cudaCodeString += ", ";
                }
            }
            cudaCodeString += "};\n";

            // Generate code for dimension sizes array
            cudaCodeString += "int dimSizes[" + std::to_string(dimCount) + "] = {";
            for (int i = 0; i < dimCount; i++)
            {
                cudaCodeString += std::to_string(tensor->GetDimSize(i));
                if (i < dimCount - 1)
                {
                    cudaCodeString += ", ";
                }
            }
            cudaCodeString += "};\n";

            // Call permute function based on data type
            if (tensorType == X::TensorDataType::FLOAT32)
            {
                cudaCodeString += "runPermuteFP32(" + inputVar + ", " + resultVar +
                    ", permOrder, dimSizes, " + std::to_string(dimCount) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT16)
            {
                cudaCodeString += "runPermuteFP16(" + inputVar + ", " + resultVar +
                    ", permOrder, dimSizes, " + std::to_string(dimCount) + ");\n";
            }
            else if (tensorType == X::TensorDataType::BFLOAT16)
            {
                cudaCodeString += "runPermuteBF16(" + inputVar + ", " + resultVar +
                    ", permOrder, dimSizes, " + std::to_string(dimCount) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E4M3FN)
            {
                cudaCodeString += "runPermuteFP8E4M3(" + inputVar + ", " + resultVar +
                    ", permOrder, dimSizes, " + std::to_string(dimCount) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT8_E5M2)
            {
                cudaCodeString += "runPermuteFP8E5M2(" + inputVar + ", " + resultVar +
                    ", permOrder, dimSizes, " + std::to_string(dimCount) + ");\n";
            }
            else
            {
                cudaCodeString += "// Error: Unsupported data type for permutation\n";
                return X::Value("// Error: Unsupported data type for permutation\n");
            }

            return X::Value(cudaCodeString);
        }
        else
        {
            return X::Value("// Error: Permute operation requires a tensor input\n");
        }
    }

    // Implementation of Gather function
    X::Value GarnetTensor::Gather(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
        X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        if (isTensor1 && isTensor2)
        {
            // Gather operation: input1 is the tensor to gather from, input2 contains indices
            X::Tensor dataTensor(input1);
            X::Tensor indicesTensor(input2);

            // Get tensor information
            auto dataType = dataTensor->GetDataType();
            auto indicesType = indicesTensor->GetDataType();

            // Gather dimension (default to 0 if not specified)
            int dim = 0;
            auto it = kwParams.find("dim");
            if (it)
            {
                dim = it->val.ToInt();
            }

            // Ensure indices tensor has integer-compatible data type
            if (indicesType != X::TensorDataType::FLOAT32)
            {
                return X::Value("// Error: Indices tensor must have FLOAT32 data type\n");
            }

            // Validate dimensions
            int dataDimCount = dataTensor->GetDimCount();
            int indicesDimCount = indicesTensor->GetDimCount();

            if (dim < 0 || dim >= dataDimCount)
            {
                return X::Value("// Error: Gather dimension out of range\n");
            }

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(dataTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for data tensor\n");
            }

            status = TensorHelper::EnsureGPUMemory(indicesTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for indices tensor\n");
            }

            // Calculate result tensor shape
            X::Port::vector<int> resultDims;
            for (int i = 0; i < dataDimCount; i++)
            {
                if (i == dim)
                {
                    // This dimension comes from indices tensor shape
                    for (int j = 0; j < indicesDimCount; j++)
                    {
                        resultDims.push_back(indicesTensor->GetDimSize(j));
                    }
                }
                else
                {
                    resultDims.push_back(dataTensor->GetDimSize(i));
                }
            }

            // Set up result tensor
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(dataType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers
            void* gpuDataPtr = TensorHelper::GetGPUMemory(dataTensor);
            void* gpuIndicesPtr = TensorHelper::GetGPUMemory(indicesTensor);
            void* gpuResultPtr = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string dataVar = mCodeGen.GetTensorName(dataTensor);
            std::string indicesVar = mCodeGen.GetTensorName(indicesTensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code
            cudaCodeString = mCodeGen.GenerateCommentHeader("Gather operation", dataTensor, indicesTensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(dataVar, dataType, gpuDataPtr);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(indicesVar, indicesType, gpuIndicesPtr);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(resultVar, dataType, gpuResultPtr);

            // Generate code for data tensor dimensions
            cudaCodeString += "int dataDims[" + std::to_string(dataDimCount) + "] = {";
            for (int i = 0; i < dataDimCount; i++)
            {
                cudaCodeString += std::to_string(dataTensor->GetDimSize(i));
                if (i < dataDimCount - 1)
                {
                    cudaCodeString += ", ";
                }
            }
            cudaCodeString += "};\n";

            // Generate code for indices tensor dimensions
            cudaCodeString += "int indicesDims[" + std::to_string(indicesDimCount) + "] = {";
            for (int i = 0; i < indicesDimCount; i++)
            {
                cudaCodeString += std::to_string(indicesTensor->GetDimSize(i));
                if (i < indicesDimCount - 1)
                {
                    cudaCodeString += ", ";
                }
            }
            cudaCodeString += "};\n";

            // Call gather function based on data type
            if (dataType == X::TensorDataType::FLOAT32)
            {
                cudaCodeString += "runGatherFP32(" + dataVar + ", " + indicesVar + ", " + resultVar +
                    ", dataDims, " + std::to_string(dataDimCount) +
                    ", indicesDims, " + std::to_string(indicesDimCount) +
                    ", " + std::to_string(dim) + ");\n";
            }
            else if (dataType == X::TensorDataType::FLOAT16)
            {
                cudaCodeString += "runGatherFP16(" + dataVar + ", " + indicesVar + ", " + resultVar +
                    ", dataDims, " + std::to_string(dataDimCount) +
                    ", indicesDims, " + std::to_string(indicesDimCount) +
                    ", " + std::to_string(dim) + ");\n";
            }
            else if (dataType == X::TensorDataType::BFLOAT16)
            {
                cudaCodeString += "runGatherBF16(" + dataVar + ", " + indicesVar + ", " + resultVar +
                    ", dataDims, " + std::to_string(dataDimCount) +
                    ", indicesDims, " + std::to_string(indicesDimCount) +
                    ", " + std::to_string(dim) + ");\n";
            }
            else if (dataType == X::TensorDataType::FLOAT8_E4M3FN)
            {
                cudaCodeString += "runGatherFP8E4M3(" + dataVar + ", " + indicesVar + ", " + resultVar +
                    ", dataDims, " + std::to_string(dataDimCount) +
                    ", indicesDims, " + std::to_string(indicesDimCount) +
                    ", " + std::to_string(dim) + ");\n";
            }
            else if (dataType == X::TensorDataType::FLOAT8_E5M2)
            {
                cudaCodeString += "runGatherFP8E5M2(" + dataVar + ", " + indicesVar + ", " + resultVar +
                    ", dataDims, " + std::to_string(dataDimCount) +
                    ", indicesDims, " + std::to_string(indicesDimCount) +
                    ", " + std::to_string(dim) + ");\n";
            }
            else
            {
                cudaCodeString += "// Error: Unsupported data type for gather operation\n";
                return X::Value("// Error: Unsupported data type for gather operation\n");
            }

            return X::Value(cudaCodeString);
        }
        else
        {
            return X::Value("// Error: Gather operation requires two tensors\n");
        }
    }

    // Implementation of Convert function
    X::Value GarnetTensor::Convert(X::Value& graph, X::ARGS& params, X::KWARGS& kwParams,
        X::Value input, X::Value& output)
    {
        std::string cudaCodeString;

        if (input.IsTensor())
        {
            X::Tensor tensor(input);
            auto srcType = tensor->GetDataType();

            // Target data type must be specified in kwParams
            auto it = kwParams.find("dtype");
            if (it == nullptr)
            {
                return X::Value("// Error: Convert operation requires dtype parameter\n");
            }

            // Parse target data type from kwParams
            std::string dtypeStr = it->val.ToString();
            X::TensorDataType destType;

            if (dtypeStr == "float32" || dtypeStr == "float")
            {
                destType = X::TensorDataType::FLOAT32;
            }
            else if (dtypeStr == "float16" || dtypeStr == "half")
            {
                destType = X::TensorDataType::FLOAT16;
            }
            else if (dtypeStr == "bfloat16")
            {
                destType = X::TensorDataType::BFLOAT16;
            }
            else if (dtypeStr == "float8_e4m3fn")
            {
                destType = X::TensorDataType::FLOAT8_E4M3FN;
            }
            else if (dtypeStr == "float8_e5m2")
            {
                destType = X::TensorDataType::FLOAT8_E5M2;
            }
            else
            {
                return X::Value("// Error: Unsupported target data type for conversion\n");
            }

            // Skip if source and destination types are the same
            if (srcType == destType)
            {
                cudaCodeString = "// Source and destination types are the same, no conversion needed\n";
                return X::Value(cudaCodeString);
            }

            // Ensure GPU memory is allocated for input tensor
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for input tensor\n");
            }

            // Create result tensor with same shape but different data type
            int dimCount = tensor->GetDimCount();
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++)
            {
                resultDims.push_back(tensor->GetDimSize(i));
            }

            // Calculate total elements
            long long totalElements = 1;
            for (int i = 0; i < dimCount; i++)
            {
                totalElements *= tensor->GetDimSize(i);
            }

            // Set up result tensor
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(destType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success)
            {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Get memory pointers
            void* gpuSrcPtr = TensorHelper::GetGPUMemory(tensor);
            void* gpuDestPtr = TensorHelper::GetGPUMemory(resultTensor);

            // Create variable names
            std::string srcVar = mCodeGen.GetTensorName(tensor);
            std::string destVar = mCodeGen.GetTensorName(resultTensor);

            // Generate CUDA code
            cudaCodeString = mCodeGen.GenerateCommentHeader("Data type conversion", tensor);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(srcVar, srcType, gpuSrcPtr);
            cudaCodeString += mCodeGen.GenerateVariableDeclaration(destVar, destType, gpuDestPtr);

            // Generate function call for the specific conversion
            std::string srcTypeStr;
            std::string destTypeStr;

            // Map data types to strings for function name
            switch (srcType) {
            case X::TensorDataType::FLOAT32: srcTypeStr = "FP32"; break;
            case X::TensorDataType::FLOAT16: srcTypeStr = "FP16"; break;
            case X::TensorDataType::BFLOAT16: srcTypeStr = "BF16"; break;
            case X::TensorDataType::FLOAT8_E4M3FN: srcTypeStr = "FP8E4M3"; break;
            case X::TensorDataType::FLOAT8_E5M2: srcTypeStr = "FP8E5M2"; break;
            default: return X::Value("// Error: Unsupported source data type for conversion\n");
            }

            switch (destType) {
            case X::TensorDataType::FLOAT32: destTypeStr = "FP32"; break;
            case X::TensorDataType::FLOAT16: destTypeStr = "FP16"; break;
            case X::TensorDataType::BFLOAT16: destTypeStr = "BF16"; break;
            case X::TensorDataType::FLOAT8_E4M3FN: destTypeStr = "FP8E4M3"; break;
            case X::TensorDataType::FLOAT8_E5M2: destTypeStr = "FP8E5M2"; break;
            default: return X::Value("// Error: Unsupported destination data type for conversion\n");
            }

            cudaCodeString += "runConvert" + srcTypeStr + "To" + destTypeStr + "(" +
                srcVar + ", " + destVar + ", " + std::to_string(totalElements) + ");\n";

            return X::Value(cudaCodeString);
        }
        else
        {
            return X::Value("// Error: Convert operation requires a tensor input\n");
        }
    }
}