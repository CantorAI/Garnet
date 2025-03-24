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
    X::Value GarnetTensor::Header(X::Value& graph, X::ARGS& params)
    {
        // Ensure that at least one parameter (the XLang function) is provided.
        if (params.size() < 1) {
            return X::Value("// Error: Insufficient parameters: expected at least one parameter (the XLang function)\n");
        }

        // The first parameter is a XLang function.
        X::Func func(params[0]);
        std::string cudaFunctionName = func->GetName().ToString();

        // Retrieve the parameter name list from the function.
        X::Value valParamNames = func->GetParameterNameList();
        X::List nameList(valParamNames);

        // Check if the provided params include values for all function parameters.
        if (params.size() < (nameList->Size() + 1)) {
            return X::Value("// Error: Insufficient parameters provided to Header function: expected " +
                std::to_string(nameList->Size() + 1) + " but got " + std::to_string(params.size()) + "\n");
        }

        // Build the parameter declaration list.
        std::string paramListStr;
        // Loop over each parameter name.
        // Note: for each parameter name at index i in nameList,
        // the corresponding calling argument is at params[i+1].
        for (int i = 0; i < (int)nameList->Size(); i++) {
            X::Value nameVal;
            nameList->GetIndexValue(i, nameVal);
            std::string paramName = nameVal.ToString();
            // Get the calling parameter (could be a tensor or a scalar).
            X::Value argValue = params[i + 1];
            std::string typeStr;

            if (argValue.IsTensor()) {
                // For tensors, cast to X::Tensor.
                X::Tensor tensor(argValue);
                // Check if the tensor is a single element.
                if (tensor->GetCount() == 1) {
                    // Single-element tensor treated as a scalar.
                    auto dt = tensor->GetDataType();
                    if (dt == X::TensorDataType::DOUBLE) {
                        typeStr = "double";
                    }
                    else if (dt == X::TensorDataType::FLOAT32) {
                        typeStr = "float";
                    }
                    else if (dt == X::TensorDataType::FLOAT16) {
                        typeStr = "__half";
                    }
                    else if (dt == X::TensorDataType::BFLOAT16) {
                        typeStr = "bfloat16";
                    }
                    else if (dt == X::TensorDataType::FLOAT8_E4M3FN) {
                        typeStr = "fp8_e4m3";  // Adjust as needed for your type name.
                    }
                    else if (dt == X::TensorDataType::FLOAT8_E5M2) {
                        typeStr = "fp8_e5m2";
                    }
                    else {
                        typeStr = "float"; // Fallback type.
                    }
                }
                else {
                    // Multi-element tensor: declare as a pointer.
                    auto dt = tensor->GetDataType();
                    if (dt == X::TensorDataType::DOUBLE) {
                        typeStr = "double*";
                    }
                    else if (dt == X::TensorDataType::FLOAT32) {
                        typeStr = "float*";
                    }
                    else if (dt == X::TensorDataType::FLOAT16) {
                        typeStr = "__half*";
                    }
                    else if (dt == X::TensorDataType::BFLOAT16) {
                        typeStr = "bfloat16*";
                    }
                    else if (dt == X::TensorDataType::FLOAT8_E4M3FN) {
                        typeStr = "fp8_e4m3*";
                    }
                    else if (dt == X::TensorDataType::FLOAT8_E5M2) {
                        typeStr = "fp8_e5m2*";
                    }
                    else {
                        typeStr = "float*"; // Fallback.
                    }
                }
            }
            else {
                // Non-tensor values are treated as scalar (default to float).
                typeStr = "float";
            }

            // Append the parameter declaration.
            paramListStr += typeStr + " " + paramName;
            if (i < (int)nameList->Size() - 1) {
                paramListStr += ", ";
            }
        }

        // Generate the header code with necessary includes.
        std::string headerCode =
            "#include <cuda_runtime.h>\n"
            "#include <cuda_fp16.h>\n"
            "#include <cuda_bf16.h>\n"
            "#include <cuda_fp8.h>\n\n"
            "// Include CUDA function declarations\n"
            "#include \"cuda_templates/cuda_function_declarations.h\"\n\n";

        // Use the obtained function name and parameter list to generate the function header.
        headerCode += "extern \"C\" void " + cudaFunctionName + "(" + paramListStr + ") {\n";

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
        X::KWARGS& kwParams, X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        // Lambda for creating tensor-tensor multiplication code
        auto generateMatrixMulCode = [&](X::Tensor& t1, X::Tensor& t2, X::Tensor& result,
            int m, int n, int k) -> std::string {

                std::string code = mCodeGen.GenerateCommentHeader("Matrix multiplication", t1, t2);
                auto t1_type = t1->GetDataType();
                auto t2_type = t2->GetDataType();

                // Get memory pointers
                void* gpu1 = TensorHelper::GetGPUMemory(t1);
                void* gpu2 = TensorHelper::GetGPUMemory(t2);
                void* gpuResult = TensorHelper::GetGPUMemory(result);

                // Create variable names
                std::string var1 = mCodeGen.GetTensorName(t1);
                std::string var2 = mCodeGen.GetTensorName(t2);
                std::string resultVar = mCodeGen.GetTensorName(result);

                // Add variable declarations
                code += mCodeGen.GenerateVariableDeclaration(var1, t1_type, gpu1);
                code += mCodeGen.GenerateVariableDeclaration(var2, t2_type, gpu2);
                code += mCodeGen.GenerateVariableDeclaration(resultVar, t1_type, gpuResult);

                // Generate kernel calls based on data types
                if (t1_type == X::TensorDataType::DOUBLE && t2_type == X::TensorDataType::DOUBLE) {
                    code += "runGemmFP64(" + var1 + ", " + var2 + ", " + resultVar +
                        ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                        std::to_string(n) + ");\n";
                }
                else if (t1_type == X::TensorDataType::FLOAT32 && t2_type == X::TensorDataType::FLOAT32) {
                    code += "runGemmFP32(" + var1 + ", " + var2 + ", " + resultVar +
                        ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                        std::to_string(n) + ");\n";
                }
                else if (t1_type == X::TensorDataType::FLOAT16 && t2_type == X::TensorDataType::FLOAT16) {
                    code += "runGemmFP16(" + var1 + ", " + var2 + ", " + resultVar +
                        ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                        std::to_string(n) + ");\n";
                }
                else if (t1_type == X::TensorDataType::BFLOAT16 && t2_type == X::TensorDataType::BFLOAT16) {
                    code += "runGemmBF16(" + var1 + ", " + var2 + ", " + resultVar +
                        ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                        std::to_string(n) + ");\n";
                }
                else if (t1_type == X::TensorDataType::FLOAT8_E4M3FN && t2_type == X::TensorDataType::FLOAT8_E4M3FN) {
                    code += "runGemmFP8E4M3(" + var1 + ", " + var2 + ", " + resultVar +
                        ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                        std::to_string(n) + ");\n";
                }
                else if (t1_type == X::TensorDataType::FLOAT8_E5M2 && t2_type == X::TensorDataType::FLOAT8_E5M2) {
                    code += "runGemmFP8E5M2(" + var1 + ", " + var2 + ", " + resultVar +
                        ", " + std::to_string(m) + ", " + std::to_string(k) + ", " +
                        std::to_string(n) + ");\n";
                }
                else {
                    code += "// Error: Unsupported data type combination\n";
                }

                return code;
            };

        if (isTensor1 && isTensor2)
        {
            // Tensor-tensor multiplication
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);
            auto tensor1_type = tensor1->GetDataType();
            auto tensor2_type = tensor2->GetDataType();

            // Optimized path for single element tensors
            if (tensor1->GetCount() == 1 && tensor2->GetCount() == 1) {
                // Direct expression for single element operations
                cudaCodeString = "// Direct single element multiplication\n";
                cudaCodeString += "*static_cast<float*>(" + mCodeGen.GetTensorName(tensor1) +
                    ") * *static_cast<float*>(" + mCodeGen.GetTensorName(tensor2) + ");\n";
                return X::Value(cudaCodeString);
            }

            // Normal tensor-tensor processing
            int dimCount1 = tensor1->GetDimCount();
            int dimCount2 = tensor2->GetDimCount();

            // Validate dimensions
            if (dimCount1 > 2 || dimCount2 > 2) {
                return X::Value("// Error: Tensor dimensions > 2 not supported\n");
            }

            int m = tensor1->GetDimSize(0);
            int n = (dimCount1 > 1) ? tensor1->GetDimSize(1) : 1;
            int k = (dimCount2 > 1) ? tensor2->GetDimSize(1) : 1;

            // Check dimension compatibility
            if (dimCount2 == 1) {
                if (n != tensor2->GetDimSize(0)) {
                    return X::Value("// Error: Dimension mismatch for vector multiplication\n");
                }
            }
            else if (n != tensor2->GetDimSize(0)) {
                return X::Value("// Error: Inner dimensions must match for matrix multiplication\n");
            }

            // Ensure GPU memory is allocated for input tensors
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor1);
            if (status != TensorOpStatus::Success) {
                return X::Value("// Error: Failed to allocate GPU memory for tensor1\n");
            }

            status = TensorHelper::EnsureGPUMemory(tensor2);
            if (status != TensorOpStatus::Success) {
                return X::Value("// Error: Failed to allocate GPU memory for tensor2\n");
            }

            // Create result tensor with proper dimensions
            int dimNum = 1;
            if (tensor2->GetDimCount() > 1) {
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
            if (status != TensorOpStatus::Success) {
                return X::Value("// Error: Failed to allocate GPU memory for result tensor\n");
            }

            // Generate CUDA code using the lambda
            cudaCodeString = generateMatrixMulCode(tensor1, tensor2, resultTensor, m, n, k);
            return X::Value(cudaCodeString);
        }
        else if (isTensor1)
        {
            // Tensor-scalar multiplication
            X::Tensor tensor(input1);
            auto tensorType = tensor->GetDataType();
            float scalar = (float)input2.ToDouble();

            // Optimized path for single element tensor
            if (tensor->GetCount() == 1) {
                cudaCodeString = "// Direct single element scalar multiplication\n";
                cudaCodeString += "*static_cast<float*>(" + mCodeGen.GetTensorName(tensor) +
                    ") * " + std::to_string(scalar) + ";\n";
                return X::Value(cudaCodeString);
            }

            // Standard tensor-scalar logic
            int dimCount = tensor->GetDimCount();
            if (dimCount > 2) {
                return X::Value("// Error: Tensor dimensions > 2 not supported for scalar multiplication\n");
            }

            // Calculate total elements
            long long totalElements = tensor->GetCount();

            // Ensure GPU memory is allocated
            TensorOpStatus status = TensorHelper::EnsureGPUMemory(tensor);
            if (status != TensorOpStatus::Success) {
                return X::Value("// Error: Failed to allocate GPU memory for tensor\n");
            }

            // Create result tensor with same shape as input
            X::Port::vector<int> resultDims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                resultDims.push_back(tensor->GetDimSize(i));
            }

            // Set up the return tensor
            X::Tensor resultTensor(output);
            X::TensorGraph tensorGraph(graph);
            resultTensor->SetDataType(tensorType);
            resultTensor->SetShape(resultDims);
            X::Value initData;
            resultTensor->Create(initData);
            tensorGraph->PutTensorIntoCache(resultTensor);

            status = TensorHelper::EnsureGPUMemory(resultTensor);
            if (status != TensorOpStatus::Success) {
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
            if (tensorType == X::TensorDataType::DOUBLE) {
                cudaCodeString += "runScalarMultiplyFP64(" +
                    inputVar + ", " + resultVar + ", scalar_value, " +
                    std::to_string(totalElements) + ");\n";
            }
            else if (tensorType == X::TensorDataType::FLOAT32) {
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
            return Multiply(graph, params, kwParams, input2, input1, output);
        }
        else
        {
            // Scalar-scalar multiplication
            float val1 = (float)input1.ToDouble();
            float val2 = (float)input2.ToDouble();
            cudaCodeString = "// Scalar-scalar multiplication\n";
            cudaCodeString += std::to_string(val1) + " * " + std::to_string(val2) + ";\n";
            return X::Value(cudaCodeString);
        }
    }

    // Implementation of Add function
    X::Value GarnetTensor::Add(X::Value& graph, X::ARGS& params,
        X::KWARGS& kwParams, X::Value input1, X::Value input2, X::Value& output)
    {
        std::string cudaCodeString;
        bool isTensor1 = input1.IsTensor();
        bool isTensor2 = input2.IsTensor();

        // Lambda to prepare a result tensor given a source tensor¡¯s shape and data type.
        auto prepareResultTensor = [&](X::Tensor srcTensor, X::Tensor resultTensor) -> bool {
            int dimCount = srcTensor->GetDimCount();
            // Initialize vector with count 'dimCount'
            X::Port::vector<int> dims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                dims[i] = srcTensor->GetDimSize(i);
            }
            resultTensor->SetDataType(srcTensor->GetDataType());
            resultTensor->SetShape(dims);
            X::TensorGraph tensorGraph(graph);
            X::Value initData;
            resultTensor->Create(initData);
            return (TensorHelper::EnsureGPUMemory(resultTensor) == TensorOpStatus::Success);
            };

        // Lambda for direct scalar-scalar addition.
        auto addScalars = [&](float a, float b) -> std::string {
            std::string code = "// Scalar-scalar addition\n";
            code += std::to_string(a) + " + " + std::to_string(b) + ";\n";
            return code;
            };

        // Lambda to handle the case when one tensor is a single element (scalar)
        // and the other is multi-element. For the single element tensor, we pass its value directly.
        auto addSingleElementTensor = [&](X::Tensor singleTensor, X::Tensor multiTensor, X::Tensor resultTensor) -> std::string {
            if (TensorHelper::EnsureGPUMemory(multiTensor) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for multi-element tensor\n";
            if (!prepareResultTensor(multiTensor, resultTensor))
                return "// Error: Failed to allocate GPU memory for result tensor\n";

            std::string code = mCodeGen.GenerateCommentHeader("Single element tensor as scalar addition", singleTensor, multiTensor);

            std::string singleVar = mCodeGen.GetTensorName(singleTensor);
            std::string multiVar = mCodeGen.GetTensorName(multiTensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);
            void* gpuMulti = TensorHelper::GetGPUMemory(multiTensor);
            void* gpuResult = TensorHelper::GetGPUMemory(resultTensor);

            code += mCodeGen.GenerateVariableDeclaration(multiVar, multiTensor->GetDataType(), gpuMulti);
            code += mCodeGen.GenerateVariableDeclaration(resultVar, multiTensor->GetDataType(), gpuResult);

            auto singleType = singleTensor->GetDataType();
            auto multiType = multiTensor->GetDataType();
            // Note: The correct order is: multi-element tensor variable first, then single element tensor variable.
            if (singleType != multiType) {
                code += "// Converting single element tensor to match multi tensor type\n";
                if (multiType == X::TensorDataType::DOUBLE) {
                    code += "runSingleElementTensorAddFP64(" + multiVar + ", (double)" + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT32) {
                    code += "runSingleElementTensorAddFP32(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT16) {
                    code += "runSingleElementTensorAddFP16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::BFLOAT16) {
                    code += "runSingleElementTensorAddBF16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT8_E4M3FN) {
                    code += "runSingleElementTensorAddFP8E4M3(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT8_E5M2) {
                    code += "runSingleElementTensorAddFP8E5M2(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else {
                    code += "// Error: Unsupported data type for conversion\n";
                }
            }
            else {
                if (singleType == X::TensorDataType::DOUBLE) {
                    code += "runSingleElementTensorAddFP64(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT32) {
                    code += "runSingleElementTensorAddFP32(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT16) {
                    code += "runSingleElementTensorAddFP16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::BFLOAT16) {
                    code += "runSingleElementTensorAddBF16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT8_E4M3FN) {
                    code += "runSingleElementTensorAddFP8E4M3(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT8_E5M2) {
                    code += "runSingleElementTensorAddFP8E5M2(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else {
                    code += "// Error: Unsupported data type for single element tensor addition\n";
                }
            }
            return code;
        };

        // Lambda for multi-element tensor-tensor addition.
        auto addTensors = [&](X::Tensor t1, X::Tensor t2, X::Tensor resultTensor, long long totalElements) -> std::string {
            if (TensorHelper::EnsureGPUMemory(t1) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for first tensor\n";
            if (TensorHelper::EnsureGPUMemory(t2) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for second tensor\n";
            if (!prepareResultTensor(t1, resultTensor))
                return "// Error: Failed to allocate GPU memory for result tensor\n";

            std::string code = mCodeGen.GenerateCommentHeader("Tensor addition", t1, t2);
            void* gpu1 = TensorHelper::GetGPUMemory(t1);
            void* gpu2 = TensorHelper::GetGPUMemory(t2);
            void* gpuResult = TensorHelper::GetGPUMemory(resultTensor);
            std::string var1 = mCodeGen.GetTensorName(t1);
            std::string var2 = mCodeGen.GetTensorName(t2);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);
            code += mCodeGen.GenerateVariableDeclaration(var1, t1->GetDataType(), gpu1);
            code += mCodeGen.GenerateVariableDeclaration(var2, t2->GetDataType(), gpu2);
            code += mCodeGen.GenerateVariableDeclaration(resultVar, t1->GetDataType(), gpuResult);

            if (t1->GetDataType() == X::TensorDataType::DOUBLE &&
                t2->GetDataType() == X::TensorDataType::DOUBLE)
            {
                code += "runAddFP64(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT32 &&
                t2->GetDataType() == X::TensorDataType::FLOAT32)
            {
                code += "runAddFP32(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT16 &&
                t2->GetDataType() == X::TensorDataType::FLOAT16)
            {
                code += "runAddFP16(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::BFLOAT16 &&
                t2->GetDataType() == X::TensorDataType::BFLOAT16)
            {
                code += "runAddBF16(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT8_E4M3FN &&
                t2->GetDataType() == X::TensorDataType::FLOAT8_E4M3FN)
            {
                code += "runAddFP8E4M3(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT8_E5M2 &&
                t2->GetDataType() == X::TensorDataType::FLOAT8_E5M2)
            {
                code += "runAddFP8E5M2(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else {
                code += "// Error: Unsupported data type combination for tensor addition\n";
            }
            return code;
            };

        // Lambda for tensor-scalar addition.
        auto addTensorScalar = [&](X::Tensor tensor, float scalar, X::Tensor resultTensor) -> std::string {
            if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for tensor\n";
            if (!prepareResultTensor(tensor, resultTensor))
                return "// Error: Failed to allocate GPU memory for result tensor\n";

            std::string code = mCodeGen.GenerateCommentHeader("Scalar addition to tensor", tensor);
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);
            code += mCodeGen.GenerateVariableDeclaration(inputVar, tensor->GetDataType(), gpuData);
            code += mCodeGen.GenerateVariableDeclaration(resultVar, tensor->GetDataType(), gpuResultData);
            // The correct order: first parameter is the tensor, second is the scalar, third is the result.
            code += "runScalarAddFP32(" + inputVar + ", " + std::to_string(scalar) + ", " + resultVar +
                ", " + std::to_string(tensor->GetCount()) + ");\n";
            return code;
        };


        // Main decision logic based on whether the inputs are tensors or scalars.
        if (isTensor1 && isTensor2)
        {
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);
            X::Tensor resultTensor(output);

            bool isSingleElement1 = (tensor1->GetCount() == 1);
            bool isSingleElement2 = (tensor2->GetCount() == 1);

            // Both tensors are single elements ¨C perform direct scalar addition.
            if (isSingleElement1 && isSingleElement2) {
                std::string resultVar = mCodeGen.GetTensorName(resultTensor);
                cudaCodeString = "// Direct single element addition\n";
                cudaCodeString += resultVar + " = " +
                    mCodeGen.GetTensorName(tensor1) + " + " +
                    mCodeGen.GetTensorName(tensor2) + ";\n";
                return X::Value(cudaCodeString);
            }
            // One tensor is a single element and the other is multi-element.
            else if (isSingleElement1 && !isSingleElement2) {
                cudaCodeString = addSingleElementTensor(tensor1, tensor2, resultTensor);
                return X::Value(cudaCodeString);
            }
            else if (!isSingleElement1 && isSingleElement2) {
                cudaCodeString = addSingleElementTensor(tensor2, tensor1, resultTensor);
                return X::Value(cudaCodeString);
            }
            // Both tensors are multi-element.
            else {
                if (tensor1->GetDimCount() != tensor2->GetDimCount()) {
                    return X::Value("// Error: Tensors must have the same number of dimensions for binary operations\n");
                }
                for (int i = 0; i < tensor1->GetDimCount(); i++) {
                    if (tensor1->GetDimSize(i) != tensor2->GetDimSize(i)) {
                        return X::Value("// Error: Tensor dimensions must match for binary operations\n");
                    }
                }
                long long totalElements = tensor1->GetCount();
                cudaCodeString = addTensors(tensor1, tensor2, resultTensor, totalElements);
                return X::Value(cudaCodeString);
            }
        }
        else if (isTensor1 && !isTensor2)
        {
            // Tensor-scalar addition.
            X::Tensor tensor(input1);
            X::Tensor resultTensor(output);
            if (tensor->GetCount() == 1) {
                std::string resultVar = mCodeGen.GetTensorName(resultTensor);
                cudaCodeString = "// Direct single element scalar addition\n";
                cudaCodeString += resultVar + " = " +
                    mCodeGen.GetTensorName(tensor) + " + " +
                    std::to_string((float)input2.ToDouble()) + ";\n";
                return X::Value(cudaCodeString);
            }
            else {
                cudaCodeString = addTensorScalar(tensor, (float)input2.ToDouble(), resultTensor);
                return X::Value(cudaCodeString);
            }
        }
        else if (!isTensor1 && isTensor2)
        {
            // For commutative scalar-tensor addition, swap the inputs.
            return Add(graph, params, kwParams, input2, input1, output);
        }
        else
        {
            // Both inputs are scalars.
            float val1 = (float)input1.ToDouble();
            float val2 = (float)input2.ToDouble();
            cudaCodeString = addScalars(val1, val2);
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

        // Lambda to prepare a result tensor using the source tensor's shape and data type.
        auto prepareResultTensor = [&](X::Tensor srcTensor, X::Tensor resultTensor) -> bool {
            int dimCount = srcTensor->GetDimCount();
            // Initialize vector with count 'dimCount'
            X::Port::vector<int> dims(dimCount);
            for (int i = 0; i < dimCount; i++) {
                dims[i] = srcTensor->GetDimSize(i);
            }
            resultTensor->SetDataType(srcTensor->GetDataType());
            resultTensor->SetShape(dims);
            X::TensorGraph tensorGraph(graph);
            X::Value initData;
            resultTensor->Create(initData);
            return (TensorHelper::EnsureGPUMemory(resultTensor) == TensorOpStatus::Success);
            };

        // Lambda for direct scalar-scalar subtraction.
        auto subtractScalars = [&](float a, float b) -> std::string {
            std::string code = "// Scalar-scalar subtraction\n";
            code += std::to_string(a) + " - " + std::to_string(b) + ";\n";
            return code;
            };

        // Lambda to handle the case when one tensor is a single element (scalar)
        // and the other is a multi-element tensor.
        // The single element tensor's value is passed directly.
        auto subtractSingleElementTensor = [&](X::Tensor singleTensor, X::Tensor multiTensor, X::Tensor resultTensor) -> std::string {
            if (TensorHelper::EnsureGPUMemory(multiTensor) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for multi-element tensor\n";
            if (!prepareResultTensor(multiTensor, resultTensor))
                return "// Error: Failed to allocate GPU memory for result tensor\n";

            std::string code = mCodeGen.GenerateCommentHeader("Single element tensor as scalar subtraction", singleTensor, multiTensor);

            std::string singleVar = mCodeGen.GetTensorName(singleTensor);
            std::string multiVar = mCodeGen.GetTensorName(multiTensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);
            void* gpuMulti = TensorHelper::GetGPUMemory(multiTensor);
            void* gpuResult = TensorHelper::GetGPUMemory(resultTensor);

            code += mCodeGen.GenerateVariableDeclaration(multiVar, multiTensor->GetDataType(), gpuMulti);
            code += mCodeGen.GenerateVariableDeclaration(resultVar, multiTensor->GetDataType(), gpuResult);

            auto singleType = singleTensor->GetDataType();
            auto multiType = multiTensor->GetDataType();
            // Correct parameter order: first argument is the multi-element tensor, then the single-element tensor.
            if (singleType != multiType) {
                code += "// Converting single element tensor to match multi tensor type\n";
                if (multiType == X::TensorDataType::DOUBLE) {
                    code += "runSingleElementTensorMinusFP64(" + multiVar + ", (double)" + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT32) {
                    code += "runSingleElementTensorMinusFP32(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT16) {
                    code += "runSingleElementTensorMinusFP16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::BFLOAT16) {
                    code += "runSingleElementTensorMinusBF16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT8_E4M3FN) {
                    code += "runSingleElementTensorMinusFP8E4M3(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (multiType == X::TensorDataType::FLOAT8_E5M2) {
                    code += "runSingleElementTensorMinusFP8E5M2(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else {
                    code += "// Error: Unsupported data type for conversion\n";
                }
            }
            else {
                if (singleType == X::TensorDataType::DOUBLE) {
                    code += "runSingleElementTensorMinusFP64(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT32) {
                    code += "runSingleElementTensorMinusFP32(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT16) {
                    code += "runSingleElementTensorMinusFP16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::BFLOAT16) {
                    code += "runSingleElementTensorMinusBF16(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT8_E4M3FN) {
                    code += "runSingleElementTensorMinusFP8E4M3(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else if (singleType == X::TensorDataType::FLOAT8_E5M2) {
                    code += "runSingleElementTensorMinusFP8E5M2(" + multiVar + ", " + singleVar + ", " + resultVar +
                        ", " + std::to_string(multiTensor->GetCount()) + ");\n";
                }
                else {
                    code += "// Error: Unsupported data type for single element tensor subtraction\n";
                }
            }
            return code;
            };

        // Lambda for multi-element tensor-tensor subtraction.
        auto subtractTensors = [&](X::Tensor t1, X::Tensor t2, X::Tensor resultTensor, long long totalElements) -> std::string {
            if (TensorHelper::EnsureGPUMemory(t1) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for first tensor\n";
            if (TensorHelper::EnsureGPUMemory(t2) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for second tensor\n";
            if (!prepareResultTensor(t1, resultTensor))
                return "// Error: Failed to allocate GPU memory for result tensor\n";

            std::string code = mCodeGen.GenerateCommentHeader("Tensor subtraction", t1, t2);
            void* gpu1 = TensorHelper::GetGPUMemory(t1);
            void* gpu2 = TensorHelper::GetGPUMemory(t2);
            void* gpuResult = TensorHelper::GetGPUMemory(resultTensor);
            std::string var1 = mCodeGen.GetTensorName(t1);
            std::string var2 = mCodeGen.GetTensorName(t2);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);
            code += mCodeGen.GenerateVariableDeclaration(var1, t1->GetDataType(), gpu1);
            code += mCodeGen.GenerateVariableDeclaration(var2, t2->GetDataType(), gpu2);
            code += mCodeGen.GenerateVariableDeclaration(resultVar, t1->GetDataType(), gpuResult);

            if (t1->GetDataType() == X::TensorDataType::DOUBLE &&
                t2->GetDataType() == X::TensorDataType::DOUBLE)
            {
                code += "runMinusFP64(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT32 &&
                t2->GetDataType() == X::TensorDataType::FLOAT32)
            {
                code += "runMinusFP32(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT16 &&
                t2->GetDataType() == X::TensorDataType::FLOAT16)
            {
                code += "runMinusFP16(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::BFLOAT16 &&
                t2->GetDataType() == X::TensorDataType::BFLOAT16)
            {
                code += "runMinusBF16(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT8_E4M3FN &&
                t2->GetDataType() == X::TensorDataType::FLOAT8_E4M3FN)
            {
                code += "runMinusFP8E4M3(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else if (t1->GetDataType() == X::TensorDataType::FLOAT8_E5M2 &&
                t2->GetDataType() == X::TensorDataType::FLOAT8_E5M2)
            {
                code += "runMinusFP8E5M2(" + var1 + ", " + var2 + ", " + resultVar +
                    ", " + std::to_string(totalElements) + ");\n";
            }
            else {
                code += "// Error: Unsupported data type combination for tensor subtraction\n";
            }
            return code;
            };

        // Lambda for tensor-scalar subtraction.
        auto subtractTensorScalar = [&](X::Tensor tensor, float scalar, X::Tensor resultTensor) -> std::string {
            if (TensorHelper::EnsureGPUMemory(tensor) != TensorOpStatus::Success)
                return "// Error: Failed to allocate GPU memory for tensor\n";
            if (!prepareResultTensor(tensor, resultTensor))
                return "// Error: Failed to allocate GPU memory for result tensor\n";

            std::string code = mCodeGen.GenerateCommentHeader("Scalar subtraction from tensor", tensor);
            void* gpuData = TensorHelper::GetGPUMemory(tensor);
            void* gpuResultData = TensorHelper::GetGPUMemory(resultTensor);
            std::string inputVar = mCodeGen.GetTensorName(tensor);
            std::string resultVar = mCodeGen.GetTensorName(resultTensor);
            code += mCodeGen.GenerateVariableDeclaration(inputVar, tensor->GetDataType(), gpuData);
            code += mCodeGen.GenerateVariableDeclaration(resultVar, tensor->GetDataType(), gpuResultData);
            // Correct order: first the tensor, then the scalar, then the result.
            code += "runScalarMinusFP32(" + inputVar + ", " + std::to_string(scalar) + ", " + resultVar +
                ", " + std::to_string(tensor->GetCount()) + ");\n";
            return code;
            };

        // Main decision logic based on whether the inputs are tensors or scalars.
        if (isTensor1 && isTensor2)
        {
            X::Tensor tensor1(input1);
            X::Tensor tensor2(input2);
            X::Tensor resultTensor(output);

            bool isSingleElement1 = (tensor1->GetCount() == 1);
            bool isSingleElement2 = (tensor2->GetCount() == 1);

            // Both tensors are single elements ¨C perform direct scalar subtraction.
            if (isSingleElement1 && isSingleElement2) {
                std::string resultVar = mCodeGen.GetTensorName(resultTensor);
                cudaCodeString = "// Direct single element subtraction\n";
                cudaCodeString += resultVar + " = " +
                    mCodeGen.GetTensorName(tensor1) + " - " +
                    mCodeGen.GetTensorName(tensor2) + ";\n";
                return X::Value(cudaCodeString);
            }
            // One tensor is a single element and the other is multi-element.
            else if (isSingleElement1 && !isSingleElement2) {
                cudaCodeString = subtractSingleElementTensor(tensor1, tensor2, resultTensor);
                return X::Value(cudaCodeString);
            }
            else if (!isSingleElement1 && isSingleElement2) {
                cudaCodeString = subtractSingleElementTensor(tensor2, tensor1, resultTensor);
                return X::Value(cudaCodeString);
            }
            // Both tensors are multi-element.
            else {
                if (tensor1->GetDimCount() != tensor2->GetDimCount()) {
                    return X::Value("// Error: Tensors must have the same number of dimensions for subtraction\n");
                }
                for (int i = 0; i < tensor1->GetDimCount(); i++) {
                    if (tensor1->GetDimSize(i) != tensor2->GetDimSize(i)) {
                        return X::Value("// Error: Tensor dimensions must match for subtraction\n");
                    }
                }
                long long totalElements = tensor1->GetCount();
                cudaCodeString = subtractTensors(tensor1, tensor2, resultTensor, totalElements);
                return X::Value(cudaCodeString);
            }
        }
        else if (isTensor1 && !isTensor2)
        {
            // Tensor-scalar subtraction.
            X::Tensor tensor(input1);
            X::Tensor resultTensor(output);
            if (tensor->GetCount() == 1) {
                std::string resultVar = mCodeGen.GetTensorName(resultTensor);
                cudaCodeString = "// Direct single element scalar subtraction\n";
                cudaCodeString += resultVar + " = " +
                    mCodeGen.GetTensorName(tensor) + " - " +
                    std::to_string((float)input2.ToDouble()) + ";\n";
                return X::Value(cudaCodeString);
            }
            else {
                cudaCodeString = subtractTensorScalar(tensor, (float)input2.ToDouble(), resultTensor);
                return X::Value(cudaCodeString);
            }
        }
        else if (!isTensor1 && isTensor2)
        {
            // For commutative scalar-tensor subtraction, swap the inputs.
            // Note: subtraction is not strictly commutative. Adjust this behavior as needed.
            return Minus(graph, params, kwParams, input2, input1, output);
        }
        else
        {
            // Both inputs are scalars.
            float val1 = (float)input1.ToDouble();
            float val2 = (float)input2.ToDouble();
            cudaCodeString = subtractScalars(val1, val2);
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