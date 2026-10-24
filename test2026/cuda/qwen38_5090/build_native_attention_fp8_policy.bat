@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
set "SOURCE=%STUDY_ROOT%\results\native-attention-fp8-policy-source"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x86_64;%PATH%"
cd /d "%STUDY_ROOT%\build"
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -arch=sm_120 -Xcompiler /utf-8 -I"%SOURCE%\src\backends\tensorrt" -I"%SOURCE%\src\cuda" "%SOURCE%\test2026\cuda\qwen38_5090\native_attention_fp8_control.cu" "%SOURCE%\src\cuda\modelopt_fp8.cu" -L"%CUDA_PATH%\lib\x64" -lcublasLt -o "%STUDY_ROOT%\build\native-attention-fp8-query.exe"
if errorlevel 1 exit /b %errorlevel%
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -arch=sm_120 -Xcompiler /utf-8 -I"%SOURCE%\src\backends\tensorrt" -I"%SOURCE%\src\cuda" "%SOURCE%\test2026\cuda\qwen38_5090\actual_layer0_fp8_control.cu" "%SOURCE%\src\cuda\modelopt_fp8.cu" -L"%CUDA_PATH%\lib\x64" -lcublasLt -o "%STUDY_ROOT%\build\native-attention-fp8-gdn.exe"
if errorlevel 1 exit /b %errorlevel%
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -arch=sm_120 -Xcompiler /utf-8 -I"%SOURCE%\src\backends\tensorrt" -I"%SOURCE%\src\cuda" -I"%STUDY_ROOT%\tools\TensorRT\include" "%SOURCE%\test2026\cuda\qwen38_5090\modelopt_fp8_plugin_test.cpp" "%SOURCE%\src\backends\tensorrt\modelopt_fp8_plugin.cpp" "%SOURCE%\src\backends\tensorrt\modelopt_cublaslt.cpp" "%SOURCE%\src\cuda\modelopt_fp8.cu" -L"%CUDA_PATH%\lib\x64" -L"%STUDY_ROOT%\tools\TensorRT\lib" -lcublasLt -lnvinfer_10 -o "%STUDY_ROOT%\build\native-attention-fp8-plugin-test.exe"
exit /b %errorlevel%
