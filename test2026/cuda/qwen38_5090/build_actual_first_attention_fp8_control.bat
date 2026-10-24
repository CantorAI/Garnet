@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x86_64;%PATH%"
cd /d "%STUDY_ROOT%\build"
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -arch=sm_120 -I"%STUDY_ROOT%\Garnet\src\backends\tensorrt" -I"%STUDY_ROOT%\Garnet\src\cuda" "%STUDY_ROOT%\actual-first-attention-fp8-control.cu" "%STUDY_ROOT%\Garnet\src\cuda\modelopt_fp8.cu" -L"%CUDA_PATH%\lib\x64" -lcublasLt -o "%STUDY_ROOT%\build\actual-first-attention-fp8-control.exe"
exit /b %errorlevel%
