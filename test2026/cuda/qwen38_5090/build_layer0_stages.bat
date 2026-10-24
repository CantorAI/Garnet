@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%/tools/cuda13/nvidia/cu13"
set "PATH=%STUDY_ROOT%\tools\python\ninja\data\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin\x86_64;%STUDY_ROOT%\tools\TensorRT\bin;%PATH%"
"%STUDY_ROOT%\tools\python\cmake\data\bin\cmake.exe" -S "%STUDY_ROOT%/Garnet/test2026/cuda/qwen38_5090" -B "%STUDY_ROOT%/build" -DGARNET_STAGE_VENDOR_DLLS=OFF
if errorlevel 1 exit /b %errorlevel%
"%STUDY_ROOT%\tools\python\cmake\data\bin\cmake.exe" --build "%STUDY_ROOT%/build" --target garnet -j 4
exit /b %errorlevel%
