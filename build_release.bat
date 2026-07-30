@echo off
setlocal

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "BUILD_DIR=%~dp0out\build\x64-Release"

"%CMAKE_EXE%" --fresh -S "%~dp0." -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl.exe ^
  -DCMAKE_CXX_COMPILER=cl.exe ^
  -DXLANG_INCLUDE_DIR=D:\CantorAI\xlang\Api
if errorlevel 1 exit /b %errorlevel%

"%CMAKE_EXE%" --build "%BUILD_DIR%" --target garnet -j 4
exit /b %errorlevel%
