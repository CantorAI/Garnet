@echo off
setlocal

set REPO_ROOT=%~dp0..\..
set TARGET=%~1
if "%TARGET%"=="" set TARGET=garnet

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%
set CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe

if "%TARGET%"=="garnet_tensor_tests" goto tensor_tests

set _CL_=/MP2 %_CL_%
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\..\CantorAIWorkspace\dev\tools\Build\build_project.ps1" -Root "%REPO_ROOT%\.." -BuildType Release -CantorOnly -WithGarnet -Target "%TARGET%"
exit /b %ERRORLEVEL%

:tensor_tests
set WORKSPACE_OUTPUT=%REPO_ROOT%\..\out\build\x64-Release\bin\Release
set TEST_BUILD=%REPO_ROOT%\..\out\build\x64-Release\garnet-tensor-tests
"%CMAKE_EXE%" -S "%REPO_ROOT%\test\xlang3" -B "%TEST_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DXLANG3_RUNTIME_LIBRARY="%WORKSPACE_OUTPUT%\xlang3_runtime.lib" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%WORKSPACE_OUTPUT%"
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build "%TEST_BUILD%" --target garnet_tensor_tests --config Release
exit /b %ERRORLEVEL%
