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

set _CL_=/MP2 %_CL_%
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\..\CantorAIWorkspace\dev\tools\Build\build_project.ps1" -Root "%REPO_ROOT%\.." -BuildType Debug -CantorOnly -WithGarnet -Target "%TARGET%"
exit /b %ERRORLEVEL%
