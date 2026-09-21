REM SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
REM SPDX-License-Identifier: Apache-2.0

@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
set "BUILD_DIR=%REPO_ROOT%\out\build\x64-Release"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=garnet"
if "%XLANG3_ROOT%"=="" set "XLANG3_ROOT=%REPO_ROOT%\..\xlang3"
if "%GARNET_TENSORRT_ROOT%"=="" set "GARNET_TENSORRT_ROOT=%REPO_ROOT%\..\ThirdPartySDK\TensorRT"

where cmake >nul 2>nul || (echo cmake was not found on PATH & exit /b 1)
where ninja >nul 2>nul || (echo ninja was not found on PATH & exit /b 1)

cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="%BUILD_DIR%\bin" ^
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="%BUILD_DIR%\bin" ^
  -DXLANG3_ROOT="%XLANG3_ROOT%" ^
  -DGARNET_TENSORRT_ROOT="%GARNET_TENSORRT_ROOT%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --target "%TARGET%"
exit /b %ERRORLEVEL%
