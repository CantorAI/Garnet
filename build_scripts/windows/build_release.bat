@echo off
setlocal

:: Initialize VS Environment for x64
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: Move to parent workspace folder (GarnetDev)
cd /d "%~dp0..\..\..\"

set BUILD_DIR=out\build\x64-Release

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if exist "%BUILD_DIR%\CMakeCache.txt" del "%BUILD_DIR%\CMakeCache.txt"

cd "%BUILD_DIR%"

:: Use VS bundled CMake if available
set CMAKE_EXE="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist %CMAKE_EXE% set CMAKE_EXE=cmake

%CMAKE_EXE% -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=bin -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=bin ..\..\..\Garnet
ninja

endlocal
