@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul
if errorlevel 1 exit /b 1
"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe" /nologo /Bv /std:c++17 /O2 /EHsc /fp:precise /I"D:\CantorAI\Garnet-metal\.work\prepared-half-inverse-windows-20261016\source-snapshot" "D:\CantorAI\Garnet-metal\.work\prepared-half-inverse-windows-20261016\source-snapshot\prepared_half_inverse_cpu.cpp" /Fe"D:\CantorAI\Garnet-metal\.work\prepared-half-inverse-windows-20261016\prepared_half_inverse_cpu.exe" /Fo"D:\CantorAI\Garnet-metal\.work\prepared-half-inverse-windows-20261016\prepared_half_inverse_cpu.obj"
exit /b %errorlevel%
