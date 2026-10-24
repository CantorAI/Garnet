@echo off
set GARNET_RUNTIME_ROOT=D:\CantorAI\5090-study-20261023\build\bin
set CUDA_VISIBLE_DEVICES=0
set GARNET_NVFP4_DOWN_CANDIDATE=0
set GARNET_PROFILE_DECODE_LAYERS=
set PATH=%GARNET_RUNTIME_ROOT%;D:\CantorAI\5090-study-20261023\tools\TensorRT\bin;D:\CantorAI\5090-study-20261023\tools\cuda13\nvidia\cu13\bin\x86_64;%PATH%
if not exist D:\CantorAI\5090-study-20261023\tmp mkdir D:\CantorAI\5090-study-20261023\tmp
set TMP=D:\CantorAI\5090-study-20261023\tmp
set TEMP=%TMP%
cd /d D:\CantorAI\5090-study-20261023\Garnet
%GARNET_RUNTIME_ROOT%\xlang3.exe D:/CantorAI/5090-study-20261023/probe-warm-mtp-trace-startup.py D:/CantorAI/5090-study-20261023 D:/CantorAI/5090-study-20261023/results/warm-mtp-trace-unprofiled-control/native
exit /b %errorlevel%
