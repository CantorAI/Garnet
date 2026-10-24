@echo off
setlocal
set GARNET_RUNTIME_ROOT=D:\CantorAI\5090-study-20261023\build\bin
set CUDA_VISIBLE_DEVICES=0
set GARNET_NVFP4_DOWN_CANDIDATE=0
set GARNET_PROFILE_DECODE_LAYERS=
set GARNET_FP8_GDN_M32_CANONICAL=1
set GARNET_FP8_FULL_ATTN_M32_CANONICAL=1
set PATH=%GARNET_RUNTIME_ROOT%;D:\CantorAI\5090-study-20261023\tools\TensorRT\bin;D:\CantorAI\5090-study-20261023\tools\cuda13\nvidia\cu13\bin\x86_64;%PATH%
set "TMP=%~2"
set "TEMP=%TMP%"
cd /d D:\CantorAI\5090-study-20261023\Garnet
%GARNET_RUNTIME_ROOT%\xlang3.exe D:/CantorAI/5090-study-20261023/probe-tokenizer-partial-utf8-repaired.py D:/CantorAI/5090-study-20261023 D:/CantorAI/5090-study-20261023/results/tokenizer-partial-utf8-repaired-attempt1
exit /b %errorlevel%
