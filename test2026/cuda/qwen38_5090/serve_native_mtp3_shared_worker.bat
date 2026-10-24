@echo off
set GARNET_RUNTIME_ROOT=D:\CantorAI\5090-study-20261023\build\bin
set CUDA_VISIBLE_DEVICES=0
set GARNET_STUDY_JOINED_MLP=0
set GARNET_STUDY_PACKED=1
set GARNET_STUDY_REUSE_VERIFIED_HEAD=0
set GARNET_STUDY_SHARED_COMPRESSED=1
set GARNET_STUDY_MTP_DEPTH=3
set GARNET_STUDY_MTP_CHUNK=4
set GARNET_STUDY_RESIDENT_VISION=0
set PATH=%GARNET_RUNTIME_ROOT%;D:\CantorAI\5090-study-20261023\tools\TensorRT\bin;D:\CantorAI\5090-study-20261023\tools\cuda13\nvidia\cu13\bin\x86_64;%PATH%
if not exist D:\CantorAI\5090-study-20261023\tmp mkdir D:\CantorAI\5090-study-20261023\tmp
set TMP=D:\CantorAI\5090-study-20261023\tmp
set TEMP=%TMP%
cd /d D:\CantorAI\5090-study-20261023\Garnet
%GARNET_RUNTIME_ROOT%\xlang3.exe test2026/cuda/qwen38_5090/native_stream_worker.py D:/CantorAI/5090-study-20261023/models/Qwen3.8-27B-NVFP4 D:/CantorAI/5090-study-20261023/tmp/native-mtp3-shared-image.jpg
exit /b %errorlevel%
