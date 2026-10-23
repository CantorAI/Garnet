# Qwen3.5-2B xModel

Checkpoint: `Qwen/Qwen3.5-2B` (multimodal post-trained model).

This package validates the 24-layer hybrid decoder, 2048 hidden width,
16 linear key/value heads, and 24-layer vision tower before graph allocation.
It shares the Qwen3.5 tensor programs with 0.8B and 4B.

The current factories select the existing Metal backend. A model contract and
serving profile do not establish CUDA support or CUDA performance: the
TensorRT lowering and end-to-end CUDA validation are separate requirements.
Use this directory as Metal serving's `xmodelRoot`; do not substitute the
Qwen3-VL-2B package, whose attention and state layout differ.
