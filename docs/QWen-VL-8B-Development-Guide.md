# Qwen-VL Development Guide

This document formerly described the removed direct CUDA source-generation
prototype. For the current Qwen3-VL implementation, use:

- [Garnet Multi-Backend Architecture](garnet-architecture-v2.md)
- [Qwen-VL Garnet Implementation Plan](qwen-vl-implementation/README.md)

Model work belongs under `xModel/qwen3/<variant>`. Runtime-independent model
source captures an xlang TensorGraph; the selected backend lowers that graph at
compile/load time.
