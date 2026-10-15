// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include <stdexcept>
#if !defined(GARNET_CPU_ONLY)
#include "tensor_helper.h"
#endif
namespace Garnet {
inline X::Tensor PrepareOpenVINOHostTensor(const X::Tensor& tensor) {
#if defined(GARNET_CPU_ONLY)
    const auto info=tensor.Info();
    if(info.symbolic || info.rank==UINT32_MAX || (info.byte_size && !info.data))
        throw std::invalid_argument("OpenVINO requires materialized tensor storage");
    if(info.device_type!=0)
        throw std::invalid_argument("CPU-only Garnet cannot transfer accelerator tensors");
    auto lease=tensor.Acquire();
    return tensor;
#else
    return TensorHelper::CopyToCPU(tensor);
#endif
}
}
