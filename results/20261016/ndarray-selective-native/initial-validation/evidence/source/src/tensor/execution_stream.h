// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
namespace Garnet {
namespace detail {
// Tensor-only builds need no model/provider registry. A compiled native-group
// scope installs its rank stream and clears it before returning to its caller.
inline thread_local cudaStream_t operatorExecutionStream=nullptr;
}
inline cudaStream_t CurrentExecutionStream() {
    return detail::operatorExecutionStream ? detail::operatorExecutionStream : cudaStreamPerThread;
}
}
