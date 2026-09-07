#include "garnet_tensor.h"

namespace Garnet {

void GarnetTensor::OnInstanceCreated(const X::Value&) {
    binary_ = RegisterTensorOperator(Host(), 2);
    unary_ = RegisterTensorOperator(Host(), 1);
    X::Module tensor(Host(), "tensor");
    fusion_ = tensor.Get("fusion");
    if (!fusion_.IsValid()) throw X::Error("XLang3 tensor.fusion is unavailable");
}

}
