// SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "graph_capture.h"

namespace Garnet {

class GarnetTensor {
public:
    void OnInstanceCreated(const X::Value&);
    X::Value BinaryOp() const { return binary_; }
    X::Value UnaryOp() const { return unary_; }
    X::Value Fusion() const { return fusion_; }

    BEGIN_PACKAGE(GarnetTensor)
        APISET().AddProp("binary_op", &GarnetTensor::BinaryOp);
        APISET().AddProp("unary_op", &GarnetTensor::UnaryOp);
        APISET().AddProp("fusion", &GarnetTensor::Fusion);
    END_PACKAGE

private:
    X::Value binary_;
    X::Value unary_;
    X::Value fusion_;
};

}
