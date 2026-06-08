#pragma once

#include "trt_builder.h"

namespace Garnet {

    class TRTCodeGenerator {
        TRTBuilder* m_builder;
    public:
        TRTCodeGenerator(TRTBuilder* builder) : m_builder(builder) {}
        void generate() {}
    };

}
