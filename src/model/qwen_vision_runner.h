#pragma once

#include "xpackage.h"
#include "xlang.h"

namespace Garnet
{
    class QwenVisionRunner
    {
        X::Value mLayerBundles;
        X::Value mMergerBundle;

    public:
        BEGIN_PACKAGE(QwenVisionRunner)
            APISET().AddVarFunc("forward", &QwenVisionRunner::Forward);
            APISET().AddVarFunc("stats", &QwenVisionRunner::Stats);
        END_PACKAGE

        void Configure(X::Value layerBundles, X::Value mergerBundle)
        {
            mLayerBundles = layerBundles;
            mMergerBundle = mergerBundle;
        }

        void Forward(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void Stats(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
    };
}
