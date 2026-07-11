#pragma once

#include "xpackage.h"
#include "xlang.h"

namespace Garnet
{
    class QwenTextRunner
    {
        X::Value mLayerBundles;

        X::Value RunLayers(
            X::Value hidden,
            X::Value cos,
            X::Value sin,
            X::Value kvHandles,
            int startPosition,
            int sequenceLength,
            bool prefill,
            X::Value deepstackFeatures = X::Value(),
            X::Value visualMask = X::Value());

    public:
        BEGIN_PACKAGE(QwenTextRunner)
            APISET().AddVarFunc("configure", &QwenTextRunner::Configure);
            APISET().AddVarFunc("prefill", &QwenTextRunner::Prefill);
            APISET().AddVarFunc("decode", &QwenTextRunner::Decode);
            APISET().AddVarFunc("stats", &QwenTextRunner::Stats);
        END_PACKAGE

        void SetLayerBundles(X::Value layerBundles) { mLayerBundles = layerBundles; }

        void Configure(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void Prefill(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void Decode(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
        void Stats(X::XRuntime* rt, X::XObj* pContext,
            X::ARGS& params, X::KWARGS& kwParams, X::Value& retValue);
    };
}
