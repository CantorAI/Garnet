#pragma once

#include "base_filter.h"
#include "ifactory.h"
#include "GalaxyFrame.h"
#include <mutex>

namespace Galaxy {

class GarnetVLMFilter : public BaseFilter {
public:
    BEGIN_PACKAGE(GarnetVLMFilter)
        APISET().AddBase<BaseFilter>();
        APISET().AddFunc<2>("PlanSearch", &GarnetVLMFilter::PlanSearch);
        APISET().AddFunc<2>("RankSearch", &GarnetVLMFilter::RankSearch);
    END_PACKAGE

    GarnetVLMFilter();
    GarnetVLMFilter(const char* library, const char* filter, IFactory* factory);
    ~GarnetVLMFilter();
    bool onPinPutFrame(IPin* pin, X::Value& frame) override;
    X::Value PlanSearch(std::string query, X::Value imageSource);
    X::Value RankSearch(std::string query, X::Value candidates);

private:
    struct Request {
        X::Value data;
        X::Value metadata;
        long long startTime = 0;
    };

    std::mutex m_mutex;
    bool m_initialized = false;
    int m_maxOutputTokens = 384;
    std::string m_modelId = "Qwen3-VL-2B-Instruct";
    std::string m_searchModelMode;
    X::Value m_garnet;
    X::Value m_json;
    void InitializeGarnet();
    void EnsureGarnetModel();
    bool LoadModel(const X::Value& parameters, std::string& error);
    bool HasTextModelHeadroom(const X::Value& parameters, std::string& reason);
    X::Value Infer(Request& request);
    void Deliver(X::Value& result, X::Value& metadata, long long startTime);
};

}
