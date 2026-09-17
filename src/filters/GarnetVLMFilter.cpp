#include "GarnetVLMFilter.h"
#include "help_func.h"
#include "../entry/garnet.h"
#include "../runtime/acceleration_detector.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

namespace {
// Garnet currently owns one serving session per loaded module. Different
// Galaxy filter instances may request different models, so model selection and
// inference must remain one atomic operation across every filter instance.
std::mutex g_servingSessionMutex;

X::Value CallValue(const X::Value& callable, const X::ARGS& args = {}, const X::KWARGS& kwargs = {}) {
    X::Value result;
    if (!callable.Call(args, kwargs, result)) {
        auto* host = callable.host();
        const char* detail = host && host->runtime_last_error
            ? host->runtime_last_error(host->runtime) : nullptr;
        throw X::Error(detail && *detail ? detail : "Garnet filter call failed");
    }
    return result;
}

bool Has(const X::Value& value, const char* key) {
    if (!value.IsDict()) return false;
    X3Value method = x3_value_invalid();
    if (x3_get_attr(value.runtime(), value.raw(), "__contains__", &method) != X3_STATUS_OK)
        throw X::Error(x3_runtime_last_error(value.runtime()));
    return CallValue(X::Value(value.host(), method, false),
        {X::Value::String(value.host(), key)}).ToLongLong() != 0;
}

std::string StripJsonFence(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first != std::string::npos) text.erase(0, first);
    if (text.rfind("```", 0) == 0) {
        const auto newline = text.find('\n');
        if (newline != std::string::npos) text.erase(0, newline + 1);
        const auto fence = text.rfind("```");
        if (fence != std::string::npos) text.erase(fence);
    }
    const auto begin = text.find('{');
    const auto end = text.rfind('}');
    return begin != std::string::npos && end != std::string::npos && end >= begin
        ? text.substr(begin, end - begin + 1) : text;
}

std::string NormalizeInstructionText(std::string text) {
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    while (!text.empty() && (std::isspace(static_cast<unsigned char>(text.back())) ||
            text.back() == '.')) text.pop_back();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool LooksLikeInstruction(std::string description, const std::string& focus) {
    description = NormalizeInstructionText(std::move(description));
    const std::string normalizedFocus = NormalizeInstructionText(focus);
    if (description.empty() || (!normalizedFocus.empty() && description == normalizedFocus)) return true;
    for (const char* prefix : {"describe ", "search for ", "find ", "generate ",
            "create ", "list ", "identify ", "provide ", "a detailed analysis of "}) {
        if (description.rfind(prefix, 0) == 0) return true;
    }
    return false;
}
}

namespace Galaxy {

GarnetVLMFilter::GarnetVLMFilter() { m_filterTypeName = "GarnetVLMFilter"; }

GarnetVLMFilter::GarnetVLMFilter(const char* library, const char* filter, IFactory* factory)
    : GarnetVLMFilter() {
    m_libInfo = library;
    m_filterName = filter;
    m_pFactory = factory;
    X::Value parameters = X::Value::Dict(factory->Host());
    parameters.SetItem("modelRoot", "");
    parameters.SetItem("xmodelRoot", "");
    parameters.SetItem("cacheRoot", "");
    parameters.SetItem("accelerationRoot", "");
    parameters.SetItem("profile", "");
    parameters.SetItem("modelId", "Qwen3-VL-2B-Instruct");
    parameters.SetItem("maxOutputTokens", 384);
    m_ParamTemplate = parameters;
}

GarnetVLMFilter::~GarnetVLMFilter() = default;

bool GarnetVLMFilter::onPinPutFrame(IPin*, X::Value& frame) {
    auto* native = frame.NativeData<GalaxyFrame>();
    if (!native || native->Head()->type != Galaxy::byteStringToNumber("VLMReq", 6)) return true;
    X::Value data = frame["data"];
    if (!data.IsDict()) return true;
    // xlang3 model graphs must execute on the Galaxy callback thread. A
    // background std::thread can deadlock while re-entering the runtime. Never
    // queue stale camera frames: if another callback is already inferring,
    // discard this candidate and let the next live frame try again.
    std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return true;
    try {
        std::lock_guard<std::mutex> servingLock(g_servingSessionMutex);
        if (!m_initialized) {
            std::cerr << "Garnet-VLM: initializing model" << '\n';
            if (m_Params.IsDict() && Has(m_Params, "maxOutputTokens"))
                m_maxOutputTokens = static_cast<int>((std::max)(32LL, m_Params.Get("maxOutputTokens").ToInt64()));
            InitializeGarnet();
            m_initialized = true;
            std::cerr << "Garnet-VLM: model ready" << '\n';
        }
        EnsureGarnetModel();
        Request request{data, frame["meta_data"], frame["startTime"].ToInt64()};
        std::cerr << "Garnet-VLM: inference start image=" << data.Get("image_id").ToInt64() << '\n';
        X::Value result = Infer(request);
        std::cerr << "Garnet-VLM: inference complete image=" << data.Get("image_id").ToInt64() << '\n';
        Deliver(result, request.metadata, request.startTime);
    } catch (const std::exception& error) {
        std::cerr << "Garnet-VLM inference: " << error.what() << '\n';
    }
    return true;
}

void GarnetVLMFilter::InitializeGarnet() {
    X::Module json(Host(), "json", nullptr);
    m_json = json;
    X::Value cantor = m_pFactory->GetCantor();
    m_garnet = cantor["QueryModule"]("garnet");
    if (!m_garnet.IsObject()) throw X::Error("Garnet module is not loaded");
    if (m_Params.IsDict() && Has(m_Params, "accelerationRoot")) {
        const std::string accelerationRoot = m_Params.Get("accelerationRoot").ToString();
        if (!accelerationRoot.empty()) {
            X::Value activation = m_json["loads"](
                Garnet::AccelerationDetector::ActivateJson(accelerationRoot));
            if (!activation.IsDict() || !Has(activation, "status") ||
                activation.Get("status").ToString() != "ready") {
                const std::string detail = activation.IsDict() && Has(activation, "message")
                    ? activation.Get("message").ToString()
                    : "Garnet acceleration activation failed";
                throw X::Error(detail);
            }
        }
    }
    if (!m_Params.IsDict() || !Has(m_Params, "modelRoot")) return;
    const std::string modelRoot = m_Params.Get("modelRoot").ToString();
    if (modelRoot.empty()) return;
    const std::string xmodelRoot = Has(m_Params, "xmodelRoot") ? m_Params.Get("xmodelRoot").ToString() : std::string();
    const std::string cacheRoot = Has(m_Params, "cacheRoot") ? m_Params.Get("cacheRoot").ToString() : std::string();
    const std::string profile = Has(m_Params, "profile") ? m_Params.Get("profile").ToString() : std::string();
    const std::string modelId = Has(m_Params, "modelId") ? m_Params.Get("modelId").ToString() : "Qwen3-VL-2B-Instruct";
    m_modelId = modelId;
    X::Value statusText = CallValue(m_garnet["serve_model"], {
        X::Value::String(Host(), modelRoot),
        X::Value::String(Host(), xmodelRoot),
        X::Value::String(Host(), cacheRoot),
        X::Value::String(Host(), profile),
        X::Value::String(Host(), modelId)});
    X::Value status = m_json["loads"](statusText.ToString());
    const bool ready = status.IsDict() && (
        (Has(status, "ready") && status.Get("ready").ToLongLong() != 0) ||
        (Has(status, "state") && status.Get("state").ToString() == "ready"));
    if (!ready) {
        const std::string detail = status.IsDict() && Has(status, "error_message")
            ? status.Get("error_message").ToString()
            : (status.IsDict() && Has(status, "error")
                ? status.Get("error").ToString() : "Garnet model did not become ready");
        throw X::Error(detail);
    }
}

void GarnetVLMFilter::EnsureGarnetModel() {
    if (!m_initialized || !m_garnet.IsObject() || m_modelId.empty()) return;
    X::Value statusText = CallValue(m_garnet["serve_status_json"]);
    X::Value status = m_json["loads"](statusText.ToString());
    const bool sameModel = status.IsDict() && Has(status, "model_id") &&
        status.Get("model_id").ToString() == m_modelId &&
        Has(status, "ready") && status.Get("ready").ToLongLong() != 0;
    if (!sameModel) InitializeGarnet();
}

X::Value GarnetVLMFilter::Infer(Request& request) {
    X::Value candidate = request.data;
    X::Value jpeg = Has(candidate, "jpeg") ? candidate.Get("jpeg") : X::Value();
    const std::string eventPrompt = Has(candidate, "vlm_prompt") ? candidate.Get("vlm_prompt").ToString() : std::string();
    const std::string searchFocus = Has(candidate, "search_focus") ? candidate.Get("search_focus").ToString() : std::string();
    const bool eventEnabled = Has(candidate, "event_enabled") && candidate.Get("event_enabled").ToLongLong();
    const bool searchEnabled = Has(candidate, "search_enabled") && candidate.Get("search_enabled").ToLongLong();

    std::ostringstream prompt;
    prompt << "Analyze the camera image for an automated monitoring skill. Return ONLY one JSON object. "
           << "Schema: {\"event\":{\"matched\":boolean,\"title\":string,\"description\":string,\"confidence\":number},"
           << "\"search\":{\"description\":string,\"entities\":[string],\"actions\":[string],\"scene\":string,\"tags\":[string]}}. ";
    if (eventEnabled) prompt << "Event instruction: " << eventPrompt << ". ";
    else prompt << "Set event.matched to false. ";
    if (searchEnabled) {
        prompt << "Generate factual searchable metadata using only visibly supported facts. "
               << "search.description must be a present-tense description of the image, never an instruction. "
               << "Do not copy or paraphrase the search focus and do not assume a focused item is present. "
               << "Observation priority: " << searchFocus << ". ";
    }
    else prompt << "Return empty search fields. ";

    X::Value responseText = CallValue(m_garnet["infer_json"],
        {X::Value::String(Host(), prompt.str()), jpeg, X::Value(m_maxOutputTokens),
         X::Value::String(Host(), m_modelId)});
    X::Value outer = m_json["loads"](responseText.ToString());
    X::Value result = X::Value::Dict(Host());
    result.SetItem("kind", "vlm_result");
    result.SetItem("image_id", candidate.Get("image_id"));
    result.SetItem("skill_ref", candidate.Get("skill_ref"));
    result.SetItem("skill_id", candidate.Get("skill_id"));
    result.SetItem("skill_name", candidate.Get("skill_name"));
    result.SetItem("jpeg", jpeg);
    result.SetItem("detections", candidate.Get("detections"));
    result.SetItem("event_enabled", eventEnabled);
    result.SetItem("direct_event", Has(candidate, "direct_event") ? candidate.Get("direct_event") : X::Value(false));
    result.SetItem("verify_event", Has(candidate, "verify_event") ? candidate.Get("verify_event") : X::Value(false));
    result.SetItem("search_enabled", searchEnabled);
    if (Has(candidate, "event_id")) result.SetItem("event_id", candidate.Get("event_id"));
    result.SetItem("model_response", outer);
    result.SetItem("model_response_json", responseText);
    if (outer.IsDict() && Has(outer, "status") && outer.Get("status").ToString() == "ok" && Has(outer, "text")) {
        try {
            X::Value structured = m_json["loads"](StripJsonFence(outer.Get("text").ToString()));
            if (structured.IsDict()) {
                X::Value event = Has(structured, "event") ? structured.Get("event") : X::Value();
                X::Value search = Has(structured, "search") ? structured.Get("search") : X::Value();
                if (searchEnabled && search.IsDict() && Has(search, "description") &&
                    LooksLikeInstruction(search.Get("description").ToString(), searchFocus)) {
                    std::string fallback;
                    if (event.IsDict() && Has(event, "description"))
                        fallback = event.Get("description").ToString();
                    if (LooksLikeInstruction(fallback, searchFocus) && Has(search, "scene"))
                        fallback = search.Get("scene").ToString();
                    search.SetItem("description", X::Value::String(Host(), fallback));
                }
                result.SetItem("status", "ok");
                result.SetItem("event", event);
                result.SetItem("search", search);
                return result;
            }
        } catch (...) {}
    }
    result.SetItem("status", "error");
    result.SetItem("error", outer.IsDict() && Has(outer, "error_message")
        ? outer.Get("error_message") : X::Value("Invalid structured VLM response"));
    return result;
}

X::Value GarnetVLMFilter::PlanSearch(std::string query, X::Value imageSource) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::lock_guard<std::mutex> servingLock(g_servingSessionMutex);
    if (!m_initialized) {
        if (m_Params.IsDict() && Has(m_Params, "maxOutputTokens"))
            m_maxOutputTokens = static_cast<int>((std::max)(32LL, m_Params.Get("maxOutputTokens").ToInt64()));
        InitializeGarnet();
        m_initialized = true;
    }
    EnsureGarnetModel();
    std::ostringstream prompt;
    prompt << "Translate this camera-history search into concise English camera-index keywords. Return JSON only. Schema: "
           << "{\"normalized_query\":string,"
           << "\"device_id\":string|null,\"channel_id\":string|null,"
           << "\"skill_ref\":string|null,\"from_ms\":integer|null,\"to_ms\":integer|null}. "
           << "normalized_query must be the shortest literal English translation containing only content explicitly "
           << "written by the user. Never expand it with synonyms or implied objects, locations, time, lighting, or "
           << "visible details. Examples: '女人在做饭' becomes 'woman cooking'; '女人在编篮子' becomes "
           << "'woman weaving basket'; 'empty parking lot at night' stays 'empty parking lot night'. Do not include generic "
           << "words such as image, video, scene, or camera unless the user explicitly searches for them. "
           << "Ignore the supplied reference image; use only the text request. "
           << "Never output SQL. Request: " << query;
    X::Value responseText = CallValue(m_garnet["infer_json"], {
        X::Value::String(Host(), prompt.str()), imageSource,
        X::Value((std::min)(m_maxOutputTokens, 160)),
        X::Value::String(Host(), m_modelId)});
    X::Value outer = m_json["loads"](responseText.ToString());
    if (!outer.IsDict() || !Has(outer, "status") || outer.Get("status").ToString() != "ok")
        throw X::Error(outer.IsDict() && Has(outer, "error_message")
            ? outer.Get("error_message").ToString() : "Garnet search planning failed");
    return m_json["loads"](StripJsonFence(outer.Get("text").ToString()));
}

void GarnetVLMFilter::Deliver(X::Value& result, X::Value& metadata, long long startTime) {
    X::Value frame = m_pFactory->NewDataFrame();
    X::KWARGS kwargs;
    kwargs.emplace_back("data", result);
    kwargs.emplace_back("meta_data", metadata);
    CallValue(frame["Set"], {}, kwargs);
    auto* native = frame.NativeData<GalaxyFrame>();
    if (!native) return;
    native->Head()->type = Galaxy::byteStringToNumber("VLMRsl", 6);
    native->Head()->startTime = startTime;
    for (const auto& output : m_outputList) {
        auto* pin = output.NativeData<Pin>();
        if (!pin) throw X::Error("expected a Galaxy pin");
        pin->Put(frame);
    }
}

}
