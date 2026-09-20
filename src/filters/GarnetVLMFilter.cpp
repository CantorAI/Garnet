#include "GarnetVLMFilter.h"
#include "help_func.h"
#include "../entry/garnet.h"
#include "../runtime/acceleration_detector.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

namespace {
// Each filter keeps its own request serialization while Garnet routes calls to
// independently resident model instances by model UID.
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
    const auto objectBegin = text.find('{');
    const auto arrayBegin = text.find('[');
    const bool useArray = arrayBegin != std::string::npos &&
        (objectBegin == std::string::npos || arrayBegin < objectBegin);
    const auto begin = useArray ? arrayBegin : objectBegin;
    const auto end = useArray ? text.rfind(']') : text.rfind('}');
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
    for (const char* prefix : {"观察画面", "请观察", "描述画面", "分析画面"}) {
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
    parameters.SetItem("searchModelMode", "");
    parameters.SetItem("minTotalVramMB", 20480);
    parameters.SetItem("minFreeVramMB", 6144);
    X::Value textModel = X::Value::Dict(factory->Host());
    textModel.SetItem("modelRoot", "");
    textModel.SetItem("xmodelRoot", "");
    textModel.SetItem("cacheRoot", "");
    textModel.SetItem("profile", "");
    textModel.SetItem("modelId", "Qwen3-1.7B");
    parameters.SetItem("textModel", textModel);
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
        if (!result.IsDict() || !Has(result, "status") || result.Get("status").ToString() != "ok") {
            std::string diagnostic = result.IsDict() && Has(result, "model_response_json")
                ? result.Get("model_response_json").ToString() : std::string();
            if (diagnostic.size() > 2048) diagnostic.resize(2048);
            std::cerr << "Garnet-VLM: invalid structured result image="
                      << data.Get("image_id").ToInt64() << " error="
                      << (result.IsDict() && Has(result, "error")
                          ? result.Get("error").ToString() : "unknown")
                      << " response=" << diagnostic << '\n';
        }
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
    m_searchModelMode = Has(m_Params, "searchModelMode")
        ? m_Params.Get("searchModelMode").ToString() : std::string();
    if (m_searchModelMode.empty()) {
        std::string error;
        if (!LoadModel(m_Params, error)) throw X::Error(error);
        return;
    }
    if (m_searchModelMode != "auto" && m_searchModelMode != "shared_vlm" &&
        m_searchModelMode != "separate_text") {
        throw X::Error("searchModelMode must be auto, shared_vlm, or separate_text");
    }

    X::Value textModel = Has(m_Params, "textModel") ? m_Params.Get("textModel") : X::Value();
    if (m_searchModelMode == "separate_text") {
        std::string error;
        if (!textModel.IsDict() || !LoadModel(textModel, error))
            throw X::Error(error.empty() ? "The separate search text model is not configured" : error);
        std::cerr << "Garnet-VLM: search selected separate text model " << m_modelId << '\n';
        return;
    }

    std::string sharedError;
    if (!LoadModel(m_Params, sharedError)) throw X::Error(sharedError);
    const std::string sharedModelId = m_modelId;
    if (m_searchModelMode == "shared_vlm") {
        std::cerr << "Garnet-VLM: search selected shared VLM " << m_modelId << '\n';
        return;
    }

    std::string reason;
    if (!textModel.IsDict() || !HasTextModelHeadroom(textModel, reason)) {
        m_modelId = sharedModelId;
        std::cerr << "Garnet-VLM: search using shared VLM: "
                  << (reason.empty() ? "text model is not configured" : reason) << '\n';
        return;
    }
    std::string textError;
    if (LoadModel(textModel, textError)) {
        std::cerr << "Garnet-VLM: search selected separate text model " << m_modelId << '\n';
        return;
    }
    m_modelId = sharedModelId;
    std::cerr << "Garnet-VLM: separate text model unavailable; using shared VLM: "
              << textError << '\n';
}

bool GarnetVLMFilter::LoadModel(const X::Value& parameters, std::string& error) {
    if (!parameters.IsDict() || !Has(parameters, "modelRoot")) {
        error = "Garnet model parameters are missing";
        return false;
    }
    const std::string modelRoot = parameters.Get("modelRoot").ToString();
    if (modelRoot.empty()) {
        error = "Garnet modelRoot is empty";
        return false;
    }
    const std::string xmodelRoot = Has(parameters, "xmodelRoot") ? parameters.Get("xmodelRoot").ToString() : std::string();
    const std::string cacheRoot = Has(parameters, "cacheRoot") ? parameters.Get("cacheRoot").ToString() : std::string();
    const std::string profile = Has(parameters, "profile") ? parameters.Get("profile").ToString() : std::string();
    const std::string modelId = Has(parameters, "modelId") ? parameters.Get("modelId").ToString() : "Qwen3-VL-2B-Instruct";
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
        error = status.IsDict() && Has(status, "error_message")
            ? status.Get("error_message").ToString()
            : (status.IsDict() && Has(status, "message")
                ? status.Get("message").ToString()
            : (status.IsDict() && Has(status, "error")
                ? status.Get("error").ToString() : "Garnet model did not become ready"));
        return false;
    }
    m_modelId = modelId;
    error.clear();
    return true;
}

bool GarnetVLMFilter::HasTextModelHeadroom(const X::Value& parameters, std::string& reason) {
    int deviceId = 0;
    if (parameters.IsDict() && Has(parameters, "profile")) {
        const std::string profileText = parameters.Get("profile").ToString();
        if (!profileText.empty()) {
            try {
                X::Value profile = m_json["loads"](profileText);
                if (profile.IsDict() && Has(profile, "deviceId"))
                    deviceId = static_cast<int>(profile.Get("deviceId").ToInt64());
            } catch (...) {}
        }
    }
    int previousDevice = 0;
    if (cudaGetDevice(&previousDevice) != cudaSuccess || cudaSetDevice(deviceId) != cudaSuccess) {
        reason = "CUDA device " + std::to_string(deviceId) + " is unavailable";
        return false;
    }
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    const cudaError_t status = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (previousDevice != deviceId) cudaSetDevice(previousDevice);
    if (status != cudaSuccess) {
        reason = "CUDA memory information is unavailable";
        return false;
    }
    const long long freeMb = static_cast<long long>(freeBytes >> 20);
    const long long totalMb = static_cast<long long>(totalBytes >> 20);
    const long long minTotalMb = Has(m_Params, "minTotalVramMB")
        ? (std::max)(0LL, m_Params.Get("minTotalVramMB").ToInt64()) : 20480LL;
    const long long minFreeMb = Has(m_Params, "minFreeVramMB")
        ? (std::max)(0LL, m_Params.Get("minFreeVramMB").ToInt64()) : 6144LL;
    if (totalMb < minTotalMb || freeMb < minFreeMb) {
        reason = "GPU memory below text-model threshold (total=" +
            std::to_string(totalMb) + " MB, free=" + std::to_string(freeMb) + " MB)";
        return false;
    }
    return true;
}

void GarnetVLMFilter::EnsureGarnetModel() {
    if (!m_initialized || !m_garnet.IsObject() || m_modelId.empty()) return;
    X::Value statusText = CallValue(m_garnet["serve_status_json"], {
        X::Value::String(Host(), m_modelId)});
    X::Value status = m_json["loads"](statusText.ToString());
    const bool sameModel = status.IsDict() && Has(status, "model_id") &&
        status.Get("model_id").ToString() == m_modelId &&
        Has(status, "ready") && status.Get("ready").ToLongLong() != 0;
    if (!sameModel) InitializeGarnet();
}

std::string GarnetVLMFilter::WorldScope(const X::Value& metadata) const {
    if (!metadata.IsDict()) return "global";
    std::string camera;
    if (Has(metadata, "source_id")) camera = metadata.Get("source_id").ToString();
    if (camera.empty() && Has(metadata, "deviceId")) {
        camera = metadata.Get("deviceId").ToString();
        if (Has(metadata, "channelId") && !metadata.Get("channelId").ToString().empty())
            camera += "/channel:" + metadata.Get("channelId").ToString();
    }
    if (camera.empty()) return "global";
    const long long preset = Has(metadata, "preset_id") ? metadata.Get("preset_id").ToInt64() : 0;
    return "camera:" + camera + "/preset:" + std::to_string(preset);
}

X::Value GarnetVLMFilter::LoadWorldContext(const std::string& scope) {
    X::Value context = X::Value::Dict(Host());
    context.SetItem("scope", X::Value::String(Host(), scope));
    context.SetItem("current", X::Value::List(Host()));
    context.SetItem("recent_history", X::Value::List(Host()));
    try {
        X::Value cantor = m_pFactory->GetCantor();
        X::Value world = cantor["WorldState"]();
        X::Value current = world["Query"](scope);
        if (!current.IsList()) return context;
        X::Value compactCurrent = X::Value::List(Host());
        X::Value histories = X::Value::List(Host());
        const uint64_t count = (std::min)(current.Size(), static_cast<uint64_t>(16));
        for (uint64_t i = 0; i < count; ++i) {
            X::Value state = current[static_cast<long long>(i)];
            if (!state.IsDict() || !Has(state, "scope") || !Has(state, "key")) continue;
            X::Value currentItem = X::Value::Dict(Host());
            currentItem.SetItem("key", state.Get("key"));
            if (Has(state, "properties")) currentItem.SetItem("properties", state.Get("properties"));
            if (Has(state, "revision")) currentItem.SetItem("revision", state.Get("revision"));
            if (Has(state, "observed_at")) currentItem.SetItem("observed_at", state.Get("observed_at"));
            compactCurrent.Append(currentItem);
            X::Value item = X::Value::Dict(Host());
            item.SetItem("key", state.Get("key"));
            X::Value revisions = world["History"](
                state.Get("scope"), state.Get("key"), X::Value(2));
            X::Value compactRevisions = X::Value::List(Host());
            if (revisions.IsList()) {
                for (uint64_t j = 0; j < revisions.Size(); ++j) {
                    X::Value revision = revisions[static_cast<long long>(j)];
                    if (!revision.IsDict()) continue;
                    X::Value revisionItem = X::Value::Dict(Host());
                    if (Has(revision, "revision")) revisionItem.SetItem("revision", revision.Get("revision"));
                    if (Has(revision, "observed_at")) revisionItem.SetItem("observed_at", revision.Get("observed_at"));
                    if (Has(revision, "properties")) revisionItem.SetItem("properties", revision.Get("properties"));
                    compactRevisions.Append(revisionItem);
                }
            }
            item.SetItem("revisions", compactRevisions);
            histories.Append(item);
        }
        context.SetItem("current", compactCurrent);
        context.SetItem("recent_history", histories);
        context.SetItem("available", X::Value(true));
    } catch (const std::exception& error) {
        context.SetItem("available", X::Value(false));
        context.SetItem("error", X::Value::String(Host(), error.what()));
    }
    return context;
}

bool GarnetVLMFilter::CommitWorldUpdates(X::Value& structured, Request& request,
    const std::string& scope, X::Value& changes) {
    changes = X::Value::List(Host());
    if (!structured.IsDict()) return false;
    X::Value updates = Has(structured, "state_updates")
        ? structured.Get("state_updates") : X::Value::List(Host());
    bool changed = false;
    try {
        X::Value cantor = m_pFactory->GetCantor();
        X::Value world = cantor["WorldState"]();
        for (uint64_t i = 0; updates.IsList() && i < updates.Size(); ++i) {
            X::Value state = updates[static_cast<long long>(i)];
            if (!state.IsDict() || !Has(state, "key") || !state.Get("key").IsString() ||
                state.Get("key").ToString().empty() || !Has(state, "properties") ||
                !state.Get("properties").IsDict()) continue;
            const std::string stateKey = state.Get("key").ToString();
            if (stateKey == "event" || stateKey == "search" || stateKey == "camera" ||
                stateKey == "world_state" || stateKey == "scene_inventory") continue;
            X::Value update = X::Value::Dict(Host());
            update.SetItem("scope", X::Value::String(Host(), scope));
            update.SetItem("key", state.Get("key"));
            update.SetItem("properties", state.Get("properties"));
            update.SetItem("observed_at", X::Value(request.startTime));
            if (Has(request.data, "image_id")) update.SetItem("update_uid", request.data.Get("image_id"));
            if (Has(request.data, "skill_id")) update.SetItem("skill_id", request.data.Get("skill_id"));
            if (Has(state, "confidence")) update.SetItem("confidence", state.Get("confidence"));
            if (Has(state, "evidence")) update.SetItem("evidence", state.Get("evidence"));
            if (request.metadata.IsDict()) {
                if (Has(request.metadata, "source_id")) update.SetItem("source", request.metadata.Get("source_id"));
                else if (Has(request.metadata, "deviceId")) update.SetItem("source", request.metadata.Get("deviceId"));
            }
            X::Value committed = world["Update"](update);
            if (committed.IsDict()) {
                changes.Append(committed);
                if (Has(committed, "changed") && committed.Get("changed").ToLongLong()) changed = true;
            }
        }

        X::Value search = Has(structured, "search") ? structured.Get("search") : X::Value();
        X::Value objectCounts = search.IsDict() && Has(search, "object_counts")
            ? search.Get("object_counts") : X::Value();
        X::Value counts = X::Value::Dict(Host());
        bool hasInventory = false;
        if (objectCounts.IsList()) {
            for (uint64_t i = 0; i < objectCounts.Size(); ++i) {
                X::Value item = objectCounts[static_cast<long long>(i)];
                if (!item.IsDict() || !Has(item, "object") || !Has(item, "count") ||
                    item.Get("count").IsNone()) continue;
                const std::string object = item.Get("object").ToString();
                if (object.empty()) continue;
                counts.SetItem(object.c_str(), item.Get("count"));
                hasInventory = true;
            }
        }
        if (search.IsDict() && Has(search, "people_count") &&
            !search.Get("people_count").IsNone()) {
            counts.SetItem("person", search.Get("people_count"));
            hasInventory = true;
        }
        if (hasInventory) {
            X::Value properties = X::Value::Dict(Host());
            properties.SetItem("counts", counts);
            X::Value inventory = X::Value::Dict(Host());
            inventory.SetItem("scope", X::Value::String(Host(), scope));
            inventory.SetItem("key", X::Value::String(Host(), "scene_inventory"));
            inventory.SetItem("properties", properties);
            inventory.SetItem("observed_at", X::Value(request.startTime));
            if (Has(request.data, "image_id")) inventory.SetItem("update_uid", request.data.Get("image_id"));
            if (Has(request.data, "skill_id")) inventory.SetItem("skill_id", request.data.Get("skill_id"));
            if (request.metadata.IsDict()) {
                if (Has(request.metadata, "source_id")) inventory.SetItem("source", request.metadata.Get("source_id"));
                else if (Has(request.metadata, "deviceId")) inventory.SetItem("source", request.metadata.Get("deviceId"));
            }
            X::Value committed = world["Update"](inventory);
            if (committed.IsDict()) {
                changes.Append(committed);
                if (Has(committed, "changed") && committed.Get("changed").ToLongLong()) changed = true;
            }
        }
    } catch (const std::exception& error) {
        X::Value failure = X::Value::Dict(Host());
        failure.SetItem("error", X::Value::String(Host(), error.what()));
        changes.Append(failure);
    }
    return changed;
}

X::Value GarnetVLMFilter::Infer(Request& request) {
    X::Value candidate = request.data;
    X::Value jpeg = Has(candidate, "jpeg") ? candidate.Get("jpeg") : X::Value();
    const std::string eventPrompt = Has(candidate, "vlm_prompt") ? candidate.Get("vlm_prompt").ToString() : std::string();
    const std::string searchFocus = Has(candidate, "search_focus") ? candidate.Get("search_focus").ToString() : std::string();
    const std::string outputLanguage = Has(candidate, "output_language") && !candidate.Get("output_language").ToString().empty()
        ? candidate.Get("output_language").ToString() : "en";
    const bool eventEnabled = Has(candidate, "event_enabled") && candidate.Get("event_enabled").ToLongLong();
    const bool searchEnabled = Has(candidate, "search_enabled") && candidate.Get("search_enabled").ToLongLong();
    const std::string worldScope = WorldScope(request.metadata);
    X::Value worldContext = LoadWorldContext(worldScope);
    const std::string worldJson = m_json["dumps"](worldContext).ToString();

    std::ostringstream prompt;
    prompt << "Analyze the camera image for an automated monitoring skill. Return ONLY one JSON object. "
           << "Schema: {\"event\":{\"matched\":boolean,\"title\":string,\"description\":string,\"confidence\":number},"
           << "\"search\":{\"description\":string,\"entities\":[string],\"actions\":[string],\"scene\":string,"
           << "\"tags\":[string],\"object_counts\":[{\"object\":string,\"count\":integer}],\"people_count\":integer|null},"
           << "\"state_updates\":[{\"key\":string,\"properties\":object,\"confidence\":number,\"evidence\":string}]}. ";
    prompt << "Emit compact/minified JSON. Keep each title/description/evidence brief and each array at most 5 items. "
           << "state_updates contain world facts only, not copies of event or search fields. ";
    prompt << "Keep JSON property names exactly as specified. Write every natural-language string value in "
           << (outputLanguage == "zh-CN" ? "Simplified Chinese" : outputLanguage == "zh-TW" ? "Traditional Chinese" : "English")
           << ". ";
    prompt << "WorldState JSON for this camera/preset: " << worldJson << ". "
           << "Use it to detect changes. Return stable snake_case state_updates needed by the event even when unmatched. "
           << "One uncertain/occluded frame is not absence. event.matched may be true only for a NEW matching transition, never an unchanged condition. "
           << "State example: {\"key\":\"cellphone\",\"properties\":{\"present\":true,\"count\":1,\"location\":\"desk\"}}. "
           << "Never use event or search as a state key. ";
    if (eventEnabled) prompt << "Event instruction: " << eventPrompt << ". ";
    else prompt << "Set event.matched to false and return an empty state_updates list unless state is required for an enabled event. ";
    if (searchEnabled) {
        prompt << "Generate factual searchable metadata using only visibly supported facts. "
               << "Set search.object_counts to counts of reliably visible distinct objects. Every object value is a machine identifier and MUST remain a canonical "
               << "lowercase English singular label such as person, bowl, dog, and car; object values are the only strings exempt from the output-language rule. "
               << "Do not guess uncertain counts. Also set search.people_count to the count for object=person when present, otherwise null, for backward compatibility. "
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
    result.SetItem("output_language", X::Value::String(Host(), outputLanguage));
    if (Has(candidate, "event_id")) result.SetItem("event_id", candidate.Get("event_id"));
    result.SetItem("model_response", outer);
    result.SetItem("model_response_json", responseText);
    if (outer.IsDict() && Has(outer, "status") && outer.Get("status").ToString() == "ok" && Has(outer, "text")) {
        try {
            X::Value structured = m_json["loads"](StripJsonFence(outer.Get("text").ToString()));
            if (structured.IsDict()) {
                X::Value event = Has(structured, "event") ? structured.Get("event") : X::Value();
                X::Value search = Has(structured, "search") ? structured.Get("search") : X::Value();
                X::Value stateChanges;
                const bool worldChanged = CommitWorldUpdates(structured, request, worldScope, stateChanges);
                X::Value stateUpdates = Has(structured, "state_updates") ? structured.Get("state_updates") : X::Value();
                if (event.IsDict() && stateChanges.IsList() && stateChanges.Size() > 0 &&
                    Has(event, "matched") && event.Get("matched").ToLongLong() && !worldChanged)
                    event.SetItem("matched", X::Value(false));
                if (search.IsDict()) search.SetItem("language", X::Value::String(Host(), outputLanguage));
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
                result.SetItem("state_updates", stateUpdates);
                result.SetItem("state_changes", stateChanges);
                result.SetItem("world_state_scope", X::Value::String(Host(), worldScope));
                return result;
            }
        } catch (...) {}
    }
    result.SetItem("status", "error");
    result.SetItem("error", outer.IsDict() && Has(outer, "error_message") &&
        !outer.Get("error_message").ToString().empty()
        ? outer.Get("error_message")
        : X::Value::String(Host(), "Invalid structured VLM response"));
    return result;
}

X::Value GarnetVLMFilter::PlanSearch(std::string query, X::Value imageSource) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_initialized) {
        if (m_Params.IsDict() && Has(m_Params, "maxOutputTokens"))
            m_maxOutputTokens = static_cast<int>((std::max)(32LL, m_Params.Get("maxOutputTokens").ToInt64()));
        InitializeGarnet();
        m_initialized = true;
    }
    EnsureGarnetModel();
    std::ostringstream prompt;
    prompt << "Plan a multilingual camera-history search. Return JSON only. Schema: "
           << "{\"normalized_query\":string,\"query_language\":string,\"source_terms\":[string],"
           << "\"quantity_constraints\":[{\"object\":string,\"operator\":\"eq\"|\"gte\"|\"lte\",\"count\":integer}],\"person_count\":integer|null,"
           << "\"device_id\":string|null,\"channel_id\":string|null,"
           << "\"skill_ref\":string|null,\"from_ms\":integer|null,\"to_ms\":integer|null}. "
           << "normalized_query must be the shortest literal English translation containing only content explicitly "
           << "written by the user. query_language must be a BCP-47 language code. source_terms must contain the shortest "
           << "literal concepts in the user's original language. Convert every explicit object quantity into quantity_constraints; "
           << "object must be a canonical lowercase English singular label. A bare number means eq; preserve at least/at most as gte/lte. "
           << "person_count must mirror an explicit exact person constraint for backward compatibility. "
           << "Never expand it with synonyms or implied objects, locations, time, lighting, or "
           << "visible details. Examples: '女人在做饭' becomes 'woman cooking'; '女人在编篮子' becomes "
           << "'woman weaving basket'; '有人扫地' becomes normalized_query 'person sweeping' with source_terms ['人','扫地']; "
           << "'empty parking lot at night' stays 'empty parking lot night'. Do not include generic "
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

X::Value GarnetVLMFilter::RankSearch(std::string query, X::Value candidates) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_initialized) {
        if (m_Params.IsDict() && Has(m_Params, "maxOutputTokens"))
            m_maxOutputTokens = static_cast<int>((std::max)(32LL, m_Params.Get("maxOutputTokens").ToInt64()));
        InitializeGarnet();
        m_initialized = true;
    }
    EnsureGarnetModel();
    std::string candidateJson = m_json["dumps"](candidates).ToString();
    std::ostringstream prompt;
    prompt << "Rank camera-history candidates for the user's request. Use only the supplied metadata. "
           << "Ignore the supplied reference image; it is only present to satisfy the vision-model input contract. "
           << "Return JSON only with schema {\"rankings\":[{\"id\":string,\"score\":integer,\"exact_match\":boolean,\"reason\":string}]}. "
           << "Include every candidate exactly once, best first. score is 0-100. reason must be at most 8 words or 12 Chinese characters. "
           << "Emit the compact JSON object and stop immediately; do not use Markdown or add commentary. "
           << "exact_match is true only when the supplied metadata explicitly satisfies every requested subject, object, action, relation, attribute, and quantity; "
           << "it is false when any required detail is absent, contradictory, or uncertain. "
           << "Explicit quantity constraints are hard requirements: a candidate with conflicting object_counts must score below every matching candidate. "
           << "Write reason in the same language as the user's request. Do not invent visible details. Request: " << query << " Candidates: " << candidateJson;
    X::Value imageSource;
    if (candidates.IsList() && candidates.Size() > 0) {
        X::Value first = candidates.Get(static_cast<uint64_t>(0));
        if (first.IsDict() && Has(first, "image_path")) imageSource = first.Get("image_path");
    }
    X::Value responseText = CallValue(m_garnet["infer_json"], {
        X::Value::String(Host(), prompt.str()), imageSource,
        X::Value((std::min)(m_maxOutputTokens, 256)),
        X::Value::String(Host(), m_modelId)});
    X::Value outer = m_json["loads"](responseText.ToString());
    if (!outer.IsDict() || !Has(outer, "status") || outer.Get("status").ToString() != "ok")
        throw X::Error(outer.IsDict() && Has(outer, "error_message")
            ? outer.Get("error_message").ToString() : "Garnet search ranking failed");
    X::Value parsed = m_json["loads"](StripJsonFence(outer.Get("text").ToString()));
    if (parsed.IsList()) {
        X::Value wrapped = X::Value::Dict(Host());
        wrapped.SetItem("rankings", parsed);
        return wrapped;
    }
    if (!parsed.IsDict() || !Has(parsed, "rankings") || !parsed.Get("rankings").IsList())
        throw X::Error("Garnet returned invalid search ranking JSON");
    return parsed;
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
