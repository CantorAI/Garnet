#include "nlohmann/json.hpp"
#include "xlang.h"
#include "xload.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace
{
    char* CopyConfigString(const std::string& value)
    {
        char* result = new char[value.size() + 1];
        std::copy(value.begin(), value.end(), result);
        result[value.size()] = '\0';
        return result;
    }

    std::string Env(const char* name, const std::string& fallback = {})
    {
        const char* value = std::getenv(name);
        return value && *value ? std::string(value) : fallback;
    }

    fs::path ExecutableDirectory(const char* argv0)
    {
        std::error_code ec;
        fs::path path = fs::absolute(argv0, ec);
        if (ec) path = fs::current_path() / argv0;
        return path.parent_path();
    }

    json ParseGarnetJson(X::Value value)
    {
        const std::string text = value.ToString();
        try {
            return json::parse(text);
        }
        catch (...) {
            return {{"status", "error"}, {"error_code", "invalid_garnet_json"},
                {"error_message", text}};
        }
    }

    std::string FirstTextFromMessages(const json& body)
    {
        if (body.contains("prompt") && body["prompt"].is_string()) {
            return body["prompt"].get<std::string>();
        }
        std::string prompt;
        for (const auto& message : body.value("messages", json::array())) {
            const json& content = message.value("content", json());
            if (content.is_string()) {
                if (!prompt.empty()) prompt += "\n";
                prompt += content.get<std::string>();
            }
            else if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.value("type", "") == "text" && part.contains("text")) {
                        if (!prompt.empty()) prompt += "\n";
                        prompt += part["text"].get<std::string>();
                    }
                }
            }
        }
        return prompt.empty() ? "Respond briefly." : prompt;
    }

    std::string FirstImageFromBody(const json& body)
    {
        if (body.contains("image_path") && body["image_path"].is_string()) {
            return body["image_path"].get<std::string>();
        }
        for (const auto& message : body.value("messages", json::array())) {
            const json& content = message.value("content", json());
            if (!content.is_array()) continue;
            for (const auto& part : content) {
                if (part.value("type", "") != "image_url") continue;
                const json& image = part.value("image_url", json());
                if (image.is_string()) return image.get<std::string>();
                if (image.contains("url") && image["url"].is_string()) {
                    return image["url"].get<std::string>();
                }
            }
        }
        return {};
    }

    class GarnetServingProcess
    {
    public:
        GarnetServingProcess(const fs::path& executableDir, const std::string& defaultModel)
            : defaultModelId_(defaultModel.empty() ? Env("GARNET_DEFAULT_MODEL_ID") : defaultModel)
        {
            const fs::path xlangDir = Env("GARNET_XLANG_DIR").empty()
                ? executableDir
                : fs::path(Env("GARNET_XLANG_DIR"));
            config_.appPath = CopyConfigString(xlangDir.string());
            config_.appFullName = CopyConfigString((executableDir / "garnet-serving").string());
            config_.dllSearchPath = CopyConfigString(xlangDir.string());
            config_.enablePython = false;
            config_.enterEventLoop = false;
            config_.runEventLoopInThread = false;
            if (loader_.Load(&config_) != 0 || loader_.Run() != 0) {
                throw std::runtime_error("failed to initialize XLang host");
            }
            X::Runtime rt;
            runtime_ = rt;
            X::Package garnet(rt, "garnet", "garnet");
            garnet_ = garnet;
            if (!garnet_.IsObject()) {
                throw std::runtime_error("failed to import Garnet XLang package");
            }
            ConfigureManagers();
        }

        ~GarnetServingProcess()
        {
            try {
                if (serving_) garnet_["stop_serving"]();
            }
            catch (...) {
            }
            garnet_ = X::Value();
            runtime_ = X::Value();
            loader_.Unload();
        }

        json Handle(const json& request)
        {
            const std::string path = request.value("path", request.value("route", ""));
            const json body = request.value("body", json::object());
            const std::string model = body.value("model", defaultModelId_);
            if (path == "component:/v1/health" || path == "/v1/health") {
                return {{"status", "ok"}, {"ready", true}, {"model", activeModelId_}};
            }
            if (path == "component:/v1/models" || path == "/v1/models") {
                return ParseGarnetJson(garnet_["list_installed_models_json"]());
            }
            if (path == "component:/v1/chat/completions" || path == "/v1/chat/completions") {
                EnsureModel(model);
                const std::string prompt = FirstTextFromMessages(body);
                const std::string image = FirstImageFromBody(body);
                const int maxTokens = body.value("max_tokens", 256);
                json result = ParseGarnetJson(garnet_["infer_json"](prompt, image, maxTokens));
                if (result.value("status", "") != "ok") return result;
                return {
                    {"id", request.value("id", "")},
                    {"object", "chat.completion"},
                    {"model", activeModelId_},
                    {"choices", json::array({{
                        {"index", 0},
                        {"message", {{"role", "assistant"}, {"content", result.value("text", "")}}},
                        {"finish_reason", "stop"}
                    }})},
                    {"garnet", result}
                };
            }
            if (path == "component:/v1/responses" || path == "/v1/responses") {
                EnsureModel(model);
                const std::string prompt = body.value("input", FirstTextFromMessages(body));
                const std::string image = FirstImageFromBody(body);
                const int maxTokens = body.value("max_output_tokens", body.value("max_tokens", 256));
                json result = ParseGarnetJson(garnet_["infer_json"](prompt, image, maxTokens));
                if (result.value("status", "") != "ok") return result;
                return {
                    {"id", request.value("id", "")},
                    {"object", "response"},
                    {"model", activeModelId_},
                    {"output_text", result.value("text", "")},
                    {"garnet", result}
                };
            }
            if (path == "component:/v1/audio/speech" || path == "/v1/audio/speech") {
                const std::string speechModel = body.contains("model")
                    ? body.value("model", "")
                    : Env("GARNET_DEFAULT_TTS_MODEL_ID", defaultModelId_);
                EnsureModel(speechModel);
                fs::path outputPath = body.value("output_path", "");
                if (outputPath.empty()) {
                    const fs::path root = Env("GARNET_DATA_ROOT").empty()
                        ? fs::temp_directory_path() / "garnet"
                        : fs::path(Env("GARNET_DATA_ROOT"));
                    fs::create_directories(root / "speech");
                    outputPath = root / "speech" / (request.value("id", "speech") + ".wav");
                }
                const std::string text = body.value("input", body.value("text", ""));
                const std::string voice = body.value("voice", Env("GARNET_TTS_SPEAKER", "Ryan"));
                const std::string language = body.value("language", "English");
                const int maxFrames = body.value("max_audio_frames", 256);
                json result = ParseGarnetJson(garnet_["synthesize_json"](
                    text, voice, language, maxFrames, outputPath.string()));
                result["output_path"] = outputPath.string();
                return result;
            }
            if (path == "component:/v1/audio/transcriptions" ||
                path == "/v1/audio/transcriptions") {
                const std::string asrModel = body.contains("model")
                    ? body.value("model", "")
                    : Env("GARNET_DEFAULT_ASR_MODEL_ID", defaultModelId_);
                EnsureModel(asrModel);
                const std::string audioPath = body.value("file", body.value("audio_path", ""));
                const std::string prompt = body.value("prompt", "");
                const std::string language = body.value("language", "English");
                const int maxTokens = body.value("max_tokens", 256);
                json result = ParseGarnetJson(garnet_["transcribe_json"](
                    audioPath, prompt, language, maxTokens));
                if (result.value("status", "") != "ok") return result;
                return {
                    {"text", result.value("text", "")},
                    {"model", activeModelId_},
                    {"garnet", result}
                };
            }
            if (path == "component:/v1/images/describe" || path == "/v1/images/describe") {
                const std::string visionModel = body.contains("model")
                    ? body.value("model", "")
                    : Env("GARNET_DEFAULT_VLM_MODEL_ID", defaultModelId_);
                EnsureModel(visionModel);
                const std::string prompt = body.value("prompt", "Describe this image.");
                const std::string image = body.value("image_path", FirstImageFromBody(body));
                const int maxTokens = body.value("max_tokens", 256);
                return ParseGarnetJson(garnet_["infer_json"](prompt, image, maxTokens));
            }
            return {{"status", "error"}, {"error_code", "route_not_found"},
                {"error_message", "Unsupported Garnet component route: " + path}};
        }

    private:
        void ConfigureManagers()
        {
            const fs::path root = Env("GARNET_DATA_ROOT").empty()
                ? fs::path(Env("USERPROFILE", ".")) / ".cantorai" / "garnet"
                : fs::path(Env("GARNET_DATA_ROOT"));
            const fs::path installRoot = Env("GARNET_MODEL_INSTALL_ROOT").empty()
                ? root / "models" : fs::path(Env("GARNET_MODEL_INSTALL_ROOT"));
            const fs::path cacheRoot = Env("GARNET_MODEL_CACHE_ROOT").empty()
                ? root / "cache" : fs::path(Env("GARNET_MODEL_CACHE_ROOT"));
            json options = {
                {"install_root", installRoot.generic_string()},
                {"cache_root", cacheRoot.generic_string()}
            };
            if (!Env("GARNET_CATALOG_URL").empty()) {
                options["catalog_url"] = Env("GARNET_CATALOG_URL");
            }
            if (!Env("GARNET_CATALOG_SIGNATURE_URL").empty()) {
                options["catalog_signature_url"] = Env("GARNET_CATALOG_SIGNATURE_URL");
            }
            garnet_["configure_model_manager_json"](options.dump());
            garnet_["configure_acceleration_manager_json"](json::object().dump());
        }

        void EnsureModel(const std::string& modelId)
        {
            const std::string wanted = modelId.empty() ? defaultModelId_ : modelId;
            if (wanted.empty()) {
                throw std::runtime_error(
                    "model is required; provide body.model or set a GARNET_DEFAULT_*_MODEL_ID environment value");
            }
            if (serving_ && activeModelId_ == wanted) return;
            if (serving_) {
                garnet_["stop_serving"]();
                serving_ = false;
                activeModelId_.clear();
            }
            const json installed = ParseGarnetJson(garnet_["list_installed_models_json"]());
            const std::string installedText = installed.dump();
            if (installedText.find("\"" + wanted + "\"") == std::string::npos) {
                if (Env("GARNET_AUTO_INSTALL", "1") != "1") {
                    throw std::runtime_error(wanted + " is not installed");
                }
                const json accepted = ParseGarnetJson(garnet_["install_model_json"](wanted, "{}"));
                const std::string job = accepted.value("job_id", "");
                if (job.empty()) throw std::runtime_error(accepted.dump());
                for (;;) {
                    const json status = ParseGarnetJson(garnet_["model_install_status_json"](job));
                    if (status.value("terminal", false)) {
                        if (status.value("phase", "") != "complete") {
                            throw std::runtime_error(status.dump());
                        }
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
            }
            const json loaded = ParseGarnetJson(
                garnet_["serve_installed_model_json"](wanted, "{}"));
            if (!loaded.value("ready", false)) throw std::runtime_error(loaded.dump());
            activeModelId_ = wanted;
            serving_ = true;
        }

        std::string defaultModelId_;
        std::string activeModelId_;
        X::Config config_{};
        X::XLoad loader_;
        X::Value runtime_;
        X::Value garnet_;
        bool serving_ = false;
    };
}

int main(int argc, char** argv)
{
    try {
        std::string defaultModel;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) defaultModel = argv[++i];
            else if (arg == "--manifest") {
                const fs::path manifest = ExecutableDirectory(argv[0]) /
                    "component" / "cantorfiber" / "component.json";
                std::ifstream stream(manifest, std::ios::binary);
                std::cout << stream.rdbuf() << std::endl;
                return stream ? 0 : 1;
            }
        }
        GarnetServingProcess process(ExecutableDirectory(argv[0]), defaultModel);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            json response;
            try {
                response = process.Handle(json::parse(line));
            }
            catch (const std::exception& error) {
                response = {{"status", "error"}, {"error_code", "request_failed"},
                    {"error_message", error.what()}};
            }
            std::cout << response.dump() << std::endl;
        }
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "garnet-serving failed: " << error.what() << std::endl;
        return 1;
    }
}
