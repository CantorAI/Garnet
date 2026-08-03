#include "model_manager.h"

#include "nlohmann/json.hpp"
#include "xhost.h"

#include <openssl/evp.h>
#include <openssl/pem.h>

#include <chrono>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace Garnet
{
    namespace
    {
        using json = nlohmann::json;
        namespace fs = std::filesystem;

        constexpr const char* DefaultCatalogUrl =
            "https://app.garnetmodel.ai/api/v1/garnet/models/catalog";
        constexpr const char* DefaultCatalogSignatureUrl =
            "https://app.garnetmodel.ai/api/v1/garnet/models/catalog.sig";

        std::string JsonError(const std::string& code, const std::string& message)
        {
            return json({
                {"schema_version", 1},
                {"status", "error"},
                {"error", code},
                {"message", message}
            }).dump();
        }

        fs::path AbsoluteNormalized(const fs::path& value)
        {
            std::error_code error;
            const fs::path absolute = fs::absolute(value, error);
            return (error ? value : absolute).lexically_normal();
        }

        bool IsSafeName(const std::string& value)
        {
            if (value.empty() || value.size() > 128) return false;
            for (const unsigned char ch : value) {
                if (!(std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-')) {
                    return false;
                }
            }
            return value != "." && value != "..";
        }

        bool IsSafeRelativePath(const fs::path& value)
        {
            if (value.empty() || value.is_absolute()) return false;
            for (const fs::path& component : value) {
                if (component == ".." || component == ".") return false;
            }
            return true;
        }

        bool IsWithin(const fs::path& root, const fs::path& candidate)
        {
            const fs::path normalizedRoot = AbsoluteNormalized(root);
            const fs::path normalizedCandidate = AbsoluteNormalized(candidate);
            auto rootIt = normalizedRoot.begin();
            auto candidateIt = normalizedCandidate.begin();
            for (; rootIt != normalizedRoot.end(); ++rootIt, ++candidateIt) {
                if (candidateIt == normalizedCandidate.end() || *rootIt != *candidateIt) {
                    return false;
                }
            }
            return true;
        }

        std::string RandomId()
        {
            std::random_device source;
            std::mt19937_64 generator(source());
            std::ostringstream value;
            value << std::hex << std::setfill('0');
            for (int index = 0; index < 2; ++index) {
                value << std::setw(16) << generator();
            }
            return value.str();
        }

        bool ReadFile(const fs::path& path, std::string& content)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input) return false;
            content.assign(
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>());
            return static_cast<bool>(input) || input.eof();
        }

        std::string Sha256(const fs::path& path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input) return {};
            EVP_MD_CTX* context = EVP_MD_CTX_new();
            if (!context) return {};
            unsigned char digest[EVP_MAX_MD_SIZE]{};
            unsigned int digestLength = 0;
            bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
            std::vector<char> buffer(4 * 1024 * 1024);
            while (ok && input) {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const std::streamsize count = input.gcount();
                if (count > 0) {
                    ok = EVP_DigestUpdate(
                        context, buffer.data(), static_cast<size_t>(count)) == 1;
                }
            }
            ok = ok && input.eof() &&
                EVP_DigestFinal_ex(context, digest, &digestLength) == 1;
            EVP_MD_CTX_free(context);
            if (!ok) return {};
            std::ostringstream text;
            text << std::hex << std::setfill('0');
            for (unsigned int index = 0; index < digestLength; ++index) {
                text << std::setw(2) << static_cast<unsigned int>(digest[index]);
            }
            return text.str();
        }

        uint16_t ReadLe16(const unsigned char* value)
        {
            return static_cast<uint16_t>(value[0]) |
                (static_cast<uint16_t>(value[1]) << 8);
        }

        uint32_t ReadLe32(const unsigned char* value)
        {
            return static_cast<uint32_t>(value[0]) |
                (static_cast<uint32_t>(value[1]) << 8) |
                (static_cast<uint32_t>(value[2]) << 16) |
                (static_cast<uint32_t>(value[3]) << 24);
        }

        bool ExtractStoredZip(
            const fs::path& archivePath,
            const fs::path& destination,
            std::string& error)
        {
            std::ifstream archive(archivePath, std::ios::binary);
            if (!archive) {
                error = "cannot open xmodel archive";
                return false;
            }
            while (true) {
                unsigned char header[30]{};
                archive.read(reinterpret_cast<char*>(header), 4);
                if (archive.gcount() == 0 && archive.eof()) return true;
                if (archive.gcount() != 4) {
                    error = "truncated xmodel ZIP header";
                    return false;
                }
                const uint32_t signature = ReadLe32(header);
                if (signature == 0x02014b50 || signature == 0x06054b50) {
                    return true;
                }
                if (signature != 0x04034b50) {
                    error = "invalid xmodel ZIP entry";
                    return false;
                }
                archive.read(reinterpret_cast<char*>(header + 4), 26);
                if (!archive) {
                    error = "truncated xmodel ZIP entry";
                    return false;
                }
                const uint16_t flags = ReadLe16(header + 6);
                const uint16_t method = ReadLe16(header + 8);
                const uint32_t compressedSize = ReadLe32(header + 18);
                const uint32_t uncompressedSize = ReadLe32(header + 22);
                const uint16_t nameLength = ReadLe16(header + 26);
                const uint16_t extraLength = ReadLe16(header + 28);
                if ((flags & 0x0009) != 0 || method != 0 ||
                    compressedSize != uncompressedSize || nameLength == 0) {
                    error = "xmodel ZIP must contain unencrypted stored entries";
                    return false;
                }
                std::string name(nameLength, '\0');
                archive.read(name.data(), nameLength);
                archive.seekg(extraLength, std::ios::cur);
                if (!archive) {
                    error = "truncated xmodel ZIP path";
                    return false;
                }
                std::replace(name.begin(), name.end(), '\\', '/');
                const bool directoryEntry = !name.empty() && name.back() == '/';
                const fs::path relative(name);
                const fs::path output = destination / relative;
                if (!IsSafeRelativePath(relative) || !IsWithin(destination, output)) {
                    error = "unsafe path in xmodel ZIP";
                    return false;
                }
                std::error_code fileError;
                if (directoryEntry) {
                    fs::create_directories(output, fileError);
                } else {
                    fs::create_directories(output.parent_path(), fileError);
                    if (!fileError) {
                        std::ofstream target(output, std::ios::binary | std::ios::trunc);
                        if (!target) {
                            error = "cannot create extracted xmodel file";
                            return false;
                        }
                        std::vector<char> buffer(1024 * 1024);
                        uint64_t remaining = uncompressedSize;
                        while (remaining > 0) {
                            const size_t count = static_cast<size_t>(
                                std::min<uint64_t>(remaining, buffer.size()));
                            archive.read(buffer.data(), static_cast<std::streamsize>(count));
                            if (archive.gcount() != static_cast<std::streamsize>(count)) {
                                error = "truncated xmodel ZIP data";
                                return false;
                            }
                            target.write(buffer.data(), static_cast<std::streamsize>(count));
                            if (!target) {
                                error = "cannot write extracted xmodel file";
                                return false;
                            }
                            remaining -= count;
                        }
                    }
                }
                if (fileError) {
                    error = fileError.message();
                    return false;
                }
            }
        }

        std::vector<unsigned char> DecodeBase64Url(std::string value)
        {
            for (char& ch : value) {
                if (ch == '-') ch = '+';
                else if (ch == '_') ch = '/';
            }
            while (value.size() % 4 != 0) value.push_back('=');
            std::vector<unsigned char> decoded((value.size() / 4) * 3 + 3);
            const int count = EVP_DecodeBlock(
                decoded.data(),
                reinterpret_cast<const unsigned char*>(value.data()),
                static_cast<int>(value.size()));
            if (count < 0) return {};
            size_t padding = 0;
            if (!value.empty() && value.back() == '=') ++padding;
            if (value.size() > 1 && value[value.size() - 2] == '=') ++padding;
            decoded.resize(static_cast<size_t>(count) - padding);
            return decoded;
        }

        bool VerifyDetachedSignature(
            const std::string& content,
            const std::string& signatureText,
            const fs::path& publicKeyPath)
        {
            std::string signature = signatureText;
            while (!signature.empty() && std::isspace(
                static_cast<unsigned char>(signature.back()))) {
                signature.pop_back();
            }
            const std::vector<unsigned char> bytes = DecodeBase64Url(signature);
            if (bytes.empty()) return false;
            std::string keyContent;
            if (!ReadFile(publicKeyPath, keyContent)) return false;
            BIO* keyBuffer = BIO_new_mem_buf(
                keyContent.data(), static_cast<int>(keyContent.size()));
            if (!keyBuffer) return false;
            EVP_PKEY* key = PEM_read_bio_PUBKEY(
                keyBuffer, nullptr, nullptr, nullptr);
            BIO_free(keyBuffer);
            if (!key) return false;
            EVP_MD_CTX* context = EVP_MD_CTX_new();
            const bool ok = context &&
                EVP_DigestVerifyInit(context, nullptr, nullptr, nullptr, key) == 1 &&
                EVP_DigestVerify(
                    context,
                    bytes.data(), bytes.size(),
                    reinterpret_cast<const unsigned char*>(content.data()),
                    content.size()) == 1;
            EVP_MD_CTX_free(context);
            EVP_PKEY_free(key);
            return ok;
        }

        struct UrlParts
        {
            std::string origin;
            std::string path;
        };

        bool ParseHttpsUrl(const std::string& url, UrlParts& parts)
        {
            constexpr const char* Prefix = "https://";
            if (url.rfind(Prefix, 0) != 0) return false;
            const size_t slash = url.find('/', 8);
            parts.origin = slash == std::string::npos ? url : url.substr(0, slash);
            parts.path = slash == std::string::npos ? "/" : url.substr(slash);
            return parts.origin.size() > 8;
        }

        bool ImportHttp(X::XRuntime* runtime, X::Value& http)
        {
            return X::g_pXHost &&
                X::g_pXHost->Import(runtime, "http", nullptr, nullptr, http) &&
                http.IsObject();
        }

        bool HttpGet(
            X::XRuntime* runtime,
            const std::string& url,
            std::string& content,
            std::string& error)
        {
            UrlParts parts;
            if (!ParseHttpsUrl(url, parts)) {
                error = "only HTTPS catalog URLs are allowed";
                return false;
            }
            X::Value http;
            if (!ImportHttp(runtime, http)) {
                error = "the XLang HTTP package is unavailable";
                return false;
            }
            X::Value client = http["Client"](parts.origin);
            if (!client.IsObject() || !client["get"](parts.path).ToBool()) {
                error = "HTTP GET failed";
                return false;
            }
            const int status = client["status"]().ToInt();
            if (status < 200 || status >= 300) {
                error = "HTTP GET returned status " + std::to_string(status);
                return false;
            }
            content = client["body"]().ToString();
            return true;
        }

        bool HttpDownload(
            X::XRuntime* runtime,
            const std::string& url,
            const fs::path& target,
            X::Value callback,
            std::string& error)
        {
            UrlParts parts;
            if (!ParseHttpsUrl(url, parts)) {
                error = "only HTTPS artifact URLs are allowed";
                return false;
            }
            X::Value http;
            if (!ImportHttp(runtime, http)) {
                error = "the XLang HTTP package is unavailable";
                return false;
            }
            X::Value client = http["Client"](parts.origin);
            if (!client.IsObject()) {
                error = "HTTP client creation failed";
                return false;
            }
            const bool downloaded =
                client["download"](parts.path, target.string(), callback).ToBool();
            const int status = client["status"]().ToInt();
            if (!downloaded) {
                error = "artifact download failed (HTTP " +
                    std::to_string(status) + ")";
                return false;
            }
            if (status != 200 && status != 206) {
                error = "artifact download returned status " + std::to_string(status);
                return false;
            }
            return true;
        }

        const json* FindModel(const json& catalog, const std::string& modelId)
        {
            if (!catalog.contains("models") || !catalog["models"].is_array()) {
                return nullptr;
            }
            for (const json& model : catalog["models"]) {
                if (model.is_object() && model.value("id", "") == modelId) {
                    return &model;
                }
            }
            return nullptr;
        }
    }

    struct ModelManager::Job
    {
        std::string id;
        std::string modelId;
        std::string phase = "queued";
        std::string message = "Queued";
        std::string error;
        uint64_t filesCompleted = 0;
        uint64_t filesTotal = 0;
        uint64_t bytesReceived = 0;
        uint64_t bytesTotal = 0;
        bool terminal = false;
        std::atomic<bool> cancelled{false};
        mutable std::mutex mutex;

        json Snapshot() const
        {
            std::lock_guard<std::mutex> guard(mutex);
            const double percent = bytesTotal == 0
                ? 0.0
                : (100.0 * static_cast<double>(bytesReceived) /
                    static_cast<double>(bytesTotal));
            return {
                {"schema_version", 1},
                {"job_id", id},
                {"model_id", modelId},
                {"phase", phase},
                {"files_completed", filesCompleted},
                {"files_total", filesTotal},
                {"bytes_received", bytesReceived},
                {"bytes_total", bytesTotal},
                {"percent", std::min(100.0, percent)},
                {"message", message},
                {"error", error},
                {"terminal", terminal}
            };
        }
    };

    ModelManager::ModelManager(ProgressSink progressSink)
        : m_progressSink(std::move(progressSink)),
          m_catalogUrl(DefaultCatalogUrl),
          m_catalogSignatureUrl(DefaultCatalogSignatureUrl)
    {
        const fs::path base = fs::temp_directory_path() / "cantorai-garnet";
        m_installRoot = base / "models";
        m_cacheRoot = base / "cache";
    }

    void ModelManager::SetProgressSink(ProgressSink progressSink)
    {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_progressSink = std::move(progressSink);
    }

    std::string ModelManager::Configure(const std::string& optionsJson)
    {
        try {
            const json options = optionsJson.empty()
                ? json::object()
                : json::parse(optionsJson);
            std::lock_guard<std::mutex> guard(m_mutex);
            if (options.contains("catalog_url")) {
                m_catalogUrl = options["catalog_url"].get<std::string>();
            }
            if (options.contains("catalog_signature_url")) {
                m_catalogSignatureUrl =
                    options["catalog_signature_url"].get<std::string>();
            }
            if (options.contains("install_root")) {
                m_installRoot = AbsoluteNormalized(
                    options["install_root"].get<std::string>());
            }
            if (options.contains("cache_root")) {
                m_cacheRoot = AbsoluteNormalized(
                    options["cache_root"].get<std::string>());
            }
            if (options.contains("public_key_path")) {
                m_publicKeyPath = AbsoluteNormalized(
                    options["public_key_path"].get<std::string>());
            }
            m_requireSignature = options.value("require_signature", true);
            fs::create_directories(m_installRoot);
            fs::create_directories(m_cacheRoot);
            return json({
                {"schema_version", 1},
                {"status", "ok"},
                {"catalog_url", m_catalogUrl},
                {"install_root", m_installRoot.string()},
                {"cache_root", m_cacheRoot.string()},
                {"require_signature", m_requireSignature}
            }).dump();
        }
        catch (const std::exception& exception) {
            return JsonError("configuration_invalid", exception.what());
        }
    }

    std::string ModelManager::FetchCatalog(X::XRuntime* runtime, bool refresh)
    {
        std::string catalogUrl;
        std::string signatureUrl;
        fs::path keyPath;
        bool requireSignature = true;
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            if (!refresh && !m_cachedCatalog.empty()) return m_cachedCatalog;
            catalogUrl = m_catalogUrl;
            signatureUrl = m_catalogSignatureUrl;
            keyPath = m_publicKeyPath;
            requireSignature = m_requireSignature;
        }

        std::string catalog;
        std::string error;
        if (!HttpGet(runtime, catalogUrl, catalog, error)) {
            throw std::runtime_error(error);
        }
        const json parsed = json::parse(catalog);
        if (parsed.value("schema_version", 0) != 1 ||
            !parsed.contains("models") || !parsed["models"].is_array()) {
            throw std::runtime_error("the remote catalog contract is invalid");
        }

        if (requireSignature) {
            if (keyPath.empty() || !fs::is_regular_file(keyPath)) {
                throw std::runtime_error("the catalog verification key is unavailable");
            }
            std::string signature;
            if (!HttpGet(runtime, signatureUrl, signature, error)) {
                throw std::runtime_error(error);
            }
            if (!VerifyDetachedSignature(catalog, signature, keyPath)) {
                throw std::runtime_error("the remote catalog signature is invalid");
            }
        }

        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_cachedCatalog = catalog;
        }
        return catalog;
    }

    std::string ModelManager::ListRemote(X::XRuntime* runtime, bool refresh)
    {
        try {
            json catalog = json::parse(FetchCatalog(runtime, refresh));
            for (json& model : catalog["models"]) {
                const std::string id = model.value("id", "");
                model["installed"] = IsSafeName(id) &&
                    fs::is_regular_file(InstalledModelRoot(id) / ".garnet-model.json");
                model["compatible"] = true;
            }
            catalog["catalog_verified"] = m_requireSignature;
            return catalog.dump();
        }
        catch (const std::exception& exception) {
            return JsonError("catalog_unavailable", exception.what());
        }
    }

    std::string ModelManager::ListInstalled() const
    {
        json response = {
            {"schema_version", 1},
            {"install_root", m_installRoot.string()},
            {"models", json::array()}
        };
        std::error_code error;
        if (!fs::is_directory(m_installRoot, error)) return response.dump();
        for (const auto& entry : fs::directory_iterator(m_installRoot, error)) {
            if (error) break;
            const fs::path manifest = entry.path() / ".garnet-model.json";
            std::string content;
            if (!entry.is_directory() || !ReadFile(manifest, content)) continue;
            try {
                json model = json::parse(content);
                model["install_root"] = entry.path().string();
                model["installed"] = true;
                response["models"].push_back(std::move(model));
            }
            catch (...) {}
        }
        return response.dump();
    }

    std::string ModelManager::StartInstall(
        X::XRuntime* runtime,
        const std::string& modelId,
        const std::string& optionsJson)
    {
        if (!IsSafeName(modelId)) {
            return JsonError("model_id_invalid", "the model ID is invalid");
        }
        auto job = std::make_shared<Job>();
        job->id = RandomId();
        job->modelId = modelId;
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            for (const auto& item : m_jobs) {
                if (item.second->modelId == modelId &&
                    !item.second->Snapshot().value("terminal", false)) {
                    return JsonError(
                        "install_in_progress",
                        "an installation for this model is already active");
                }
            }
            m_jobs[job->id] = job;
        }

        try {
            std::thread([
                this, runtime, jobId = job->id, modelId, optionsJson
            ]() {
                RunInstallJob(runtime, jobId, modelId, optionsJson);
            }).detach();
        }
        catch (const std::exception& exception) {
            UpdateJob(job, "error", "Could not start install worker", true,
                exception.what());
            return JsonError(
                "worker_start_failed",
                job->Snapshot().value("error", "worker start failed"));
        }
        return json({
            {"schema_version", 1},
            {"status", "accepted"},
            {"job_id", job->id},
            {"model_id", modelId}
        }).dump();
    }

    std::string ModelManager::InstallStatus(const std::string& jobId) const
    {
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            const auto found = m_jobs.find(jobId);
            if (found == m_jobs.end()) {
                return JsonError("job_not_found", "the install job was not found");
            }
            job = found->second;
        }
        return job->Snapshot().dump();
    }

    std::string ModelManager::CancelInstall(const std::string& jobId)
    {
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            const auto found = m_jobs.find(jobId);
            if (found == m_jobs.end()) {
                return JsonError("job_not_found", "the install job was not found");
            }
            job = found->second;
        }
        job->cancelled = true;
        return job->Snapshot().dump();
    }

    void ModelManager::UpdateJob(
        const std::shared_ptr<Job>& job,
        const std::string& phase,
        const std::string& message,
        bool terminal,
        const std::string& error) const
    {
        {
            std::lock_guard<std::mutex> guard(job->mutex);
            job->phase = phase;
            job->message = message;
            job->terminal = terminal;
            job->error = error;
        }
        if (m_progressSink) m_progressSink(job->Snapshot().dump());
    }

    void ModelManager::RunInstallJob(
        X::XRuntime* runtime,
        const std::string& jobId,
        const std::string& modelId,
        const std::string&)
    {
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            const auto found = m_jobs.find(jobId);
            if (found == m_jobs.end()) return;
            job = found->second;
        }

        try {
            UpdateJob(job, "catalog", "Resolving signed model catalog");
            const json catalog = json::parse(FetchCatalog(runtime, false));
            const json* model = FindModel(catalog, modelId);
            if (!model) throw std::runtime_error("the selected model is not in the catalog");
            if (!model->contains("xmodel") || !(*model)["xmodel"].is_object() ||
                !model->contains("weights") || !(*model)["weights"].is_array()) {
                throw std::runtime_error("the model file manifest is invalid");
            }

            const json& xmodel = (*model)["xmodel"];
            const json& weights = (*model)["weights"];
            uint64_t totalBytes = xmodel.value("size_bytes", uint64_t{0});
            for (const json& file : weights) {
                totalBytes += file.value("size_bytes", uint64_t{0});
            }
            {
                std::lock_guard<std::mutex> guard(job->mutex);
                job->bytesTotal = totalBytes;
                job->filesTotal = weights.size() + 1;
            }

            const fs::path staging = m_installRoot /
                ("." + modelId + ".staging-" + jobId);
            const fs::path downloadRoot = m_cacheRoot / "downloads" /
                modelId / model->value("version", "unknown");
            if (!IsWithin(m_installRoot, staging)) {
                throw std::runtime_error("unsafe staging path");
            }
            if (!IsWithin(m_cacheRoot, downloadRoot)) {
                throw std::runtime_error("unsafe download cache path");
            }
            std::error_code error;
            fs::remove_all(staging, error);
            error.clear();
            fs::create_directories(staging, error);
            if (error) throw std::runtime_error(error.message());

            auto downloadPart = [this, runtime, job, &error](
                const json& part,
                const fs::path& partPath,
                uint64_t baseBytes,
                const std::string& label) -> uint64_t {
                fs::create_directories(partPath.parent_path());
                const uint64_t expectedSize =
                    part.value("size_bytes", uint64_t{0});
                X::U_FUNC progress = [this, job, baseBytes](
                    X::XRuntime*, X::XObj*, X::XObj*,
                    X::ARGS& params, X::KWARGS&, X::Value& result) {
                    const uint64_t received = params.size() == 0
                        ? 0
                        : static_cast<uint64_t>(params[0].ToLongLong());
                    {
                        std::lock_guard<std::mutex> guard(job->mutex);
                        job->bytesReceived = baseBytes + received;
                    }
                    if (m_progressSink) m_progressSink(job->Snapshot().dump());
                    result = !job->cancelled.load();
                    return true;
                };
                X::Value callback(
                    X::g_pXHost->CreateFunction(
                        "garnet_download_progress", progress, nullptr),
                    false);
                bool ready = false;
                if (fs::is_regular_file(partPath, error) && !error) {
                    const uint64_t cachedSize = fs::file_size(partPath, error);
                    if (!error && cachedSize == expectedSize &&
                        Sha256(partPath) == part.value("sha256", "")) {
                        ready = true;
                    } else if (!error && cachedSize > expectedSize) {
                        fs::remove(partPath, error);
                        error.clear();
                    }
                }
                if (!ready) {
                    std::string downloadError;
                    UpdateJob(job, "downloading", "Downloading " + label);
                    if (!HttpDownload(
                            runtime, part.value("url", ""), partPath,
                            callback, downloadError)) {
                        throw std::runtime_error(
                            "download failed for " + label + ": " + downloadError);
                    }
                }
                const uint64_t actualSize = fs::file_size(partPath);
                if (actualSize != expectedSize ||
                    Sha256(partPath) != part.value("sha256", "")) {
                    throw std::runtime_error(
                        "download verification failed for " + label);
                }
                return actualSize;
            };

            const fs::path archiveCache = downloadRoot / "xmodel.zip.partial";
            const uint64_t archiveBytes = downloadPart(
                xmodel, archiveCache, 0, "xmodel.zip");
            const fs::path installedArchive = staging / "xmodel.zip";
            fs::copy_file(
                archiveCache, installedArchive,
                fs::copy_options::overwrite_existing, error);
            if (error) throw std::runtime_error(error.message());
            UpdateJob(job, "extracting", "Extracting xmodel.zip");
            std::string extractionError;
            if (!ExtractStoredZip(installedArchive, staging, extractionError)) {
                throw std::runtime_error(extractionError);
            }

            uint64_t completedBytes = archiveBytes;
            size_t fileIndex = 0;
            {
                std::lock_guard<std::mutex> guard(job->mutex);
                job->bytesReceived = completedBytes;
                job->filesCompleted = 1;
            }
            for (const json& file : weights) {
                if (job->cancelled) throw std::runtime_error("installation cancelled");
                const fs::path relative(file.value("path", ""));
                if (!IsSafeRelativePath(relative)) {
                    throw std::runtime_error("unsafe model package path");
                }
                const fs::path outputPath = staging / relative;
                if (!IsWithin(staging, outputPath)) {
                    throw std::runtime_error("model package path escaped staging");
                }
                fs::create_directories(outputPath.parent_path());
                const json& parts = file["parts"];
                std::vector<fs::path> downloadedParts;
                uint64_t fileDownloaded = 0;
                size_t partIndex = 0;

                for (const json& part : parts) {
                    if (job->cancelled) throw std::runtime_error("installation cancelled");
                    const fs::path partPath = downloadRoot /
                        ("weight-" + std::to_string(fileIndex) + "-" +
                         std::to_string(partIndex) + ".partial");
                    const uint64_t baseBytes = completedBytes + fileDownloaded;
                    const uint64_t actualSize = downloadPart(
                        part, partPath, baseBytes,
                        relative.string() + " part " +
                            std::to_string(partIndex + 1));
                    downloadedParts.push_back(partPath);
                    fileDownloaded += actualSize;
                    ++partIndex;
                }

                UpdateJob(job, "reconstructing", "Reconstructing " + relative.string());
                std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
                if (!output) throw std::runtime_error("cannot create installed model file");
                std::vector<char> buffer(4 * 1024 * 1024);
                for (const fs::path& partPath : downloadedParts) {
                    std::ifstream input(partPath, std::ios::binary);
                    while (input) {
                        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                        const std::streamsize count = input.gcount();
                        if (count > 0) output.write(buffer.data(), count);
                    }
                    if (!input.eof() || !output) {
                        throw std::runtime_error("model file reconstruction failed");
                    }
                }
                output.close();

                UpdateJob(job, "verifying", "Verifying " + relative.string());
                if (fs::file_size(outputPath) != file.value("size_bytes", uint64_t{0}) ||
                    Sha256(outputPath) != file.value("sha256", "")) {
                    throw std::runtime_error("reconstructed model file verification failed");
                }
                completedBytes += fs::file_size(outputPath);
                {
                    std::lock_guard<std::mutex> guard(job->mutex);
                    job->bytesReceived = completedBytes;
                    ++fileIndex;
                    job->filesCompleted = fileIndex + 1;
                }
            }

            std::ofstream marker(staging / ".garnet-model.json", std::ios::binary);
            marker << model->dump(2) << '\n';
            marker.close();
            if (!marker) throw std::runtime_error("cannot write installed model marker");

            UpdateJob(job, "installing", "Activating verified model package");
            const fs::path target = InstalledModelRoot(modelId);
            const fs::path backup = m_installRoot / ("." + modelId + ".previous");
            if (!IsWithin(m_installRoot, target) || !IsWithin(m_installRoot, backup)) {
                throw std::runtime_error("unsafe installation target");
            }
            fs::remove_all(backup, error);
            error.clear();
            if (fs::exists(target)) {
                fs::rename(target, backup, error);
                if (error) throw std::runtime_error(error.message());
            }
            fs::rename(staging, target, error);
            if (error) {
                if (fs::exists(backup)) {
                    std::error_code rollbackError;
                    fs::rename(backup, target, rollbackError);
                }
                throw std::runtime_error(error.message());
            }
            fs::remove_all(backup, error);
            error.clear();
            fs::remove_all(downloadRoot, error);
            UpdateJob(job, "complete", "Model installed", true);
        }
        catch (const std::exception& exception) {
            const bool cancelled = job->cancelled ||
                std::string(exception.what()) == "installation cancelled";
            UpdateJob(
                job,
                cancelled ? "cancelled" : "error",
                cancelled ? "Installation cancelled" : "Installation failed",
                true,
                cancelled ? std::string() : exception.what());
        }
    }

    fs::path ModelManager::InstalledModelRoot(const std::string& modelId) const
    {
        if (!IsSafeName(modelId)) return {};
        return AbsoluteNormalized(m_installRoot / modelId);
    }

    fs::path ModelManager::CacheRoot(const std::string& modelId) const
    {
        if (!IsSafeName(modelId)) return {};
        return AbsoluteNormalized(m_cacheRoot / modelId);
    }

    std::string ModelManager::VerifyInstalled(const std::string& modelId) const
    {
        try {
            const fs::path root = InstalledModelRoot(modelId);
            std::string markerText;
            if (root.empty() || !ReadFile(root / ".garnet-model.json", markerText)) {
                return JsonError("model_not_installed", "the model is not installed");
            }
            const json marker = json::parse(markerText);
            auto verifyFile = [&root](const json& file) -> std::string {
                const fs::path relative(file.value("path", ""));
                const fs::path candidate = root / relative;
                if (!IsSafeRelativePath(relative) || !IsWithin(root, candidate) ||
                    !fs::is_regular_file(candidate) ||
                    fs::file_size(candidate) != file.value("size_bytes", uint64_t{0}) ||
                    Sha256(candidate) != file.value("sha256", "")) {
                    return relative.string();
                }
                return {};
            };
            std::string failed;
            if (marker.contains("files") && marker["files"].is_array()) {
                for (const json& file : marker["files"]) {
                    failed = verifyFile(file);
                    if (!failed.empty()) break;
                }
            } else {
                failed = verifyFile(marker["xmodel"]);
                if (failed.empty()) {
                    for (const json& file : marker["weights"]) {
                        failed = verifyFile(file);
                        if (!failed.empty()) break;
                    }
                }
            }
            if (!failed.empty()) {
                return JsonError(
                    "model_verification_failed",
                    "installed file verification failed: " + failed);
            }
            return json({
                {"schema_version", 1},
                {"status", "ok"},
                {"model_id", modelId},
                {"verified", true}
            }).dump();
        }
        catch (const std::exception& exception) {
            return JsonError("model_verification_failed", exception.what());
        }
    }

    std::string ModelManager::RemoveInstalled(const std::string& modelId)
    {
        if (!IsSafeName(modelId)) {
            return JsonError("model_id_invalid", "the model ID is invalid");
        }
        const fs::path target = InstalledModelRoot(modelId);
        if (target.empty() || !IsWithin(m_installRoot, target)) {
            return JsonError("model_path_invalid", "the installed model path is invalid");
        }
        std::error_code error;
        const uintmax_t removed = fs::remove_all(target, error);
        if (error) return JsonError("model_remove_failed", error.message());
        return json({
            {"schema_version", 1},
            {"status", "ok"},
            {"model_id", modelId},
            {"removed", removed > 0}
        }).dump();
    }
}
