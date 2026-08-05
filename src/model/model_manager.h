#pragma once

#include "xlang.h"

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Garnet
{
    class ModelManager
    {
    public:
        using ProgressSink = std::function<void(const std::string&)>;

        explicit ModelManager(
            ProgressSink progressSink = {},
            std::string catalogUrl = {},
            std::string catalogSignatureUrl = {},
            std::string storeName = "models",
            std::string archiveField = "xmodel",
            bool retainArchive = false);
        void SetProgressSink(ProgressSink progressSink);

        std::string Configure(const std::string& optionsJson);
        std::string ListRemote(X::XRuntime* runtime, bool refresh);
        std::string ListInstalled() const;
        std::string StartInstall(
            X::XRuntime* runtime,
            const std::string& modelId,
            const std::string& optionsJson);
        std::string InstallStatus(const std::string& jobId) const;
        std::string CancelInstall(const std::string& jobId);
        std::string VerifyInstalled(const std::string& modelId) const;
        std::string RemoveInstalled(const std::string& modelId);
        std::filesystem::path InstalledModelRoot(const std::string& modelId) const;
        std::filesystem::path CacheRoot(const std::string& modelId) const;

        void RunInstallJob(
            X::XRuntime* runtime,
            const std::string& jobId,
            const std::string& modelId,
            const std::string& optionsJson);

    private:
        struct Job;

        ProgressSink m_progressSink;
        mutable std::mutex m_mutex;
        std::filesystem::path m_installRoot;
        std::filesystem::path m_cacheRoot;
        std::filesystem::path m_publicKeyPath;
        std::string m_catalogUrl;
        std::string m_catalogSignatureUrl;
        std::string m_cachedCatalog;
        std::string m_archiveField;
        bool m_retainArchive = false;
        bool m_requireSignature = true;
        std::unordered_map<std::string, std::shared_ptr<Job>> m_jobs;

        std::string FetchCatalog(X::XRuntime* runtime, bool refresh);
        void UpdateJob(
            const std::shared_ptr<Job>& job,
            const std::string& phase,
            const std::string& message,
            bool terminal = false,
            const std::string& error = std::string()) const;
    };
}
