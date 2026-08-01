#pragma once

#include <filesystem>
#include <string>

namespace Garnet
{
    std::filesystem::path RuntimeModuleFolder();

    std::filesystem::path ResolveModelCatalogRoot(
        const std::string& requestedRoot,
        const std::string& runtimeBaseFolder);

    std::string EnumerateAvailableModelsJson(
        const std::filesystem::path& catalogRoot);
}
