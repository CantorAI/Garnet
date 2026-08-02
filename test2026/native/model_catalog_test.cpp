#include "model_catalog.h"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <iostream>
#include <set>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: garnet_model_catalog_test <xModel-root>\n";
        return 2;
    }

    const auto catalog = nlohmann::json::parse(
        Garnet::EnumerateAvailableModelsJson(argv[1]));
    if (catalog.value("schema_version", 0) != 1 ||
        !catalog["errors"].empty() || catalog["models"].size() != 4) {
        std::cerr << catalog.dump(2) << '\n';
        return 1;
    }

    std::set<std::string> ids;
    for (const auto& model : catalog["models"]) {
        if (!model.value("available", false) ||
            model["profiles"].empty() || model["entrypoints"].empty()) {
            std::cerr << model.dump(2) << '\n';
            return 1;
        }
        ids.insert(model.value("id", ""));
        if (model["weights"].value("bundled", true)) {
            std::cerr << "weights must remain external\n";
            return 1;
        }
    }
    if (ids != std::set<std::string>{
            "Qwen3-1.7B", "Qwen3-ASR-0.6B",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen3-VL-2B-Instruct"}) {
        std::cerr << "unexpected model ids\n";
        return 1;
    }

    const auto missing = nlohmann::json::parse(
        Garnet::EnumerateAvailableModelsJson(
            std::filesystem::path(argv[1]) / "does-not-exist"));
    if (!missing["models"].empty() || missing["errors"].size() != 1 ||
        missing["errors"][0].value("code", "") != "catalog_not_found") {
        std::cerr << missing.dump(2) << '\n';
        return 1;
    }

    std::cout << "Garnet model catalog passed: models="
              << catalog["models"].size() << '\n';
    return 0;
}
