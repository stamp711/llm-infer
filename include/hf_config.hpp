#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

/// HuggingFace model configuration parser
/// Provides direct access to the configuration JSON
class HFConfig {
   public:
    /// Load configuration from config.json file
    explicit HFConfig(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }

        try {
            file >> config_;
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse config JSON: " + std::string(e.what()));
        }
    }

    /// Direct access to the configuration JSON
    [[nodiscard]] const nlohmann::json& config() const noexcept { return config_; }

   private:
    nlohmann::json config_;
};
