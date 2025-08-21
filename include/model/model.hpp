#pragma once

#include <memory>
#include <vector>

#include "block.hpp"
#include "config.hpp"

struct InferenceState {
    float* x = nullptr;  // (dim) for a single new token
    float* q = nullptr;  // n_heads * (head_dim)
    float* k = nullptr;  // n_kv_heads * (head_dim)
    float* v = nullptr;  // n_kv_heads * (head_dim)
};

class Model {
   public:
    Model();

   private:
    DeviceType device_type_;
    std::unique_ptr<ModelConfig> model_config_;
    std::vector<Block> blocks;
};
