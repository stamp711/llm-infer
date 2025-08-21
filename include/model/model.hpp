#pragma once

#include <memory>
#include <vector>

#include "block.hpp"
#include "config.hpp"
#include "device.hpp"
#include "gguf.hpp"
#include "model/tensor.hpp"

class Model {
   public:
    Model(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

    explicit Model(const GGUF& gguf, DeviceType device_type);
    ~Model();

    DeviceType device_type;

    // Model configuration
    std::unique_ptr<ModelConfig> config;

    // Model components
    std::vector<std::unique_ptr<Block>> blocks;

    // When the tensor has dimensions [x, y, z], the data is laid out in memory such that the innermost
    // (fastest-changing) index corresponds to the first dimension (x), followed by y, then z.

    // Global tensors
    Tensor<const void> token_embedding_table;  // [dim, vocab_size] - look up table
    Tensor<const void> rms_final_weight;       // [dim]
    Tensor<const void> wcls;                   // [dim, vocab_size] - classifier weights
};
