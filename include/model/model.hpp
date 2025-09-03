#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "block.hpp"
#include "config.hpp"
#include "device.hpp"
#include "gguf.hpp"
#include "model/inference_state.hpp"
#include "model/tensor.hpp"

class Model {
   public:
    Model(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = delete;

    explicit Model(const GGUF& gguf, DeviceType device_type);
    ~Model();

    [[nodiscard]] const ModelConfig& config() const { return *config_; }
    [[nodiscard]] DeviceType device_type() const { return device_type_; }

    void forward(InferenceState& s, std::uint32_t token, std::uint32_t pos, InferenceMode mode = InferenceMode::Decode);

   private:
    // CPU
    void forward_cpu_(InferenceState& s, std::uint32_t token, std::uint32_t pos, InferenceMode mode);
    void copy_embedding_(InferenceState& s, std::uint32_t token);

    // CUDA
    void forward_cuda_(InferenceState& s, uint32_t token, uint32_t pos, InferenceMode mode);
    void forward_cuda_build_or_update_graph_(InferenceState& s, uint32_t token, uint32_t pos, InferenceMode mode);

    DeviceType device_type_;

    // Model configuration
    std::unique_ptr<ModelConfig> config_;

    // Model components
    std::vector<std::unique_ptr<Block>> blocks_;

    // When the tensor has dimensions [x, y, z], the data is laid out in memory such that the innermost
    // (fastest-changing) index corresponds to the first dimension (x), followed by y, then z.

    // Global tensors
    Tensor<const void> token_embedding_table_;  // [dim, vocab_size] - look up table
    Tensor<const void> rms_final_weight_;       // [dim]
    Tensor<const void> wcls_;                   // [dim, vocab_size] - classifier weights
};
