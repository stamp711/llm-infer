#include "model/model.hpp"

#include "model/config.hpp"
#include "model/device.hpp"

Model::~Model() {
    if (device_type == DeviceType::CUDA) {
        CUDAContext::synchronize();  // TODO: move this to the outside. Destructors are noexcept by default.
    }
}

Model::Model(const GGUF& gguf, DeviceType device_type) : device_type(device_type) {
    config = std::make_unique<ModelConfig>(gguf);

    if (device_type == DeviceType::CUDA) {
        CUDAContext::initialize();  // TODO: move this to the outside
    }

    // Load global tensors with shape and quantization checking
    std::string token_embd_name = "token_embd.weight";
    const auto& token_embd = get_tensor_required(gguf, token_embd_name);
    check_tensor(token_embd, {config->dim, config->vocab_size}, config->weight_quantization);
    token_embedding_table = tensor_from_gguf<void>(token_embd, device_type);

    std::string output_norm_name = "output_norm.weight";
    const auto& output_norm = get_tensor_required(gguf, output_norm_name);
    check_tensor(output_norm, {config->dim}, config->norms_weight_quantization);
    rms_final_weight = tensor_from_gguf<void>(output_norm, device_type);

    std::string output_name = "output.weight";
    const auto& output = get_tensor_required(gguf, output_name);
    check_tensor(output, {config->dim, config->vocab_size}, config->weight_quantization);
    wcls = tensor_from_gguf<void>(output, device_type);

    // Create blocks
    blocks.reserve(config->n_layers);
    for (std::uint32_t i = 0; i < config->n_layers; ++i) {
        // Create block (following yalm pattern but adapted for GGUF TensorInfo&)
        blocks.emplace_back(std::make_unique<Block>(gguf, device_type, *config, i));
    }
}
