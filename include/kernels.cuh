#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

__global__ void copy_embedding_float(float* out, const float* token_embedding_table, uint32_t token, uint32_t dim);
__global__ void copy_embedding_half(float* out, const half* token_embedding_table, uint32_t token, uint32_t dim);

__global__ void rmsnorm(float* out, const float* x, const float* weight, uint32_t size, float eps);

__global__ void fused_gemv_clip(float* out, const float* x /* 1xn */, const half* w /* nxd */, uint32_t n, uint32_t d,
                                std::optional<float> clip);
__global__ void fused_gemv_clip_f2h(half* out, const float* x /* 1xn */, const half* w /* nxd */, uint32_t n,
                                    uint32_t d, std::optional<float> clip);

__global__ void rope_f2f(float* out, const float* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta,
                         uint32_t rotary_dim);
__global__ void rope_f2h(half* out, const float* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta,
                         uint32_t rotary_dim);
__global__ void rope_h2h(half* out, const half* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta,
                         uint32_t rotary_dim);

__global__ void attention(float* o, const float* q, const half* k, const half* v, uint32_t kv_tokens);
