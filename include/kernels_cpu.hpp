#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

static_assert(__AVX2__ && __F16C__, "AVX2 and F16C required");

using f16_t = std::uint16_t;

inline float f16_to_f32(f16_t value) { return _cvtsh_ss(value); }
inline f16_t f32_to_f16(float value) { return _cvtss_sh(value, 0); }

inline void copy_f16_to_f32(const f16_t* src, float* dst, size_t count) {
    const size_t simd_width = 8;
    const size_t simd_count = count & ~(simd_width - 1);

    // SIMD loop
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i f16_x8 = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f32_x8 = _mm256_cvtph_ps(f16_x8);
        _mm256_storeu_ps(dst + i, f32_x8);
    }

    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        // Load single f16 value into 128-bit register
        __m128i single_half = _mm_cvtsi32_si128(src[i]);

        // Convert single half to single float
        __m128 single_float = _mm_cvtph_ps(single_half);

        // Extract and store the float value
        dst[i] = _mm_cvtss_f32(single_float);
    }
}

inline void rmsnorm(float* output, const float* x, const float* weight, uint32_t size, float eps) {
    // naive implementation for now
    float rms = 0.0F;
    for (uint32_t i = 0; i < size; ++i) {
        float x_i = x[i];
        rms += x_i * x_i;
    }
    rms = std::sqrt((rms / static_cast<float>(size)) + eps);
    for (uint32_t i = 0; i < size; ++i) {
        output[i] = x[i] / rms * weight[i];
    }
}

inline void matmul(float* out, const float* x /* 1 x n */, const f16_t* w /* n x d */, uint32_t n, uint32_t d) {
    // naive implementation for now
#pragma omp parallel for
    for (uint32_t j = 0; j < d; ++j) {
        float sum = 0.0F;
        for (uint32_t i = 0; i < n; ++i) {
            float w_val = f16_to_f32(w[i + (j * n)]);
            sum += x[i] * w_val;
        }
        out[j] = sum;
    }
}

inline void rope(float* vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta, uint32_t rotary_dim) {
    // naive implementation for now
    for (uint32_t k = 0; k < n_heads; ++k) {
        uint32_t base_idx = k * head_dim;
        for (uint32_t i = 0; i < rotary_dim; i += 2) {
            float freq = 1.F / std::powf(theta, static_cast<float>(i) / static_cast<float>(rotary_dim));
            float deg = static_cast<float>(pos) * freq;

            float cos = std::cosf(deg);
            float sin = std::sinf(deg);

            auto v0 = vec[base_idx + i];
            auto v1 = vec[base_idx + i + 1];

            vec[base_idx + i] = v0 * cos - v1 * sin;
            vec[base_idx + i + 1] = v0 * sin + v1 * cos;
        }
    }
}

inline void rope(f16_t* vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta, uint32_t rotary_dim) {
    // naive implementation for now
    for (uint32_t k = 0; k < n_heads; ++k) {
        uint32_t base_idx = k * head_dim;
        for (uint32_t i = 0; i < rotary_dim; i += 2) {
            float freq = 1.F / std::powf(theta, static_cast<float>(i) / static_cast<float>(rotary_dim));
            float deg = static_cast<float>(pos) * freq;

            float cos = std::cosf(deg);
            float sin = std::sinf(deg);

            float v0 = f16_to_f32(vec[base_idx + i]);
            float v1 = f16_to_f32(vec[base_idx + i + 1]);

            vec[base_idx + i] = f32_to_f16((v0 * cos) - (v1 * sin));
            vec[base_idx + i + 1] = f32_to_f16((v0 * sin) + (v1 * cos));
        }
    }
}

inline void softmax(float* v, uint32_t size) {
    float max_val = -FLT_MAX;
    for (uint32_t i = 0; i < size; ++i) {
        max_val = std::max(v[i], max_val);
    }

    float sum = 0.F;
    for (uint32_t i = 0; i < size; ++i) {
        float ev = std::expf(v[i] - max_val);
        v[i] = ev;
        sum += ev;
    }

    for (uint32_t i = 0; i < size; ++i) {
        v[i] /= sum;
    }
}

inline void attn(float* out, float* attn_scores, const float* q, const f16_t* k_cache, const f16_t* v_cache,
                 uint32_t group_i, uint32_t head_dim, uint32_t n_kv_heads, uint32_t kv_len) {
    const f16_t* k_start = k_cache + static_cast<ptrdiff_t>(head_dim * group_i);
    const f16_t* v_start = v_cache + static_cast<ptrdiff_t>(head_dim * group_i);
    ptrdiff_t kv_stride = static_cast<ptrdiff_t>(head_dim) * n_kv_heads;
    // calculate attention scores
    for (uint32_t t = 0; t < kv_len; ++t) {
        const f16_t* token_k = k_start + (kv_stride * t);
        float score = 0.F;
        for (uint32_t i = 0; i < head_dim; ++i) {
            score += q[i] * f16_to_f32(token_k[i]);
        }
        score /= std::sqrtf(static_cast<float>(head_dim));
        attn_scores[t] = score;
    }

    softmax(attn_scores, kv_len);

    // calculate output value
    for (uint32_t i = 0; i < head_dim; ++i) {
        float v = 0.F;
        for (uint32_t t = 0; t < kv_len; ++t) {
            const f16_t* token_v = v_start + static_cast<ptrdiff_t>(t * kv_stride);
            v += attn_scores[t] * f16_to_f32(token_v[i]);
        }
        out[i] = v;
    }
}

inline void moe_gate(float* active_experts_weights, int* active_experts, const float* moe_weights, uint32_t n_experts,
                     uint32_t n_experts_active) {
    active_experts_weights[0] = 1.F;
    active_experts[0] = 0;
    (void)moe_weights;
    (void)n_experts;
    (void)n_experts_active;
    throw std::runtime_error("unimplemented");
}

inline float gelu(float x) {
    // return x * 0.5F * (1.F + std::erff(x / std::sqrtf(2.F)));
    // We computate approximation instead
    constexpr float sqrt2overpi = std::sqrtf(2.F / M_PI);
    return 0.5F * x * (1.F + std::tanhf(sqrt2overpi * (x + 0.044715F * x * x * x)));
}

inline float silu(float x) { return x / (1.F + std::expf(-x)); }
