#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <string_view>

enum class DeviceType : std::uint8_t { CPU, CPU_UnAligned, CUDA };

constexpr size_t CPU_ALIGN = 32;

template <std::size_t N>
inline bool is_aligned(const void *ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % N == 0;
}

inline void check_cpu_alignment(const void *ptr) {
    if (!is_aligned<CPU_ALIGN>(ptr)) {
        throw std::runtime_error("Pointer is not aligned for CPU");
    }
}

class CUDAContext {
   public:
    CUDAContext(const CUDAContext &) = delete;
    CUDAContext(CUDAContext &&) = delete;
    CUDAContext &operator=(const CUDAContext &) = delete;
    CUDAContext &operator=(CUDAContext &&) = delete;
    ~CUDAContext() { cudaDeviceReset(); }

    // ========== singleton methods ==========
    static void initialize(int device = 0) {
        static bool initialized = false;
        if (initialized) {
            throw std::runtime_error("CUDAContext already initialized");
        }
        instance_ptr = std::unique_ptr<CUDAContext>(new CUDAContext(device));
        initialized = true;
    }

    static CUDAContext &get() {
        if (!instance_ptr) {
            throw std::runtime_error("CUDAContext not initialized. Call initialize() first.");
        }
        return *instance_ptr;
    }

    static void cleanup() {
        if (instance_ptr) {
            instance_ptr.reset();
        }
    }

    // ========== device properties ==========
    [[nodiscard]] int warp_size() const { return warp_size_; }
    [[nodiscard]] int max_threads_per_block() const { return max_threads_per_block_; }

    // ========== synchronization ==========
    static void synchronize() { check_cuda(cudaDeviceSynchronize()); }

    // ========== memory management ==========

    static void *malloc(size_t size) {
        void *ptr = nullptr;
        check_cuda(cudaMalloc(&ptr, size));
        return ptr;
    }

    static void *copy_to_device(const void *host, size_t size) {
        void *device_ptr = nullptr;
        check_cuda(cudaMalloc(&device_ptr, size));
        check_cuda(cudaMemcpy(device_ptr, host, size, cudaMemcpyHostToDevice));
        return device_ptr;
    }

    static void *copy_to_device_async(const void *host, size_t size, cudaStream_t stream = nullptr) {
        void *device_ptr = nullptr;
        check_cuda(cudaMalloc(&device_ptr, size));
        check_cuda(cudaMemcpyAsync(device_ptr, host, size, cudaMemcpyHostToDevice, stream));
        return device_ptr;
    }

    template <typename T>
    static void free(T *device_ptr) {
        check_cuda(cudaFree(reinterpret_cast<void *>(device_ptr)));
    }

   private:
    CUDAContext(int device) {
        check_cuda(cudaSetDevice(device));

        int value = 0;
        check_cuda(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, device));
        warp_size_ = value;
        check_cuda(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device));
        max_threads_per_block_ = value;
    }

    static void check_cuda(cudaError_t error, std::string_view msg = {},
                           const std::source_location &loc = std::source_location::current()) {
        if (error != cudaSuccess) {
            const auto *cuda_msg = cudaGetErrorString(error);
            auto error_msg =
                msg.empty() ? fmt::format("CUDA error at {}:{} - {}", loc.file_name(), loc.line(), cuda_msg)
                            : fmt::format("[{}] CUDA error at {}:{} - {}", msg, loc.file_name(), loc.line(), cuda_msg);
            fmt::print(stderr, "{}\n", error_msg);
            throw std::runtime_error(error_msg);
        }
    }

    static std::unique_ptr<CUDAContext> instance_ptr;

    int warp_size_;
    int max_threads_per_block_;
};
