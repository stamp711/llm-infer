#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <fmt/base.h>
#include <fmt/format.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

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

inline void check_cuda(cudaError_t error, std::string_view msg = {},
                       const std::source_location &loc = std::source_location::current()) {
    if (error != cudaSuccess) {
        const auto *cuda_msg = cudaGetErrorString(error);
        auto error_msg = msg.empty()
                             ? fmt::format("CUDA error at {}:{} - {}", loc.file_name(), loc.line(), cuda_msg)
                             : fmt::format("[{}] CUDA error at {}:{} - {}", msg, loc.file_name(), loc.line(), cuda_msg);
        fmt::print(stderr, "{}\n", error_msg);
        throw std::runtime_error(error_msg);
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

    static cudaStream_t create_cuda_stream() {
        cudaStream_t stream;  // NOLINT
        check_cuda(cudaStreamCreate(&stream));
        return stream;  // travially copyable
    }

    static void synchronize_stream(cudaStream_t stream) { check_cuda(cudaStreamSynchronize(stream)); }

    static void check_last_error() {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            const auto *cuda_msg = cudaGetErrorString(error);
            throw std::runtime_error(fmt::format("CUDA error: {}", cuda_msg));
        }
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

    static std::unique_ptr<CUDAContext> instance_ptr;

    int warp_size_;
    int max_threads_per_block_;
};

template <typename... Args>
struct KernelArgs {
    std::tuple<Args...> args;

    KernelArgs(Args... a) : args(a...) {}
};

struct KernelDeps {
    std::vector<std::string> deps;

    template <typename... Args>
    KernelDeps(Args... a) : deps{std::string(a)...} {}

    KernelDeps(const std::vector<std::string> &d) : deps(d) {}

    KernelDeps() = default;
};

struct CudaGraph {
    cudaGraph_t graph{};
    std::optional<cudaGraphExec_t> instance;

    std::unordered_map<std::string, std::pair<cudaGraphNode_t, cudaKernelNodeParams>> nodes;

    CudaGraph() { check_cuda(cudaGraphCreate(&graph, 0)); }

    void instantiate() {
        if (instance.has_value()) {
            throw std::runtime_error("Graph already instantiated");
        }
        cudaGraphExec_t exec_graph{};
        check_cuda(cudaGraphInstantiate(&exec_graph, graph));
        instance = exec_graph;
    }

    void launch(cudaStream_t stream) {
        if (!instance.has_value()) {
            instantiate();
        }
        check_cuda(cudaGraphLaunch(instance.value(), stream));
    }

    template <typename... Args>
    void add_kernel_node(std::string key, const KernelDeps &dependencies, cudaKernelNodeParams params,
                         KernelArgs<Args...> kernel_args) {
        if (instance.has_value()) {
            throw std::runtime_error("Cannot add nodes after graph instantiation");
        }

        fmt::println("adding kernel {}", key);
        if (nodes.find(key) != nodes.end()) {
            throw std::runtime_error(fmt::format("Node with key {} already exists", key));
        }

        auto args_array =
            std::apply([](auto &...args) { return std::array<void *, sizeof...(Args)>{&args...}; }, kernel_args.args);
        params.kernelParams = args_array.data();

        cudaGraphNode_t node{};
        if (dependencies.deps.empty()) {
            check_cuda(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
        } else {
            std::vector<cudaGraphNode_t> dep_nodes;
            for (const auto &dep : dependencies.deps) {
                auto it = nodes.find(dep);
                if (it == nodes.end()) {
                    throw std::runtime_error(fmt::format("Dependency node {} not found", dep));
                }
                dep_nodes.push_back(it->second.first);
            }
            check_cuda(cudaGraphAddKernelNode(&node, graph, dep_nodes.data(), dep_nodes.size(), &params));
        }

        nodes.emplace(key, std::make_pair(node, params));
    }

    template <typename... Args>
    void update_kernel_args(std::string key, KernelArgs<Args...> kernel_args) {
        if (!instance.has_value()) {
            throw std::runtime_error("Cannot update args before graph instantiation");
        }

        auto it = nodes.find(key);
        if (it == nodes.end()) {
            throw std::runtime_error(fmt::format("Node with key {} not found", key));
        }

        auto args_array =
            std::apply([](auto &...args) { return std::array<void *, sizeof...(Args)>{&args...}; }, kernel_args.args);

        auto &cached_params = it->second.second;
        cached_params.kernelParams = args_array.data();

        check_cuda(cudaGraphExecKernelNodeSetParams(instance.value(), it->second.first, &cached_params));
    }

    template <typename... Args>
    void add_or_update_kernel_node(std::string key, const KernelDeps &dependencies, cudaKernelNodeParams params,
                                   KernelArgs<Args...> kernel_args) {
        if (instance.has_value()) {
            update_kernel_args(key, kernel_args);
        } else {
            add_kernel_node(key, dependencies, params, kernel_args);
        }
    }

    template <typename... Args>
    void add_or_update_kernel_node(std::string key, cudaKernelNodeParams params, KernelArgs<Args...> kernel_args) {
        if (instance.has_value()) {
            update_kernel_args(key, kernel_args);
        } else {
            std::vector<std::string> all_deps;
            all_deps.reserve(nodes.size());
            for (const auto &[node_key, _] : nodes) {
                all_deps.push_back(node_key);
            }
            add_kernel_node(key, KernelDeps{all_deps}, params, kernel_args);
        }
    }
};
