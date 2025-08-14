#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void printCudaInfo() {
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Getting device count");
    
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, i), "Getting device properties");
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  SM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    }
}

class LLMInference {
public:
    LLMInference() {
        std::cout << "Initializing LLM Inference Engine..." << std::endl;
        checkCudaError(cudaSetDevice(0), "Setting CUDA device");
    }
    
    ~LLMInference() {
        cleanup();
    }
    
    void run() {
        std::cout << "\nRunning LLM Inference..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
    }
    
private:
    void cleanup() {
        cudaDeviceReset();
    }
};

int main() {
    std::cout << "LLM Inference Engine v0.1" << std::endl;
    std::cout << "=========================" << std::endl;
    
    printCudaInfo();
    
    try {
        LLMInference engine;
        engine.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}