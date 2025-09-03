#include "model/device.hpp"

#include <cuda_runtime_api.h>

std::unique_ptr<CUDAContext> CUDAContext::instance_ptr = nullptr;
