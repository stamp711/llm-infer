#include "model/device.hpp"

std::unique_ptr<CUDAContext> CUDAContext::instance_ptr = nullptr;