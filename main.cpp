#include <cstdint>
#include <cstdlib>

uint64_t alloc(uint64_t size, uint64_t alignemnt = 4) {
    auto addr = reinterpret_cast<uint64_t>(malloc(size + alignemnt - 1));
    uint64_t mask = ~(alignemnt - 1);
    addr = (addr + alignemnt - 1) & mask;
    return addr;
}

// free(addr)
//
// metadata (8 byte) | data
