#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::core::glmfast {

// All inputs are bytes. On an APU the GPU pool and pinned host allocations spend
// the same physical memory; the runtime's GTT/carve-out size is not free RAM.
inline size_t expert_pool_budget(bool unified, size_t device_free, size_t ram_available,
                                size_t reserve, size_t ram_headroom, size_t expert_bytes) {
    if (!unified) return device_free > reserve ? device_free - reserve : 0;
    const size_t room = ram_available > ram_headroom ? ram_available - ram_headroom : 0;
    return std::min(expert_bytes, room > reserve ? room - reserve : 0);
}

inline bool minimal_ram_tier(bool unified, bool all_experts_fit, bool explicit_budget) {
    return unified && all_experts_fit && !explicit_budget;
}

inline bool full_unified_pool(bool unified, int slots, int experts) {
    return unified && slots >= experts;
}

inline int warm_pool_slots(bool unified, int slots, int experts, int spares) {
    return full_unified_pool(unified, slots, experts) ? experts : std::max(0, slots - spares);
}

// Ordinary tiers and staging-only APU tiers retain their existing sizing policy.
inline bool balanced_resident_ram(bool requested, bool resident, bool staging_only) {
    return requested && resident && !staging_only;
}

// Reserve each class's mandatory slots before spending anything on optional slack.
// Failure leaves slots untouched. Counts must fit RamClass::n (int), and every
// multiplication is bounded by the remaining byte budget before it is performed.
inline bool ram_class_plan(uint64_t budget, const std::vector<size_t>& stride,
                           const std::vector<int64_t>& required, const std::vector<int64_t>& preferred,
                           std::vector<int64_t>& slots) {
    if (stride.size() != required.size() || stride.size() != preferred.size()) return false;
    std::vector<int64_t> plan(required);
    uint64_t left = budget;
    long double optional_bytes = 0;
    for (size_t c = 0; c < stride.size(); ++c) {
        if (!stride[c] || required[c] < 0 || preferred[c] < required[c] ||
            preferred[c] > std::numeric_limits<int>::max()) return false;
        if ((uint64_t) required[c] > left / stride[c]) return false;
        left -= (uint64_t) required[c] * stride[c];
        optional_bytes += (long double) (preferred[c] - required[c]) * stride[c];
    }
    const uint64_t extra_budget = left;
    if (optional_bytes > 0) {
        for (size_t c = 0; c < stride.size(); ++c) {
            const int64_t room = preferred[c] - required[c];
            const long double share = (long double) extra_budget * room / optional_bytes;
            const uint64_t extra = std::min<uint64_t>((uint64_t) std::min<long double>(room, share),
                                                       left / stride[c]);
            plan[c] += (int64_t) extra;
            left -= extra * stride[c];
        }
    }
    // Spend rounding residue without violating any cap or the byte budget.
    for (size_t c = 0; c < stride.size(); ++c) {
        const uint64_t extra = std::min<uint64_t>((uint64_t) (preferred[c] - plan[c]), left / stride[c]);
        plan[c] += (int64_t) extra;
        left -= extra * stride[c];
    }
    slots.swap(plan);
    return true;
}

}  // namespace strata::core::glmfast
