// CPU-only policy checks: no HIP/CUDA headers, device queries or allocations.
#include "../src/core/glm_memory.hpp"

#undef NDEBUG   // the checks are asserts: keep them in Release builds
#include <cassert>
#include <cstdio>

static void check_ram_plan() {
    using namespace strata::core::glmfast;
    // The old weighted split gives class 0 excess slots while starving class 1,
    // even though the byte budget exactly covers both mandatory capacities.
    const uint64_t budget = 600;
    const int64_t old0 = (int64_t) ((double) budget * 768 / 1104 / 12);
    const int64_t old1 = (int64_t) ((double) budget * 336 / 1104 / 4);
    assert(old0 > 20 && old1 < 90);
    std::vector<int64_t> slots;
    assert(ram_class_plan(budget, {12, 4}, {20, 90}, {80, 100}, slots));
    assert((slots == std::vector<int64_t>{20, 90}));
    assert(ram_class_plan(900, {12, 4}, {20, 90}, {80, 100}, slots));
    assert(slots[0] >= 20 && slots[1] >= 90);
    assert((uint64_t) slots[0] * 12 + (uint64_t) slots[1] * 4 <= 900);
    assert(ram_class_plan(2000, {12, 4}, {20, 90}, {80, 100}, slots));
    assert((slots == std::vector<int64_t>{80, 100}));
    const auto previous = slots;
    assert(!ram_class_plan(budget - 1, {12, 4}, {20, 90}, {80, 100}, slots));
    assert(slots == previous);  // no partially successful plan

    // The pool tail may all be lent to prefill; the main pool's three promotion
    // spares are not resident expert capacity either (288 - (24 - 3) = 267).
    const int warm = warm_pool_slots(false, 28, 288, 3);
    const int main = warm_pool_slots(false, 24, 288, 3);
    assert(warm == 25 && main == 21);
    assert(288 - main == (288 - warm) + 4);
    assert(ram_class_plan(267 * 16, {16}, {288 - main}, {300}, slots));
    assert(slots[0] == 267);
    assert(!ram_class_plan(266 * 16, {16}, {288 - main}, {300}, slots));

    // Geometry from the Windows Maya-L experiment; this tests the planner, not
    // a claim that these budgets are available on every Windows system.
    const std::vector<size_t> stride{12976128, 11665408};
    const uint64_t measured_budget = 135674292224ull;
    assert(ram_class_plan(measured_budget, stride, {2168, 9214}, {2200, 9298}, slots));
    assert(slots[0] >= 2168 && slots[1] >= 9214);
    assert((uint64_t) slots[0] * stride[0] + (uint64_t) slots[1] * stride[1] <= measured_budget);
    assert(!ram_class_plan(measured_budget, stride, {2184, 9282}, {2216, 9366}, slots));

    // Reject malformed sizes/counts and multiplication overflow before writing
    // output. Test UINT64_MAX even on MSVC, where long double is only double.
    const auto saved = slots;
    assert(!ram_class_plan(100, {0}, {0}, {1}, slots));
    assert(!ram_class_plan(100, {4}, {-1}, {1}, slots));
    assert(!ram_class_plan(100, {4}, {2}, {1}, slots));
    assert(!ram_class_plan(100, {4}, {1}, {}, slots));
    assert(!ram_class_plan(100, {4}, {1}, {(int64_t) std::numeric_limits<int>::max() + 1}, slots));
    assert(!ram_class_plan(std::numeric_limits<size_t>::max(), {std::numeric_limits<size_t>::max()},
                          {2}, {2}, slots));
    assert(slots == saved);
    assert(ram_class_plan(std::numeric_limits<uint64_t>::max(), {1}, {0}, {1}, slots));
    assert(slots[0] == 1);
    assert(ram_class_plan(0, {}, {}, {}, slots) && slots.empty());
    assert(ram_class_plan(0, {4}, {0}, {0}, slots) && slots[0] == 0);

    // Exhaust small integer budgets, including rounding residue and zero slots.
    for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) {
        for (uint64_t bytes = 0; bytes < 256; ++bytes) {
            const uint64_t mandatory = (uint64_t) a * 4 + (uint64_t) b * 9;
            const bool ok = ram_class_plan(bytes, {4, 9}, {a, b}, {a + 3, b + 5}, slots);
            assert(ok == (bytes >= mandatory));
            if (!ok) continue;
            assert(slots[0] >= a && slots[0] <= a + 3);
            assert(slots[1] >= b && slots[1] <= b + 5);
            assert((uint64_t) slots[0] * 4 + (uint64_t) slots[1] * 9 <= bytes);
        }
    }
    assert(!balanced_resident_ram(false, true, false));  // opt-in only
    assert(!balanced_resident_ram(true, false, false));  // non-resident unchanged
    assert(!balanced_resident_ram(true, true, true));    // staging-only unchanged
    assert(balanced_resident_ram(true, true, false));
}

int main() {
    check_ram_plan();
    using strata::core::glmfast::expert_pool_budget;
    using strata::core::glmfast::minimal_ram_tier;
    using strata::core::glmfast::full_unified_pool;
    using strata::core::glmfast::warm_pool_slots;
    constexpr size_t G = size_t{1} << 30;
    constexpr size_t experts = 80 * G + 174 * (size_t{1} << 20);

    // A 128 GiB APU with 110 GiB available after setup: the 112 GiB runtime
    // aperture (and even a tiny reported free value) cannot change the OS budget.
    assert(expert_pool_budget(true, 112 * G, 110 * G, G, 16 * G, experts) == experts);
    assert(expert_pool_budget(true, G, 110 * G, G, 16 * G, experts) == experts);
    // Memory pressure leaves 53 GiB; missing MemAvailable or less than headroom
    // leaves zero rather than wrapping or trusting HIP's much larger free figure.
    assert(expert_pool_budget(true, 112 * G, 70 * G, G, 16 * G, experts) == 53 * G);
    assert(expert_pool_budget(true, 112 * G, 0, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 16 * G, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 17 * G, G, 16 * G, experts) == 0);
    // User headroom/reserve and the discrete formula retain their meaning.
    assert(expert_pool_budget(true, 112 * G, 70 * G, 2 * G, 8 * G, experts) == 60 * G);
    assert(expert_pool_budget(false, 12 * G, 110 * G, 3 * G, 16 * G, experts) == 9 * G);
    assert(expert_pool_budget(false, G, 110 * G, 3 * G, 16 * G, experts) == 0);
    assert(minimal_ram_tier(true, true, false));
    assert(!minimal_ram_tier(true, true, true));  // STRATA_GLM_RAM_GB / supplied budget wins
    assert(!minimal_ram_tier(true, false, false));
    assert(!minimal_ram_tier(false, true, false));
    // Warm every expert when it fits, and never evict residents to make spares.
    // Smaller APU pools and all discrete pools retain the existing spare quota.
    assert(warm_pool_slots(true, 288, 288, 3) == 288);
    assert(full_unified_pool(true, 288, 288));
    assert(warm_pool_slots(true, 192, 288, 3) == 189);
    assert(!full_unified_pool(true, 192, 288));
    assert(warm_pool_slots(false, 288, 288, 3) == 285);
    assert(!full_unified_pool(false, 288, 288));
    std::puts("GLM memory policy checks passed (CPU only)");
}
