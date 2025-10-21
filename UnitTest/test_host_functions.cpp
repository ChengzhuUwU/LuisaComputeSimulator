#include "Utils/cpu_parallel.h"
#include <iostream>
#include <atomic>
#include <luisa/core/basic_types.h>

template <typename T>
struct spin_atomic
{
    // static_assert(sizeof(T) % 4 == 0, "spin_atomic only supports types with size multiple of 4 bytes.");
    // constexpr static size_t N = sizeof(T) / 4;

    using AtomicView = std::atomic<T>;
    using MemoryView = T;

    AtomicView bits;
    spin_atomic<T>()
        : bits({0})
    {
    }
    spin_atomic<T>(const T& f) { store(f); }

    void store(const T& f, std::memory_order order = std::memory_order_seq_cst)
    {
        MemoryView i;
        std::memcpy(&i, &f, sizeof(T));
        bits.store(i);
    }

    T load(std::memory_order order = std::memory_order_seq_cst) const
    {
        MemoryView i = bits.load(order);
        T          f;
        std::memcpy(&f, &i, sizeof(T));
        return f;
    }

    T fetch_add(const T& arg, std::memory_order order = std::memory_order_seq_cst)
    {
        MemoryView old_bits = bits.load(order);
        while (true)
        {
            T old_val;
            std::memcpy(&old_val, &old_bits, sizeof(T));
            T new_val = old_val + arg;

            MemoryView new_bits;
            std::memcpy(&new_bits, &new_val, sizeof(T));

            if (bits.compare_exchange_weak(old_bits, new_bits, order))
                return old_val;
        }
    }
};

using atomic_float  = spin_atomic<float>;
using atomic_float3 = spin_atomic<luisa::float3>;

int main()
{

    luisa::fiber::scheduler scheduler;

    atomic_float af(0.0f);
    auto         fn_test_atomic_add = [](atomic_float* af)
    {
        CpuParallel::parallel_for(0, 100000, [&](const uint i) { af->fetch_add(1.0f); });
        printf("Result = %f\n", af->load());
    };

    for (int i = 0; i < 10; i++)
    {
        fn_test_atomic_add(&af);
        af.store(0.0f);
    }

    auto fn_test_atomic_float3_add = [](std::vector<luisa::float3>& af)
    {
        CpuParallel::parallel_for(0,
                                  100000,
                                  [&](const uint i)
                                  {
                                      const uint target_index   = i % 10;
                                      auto       af_atomic_view = (atomic_float3*)(&af[target_index]);
                                      af_atomic_view->fetch_add(luisa::make_float3(1, 2, 3));
                                      //   auto       af_atomic_view = (atomic_float*)(&af[target_index]);
                                      //   af_atomic_view[0].fetch_add(1.0f);
                                      //   af_atomic_view[1].fetch_add(2.0f);
                                      //   af_atomic_view[2].fetch_add(3.0f);
                                  });
    };
    std::vector<luisa::float3> af3_array(10, luisa::make_float3(0.0f));

    for (int i = 0; i < 10; i++)
    {
        fn_test_atomic_float3_add(af3_array);
        for (int j = 0; j < 10; j++)
        {
            printf("Result in iter %d: [%d] = (%f, %f, %f)\n",
                   i,
                   j,
                   af3_array[j].x,
                   af3_array[j].y,
                   af3_array[j].z);
        }
        for (auto& val : af3_array)
            val = luisa::make_float3(0.0f);
    }
}