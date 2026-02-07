#include "Utils/cpu_parallel.h"
#include "luisa/core/fiber.h"
#include "luisa/core/spin_mutex.h"
#include <cstdint>
#include <iostream>
#include <atomic>
#include <luisa/core/basic_types.h>
#include <mutex>
#include <vector>

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
    spin_atomic(const spin_atomic& other) { store(other.load()); }
    spin_atomic& operator=(const spin_atomic& other)
    {
        if (this != &other)
        {
            store(other.load());
        }
        return *this;
    }

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
        CpuParallel::parallel_for(0, 100000, [&](const uint32_t i) { af->fetch_add(1.0f); });
        printf("Result = %f\n", af->load());
    };

    for (int i = 0; i < 10; i++)
    {
        fn_test_atomic_add(&af);
        af.store(0.0f);
    }

    auto fn_test_atomic_float3_add = [](const std::span<luisa::float3>& af, const std::span<luisa::spin_mutex>& mtx_array)
    {
        CpuParallel::parallel_for(0,
                                  100000,
                                  [&](const uint32_t i)
                                  {
                                      const uint32_t target_index = i % 10;
                                      {
                                          mtx_array[target_index].lock();
                                          af[target_index] += luisa::make_float3(1, 2, 3);
                                          mtx_array[target_index].unlock();
                                      }
                                  });
    };
    std::vector<luisa::float3> f3_array(10, luisa::make_float3(0.0f));
    // std::vector<atomic_float3> af3_array(10, atomic_float3(luisa::make_float3(0.0f)));
    // auto af3_array = std::span(reinterpret_cast<atomic_float3*>(f3_array.data()), f3_array.size());

    constexpr size_t stride_mutex  = sizeof(luisa::spin_mutex);
    constexpr size_t stride_mutex2 = sizeof(std::atomic_flag);
    constexpr size_t stride_mutex3 = sizeof(uint32_t);

    std::vector<uint32_t> mtx_array(10);
    auto mtx_array2 = std::span(reinterpret_cast<luisa::spin_mutex*>(mtx_array.data()), mtx_array.size());

    for (int i = 0; i < 10; i++)
    {
        fn_test_atomic_float3_add(std::span(f3_array), mtx_array2);
        for (int j = 0; j < 10; j++)
        {
            auto v = f3_array[j];
            printf("Result in iter %d: [%d] = (%f, %f, %f)\n", i, j, v.x, v.y, v.z);
        }
        for (auto& val : f3_array)
            val = luisa::make_float3(0.0f);
    }
}