#include "Utils/cpu_parallel.h"
#include <iostream>
#include <atomic>

struct atomic_float
{
    std::atomic<uint32_t> bits;

    atomic_float()
        : bits(0)
    {
    }
    atomic_float(float f) { store(f); }

    void store(float f, std::memory_order order = std::memory_order_seq_cst)
    {
        uint32_t i;
        std::memcpy(&i, &f, sizeof(float));
        bits.store(i, order);
    }

    float load(std::memory_order order = std::memory_order_seq_cst) const
    {
        uint32_t i = bits.load(order);
        float    f;
        std::memcpy(&f, &i, sizeof(float));
        return f;
    }

    float fetch_add(float arg, std::memory_order order = std::memory_order_seq_cst)
    {
        uint32_t old_bits = bits.load(order);
        while (true)
        {
            float old_val;
            std::memcpy(&old_val, &old_bits, sizeof(float));
            float new_val = old_val + arg;

            uint32_t new_bits;
            std::memcpy(&new_bits, &new_val, sizeof(float));

            if (bits.compare_exchange_weak(old_bits, new_bits, order))
                return old_val;
        }
    }
};

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
                                      auto       af_atomic_view = (atomic_float*)(&af[target_index]);
                                      af_atomic_view[0].fetch_add(1.0f);
                                      af_atomic_view[1].fetch_add(2.0f);
                                      af_atomic_view[2].fetch_add(3.0f);
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