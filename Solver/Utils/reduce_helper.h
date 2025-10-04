#pragma once

#include <luisa/luisa-compute.h>
#include "Core/float_n.h"
#include "luisa/dsl/builtin.h"
#include "luisa/dsl/shared.h"
#include "luisa/runtime/device.h"

// #define reduce_with_cache(thread_value, tid, wid, dim, type, set_op, reduce_op, get_op)  \
//         luisa::compute::Shared<type> cache_aabb(dim);                                   \
//         set_op(cache_aabb[tid], thread_value)                                           \
//             luisa::compute::sync_block();                                               \
//             luisa::compute::Uint s = dim >> 1;                                          \
//             $while(true) {                                                               \
//                 $if (s == 0) { $break; };                                               \
//                 $if(tid < s) {                                                          \
//                     reduce_op(cache_aabb[tid], cache_aabb[tid + s]);                    \
//                 };                                                                      \
//                 luisa::compute::sync_block();                                           \
//                 s >>= 1;                                                                \
//             };                                                                          \
//             $if (wid == 0) {                                                            \
//                 get_op(thread_value, cache_aabb[0]);                                    \
//             };


namespace lcs
{

namespace ParallelIntrinsic
{

    constexpr uint reduce_block_dim = 256;

    template <typename T>
    inline void default_reduce_op_unary(Var<T>& left, const Var<T>& right)
    {
        left = left + right;
    }

    // !Warning: Cache-based reduce relies on num-threads dispatched is devidable to 256
    template <typename T, typename ReduceOp>
    inline Var<T> block_reduce(const luisa::compute::UInt& vid,
                               const Var<T>&               thread_value,
                               const ReduceOp              reduce_op_unary = default_reduce_op_unary<T>)
    {
        using Uint = luisa::compute::UInt;
        luisa::compute::set_block_size(reduce_block_dim);
        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;

        luisa::compute::Shared<T> cache(reduce_block_dim);
        cache[threadIdx]   = thread_value;
        Var<T> block_value = thread_value;
        luisa::compute::sync_block();

        luisa::compute::UInt s = reduce_block_dim >> 1;
        $while(true)
        {
            $if(threadIdx < s)
            {
                reduce_op_unary(cache[threadIdx], cache[threadIdx + s]);
            };
            luisa::compute::sync_block();
            s >>= 1;
            $if(s == 0)
            {
                $break;
            };
        };
        $if(threadIdx == 0)
        {
            block_value = cache[0];
        };
        return block_value;
    }


    template <typename T>
    inline Var<T> warp_reduce_op_sum(const Var<T>& lane_value)
    {
        return luisa::compute::warp_active_sum(lane_value);
    };
    template <typename T>
    inline Var<T> warp_reduce_op_min(const Var<T>& lane_value)
    {
        return luisa::compute::warp_active_min(lane_value);
    };
    template <typename T>
    inline Var<T> warp_reduce_op_max(const Var<T>& lane_value)
    {
        return luisa::compute::warp_active_max(lane_value);
    };

    template <typename T>
    inline Var<T> warp_scan_op(const Var<T>& lane_value)
    {
        return luisa::compute::warp_prefix_sum(lane_value);
    };

    constexpr uint warp_dim = 32;
    constexpr uint warp_num = 32;

    // !!!Only the first warp will get block reduce result
    template <typename T, typename ReduceOp>
    inline Var<T> block_intrinsic_reduce(const luisa::compute::UInt& vid, const Var<T>& thread_value, const ReduceOp warp_reduce_op_binary)
    {
        using Uint = luisa::compute::UInt;
        luisa::compute::set_block_size(reduce_block_dim);

        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;
        const luisa::compute::UInt warpIdx   = threadIdx / warp_dim;
        const luisa::compute::UInt laneIdx   = threadIdx % warp_dim;

        Var<T> block_value = thread_value;
        block_value        = warp_reduce_op_binary(block_value);  // warp reduced value

        luisa::compute::Shared<uint> cache_active_warp_count(1);
        luisa::compute::Shared<T>    cache(warp_num);

        $if(threadIdx == 0)
        {
            cache_active_warp_count[0] = 0;
        };
        luisa::compute::sync_block();

        // $if (warpIdx == 0) { cache[threadIdx] = zero_value; };
        // $if (warpIdx == 0) { cache[threadIdx] = luisa::compute::warp_read_first_active_lane(thread_value); };
        // luisa::compute::sync_block();
        $if(laneIdx == 0)
        {
            cache[warpIdx] = block_value;
            cache_active_warp_count.atomic(0).fetch_add(1u);
        };
        luisa::compute::sync_block();
        $if(threadIdx < cache_active_warp_count[0])
        {
            block_value = warp_reduce_op_binary(cache[threadIdx]);
        };
        return block_value;
    }

    template <typename T>
    inline Var<T> block_intrinsic_scan_exclusive(const luisa::compute::UInt& vid,
                                                 const Var<T>&               thread_value,
                                                 Var<T>&                     output_block_sum)
    {
        using Uint = luisa::compute::UInt;
        luisa::compute::set_block_size(reduce_block_dim);

        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;
        const luisa::compute::UInt warpIdx   = threadIdx / warp_dim;
        const luisa::compute::UInt laneIdx   = threadIdx % warp_dim;

        Var<T> warp_prefix = warp_scan_op<T>(thread_value);

        luisa::compute::Shared<uint> cache_active_warp_count(1);
        luisa::compute::Shared<T>    cache_block_sum(1);
        luisa::compute::Shared<T>    cache(warp_num);

        $if(threadIdx == 0)
        {
            cache_active_warp_count[0] = 0;
        };
        luisa::compute::sync_block();

        $if(laneIdx == luisa::compute::warp_active_count_bits(true) - 1)  // Is the last active thread in the warp
        {
            cache[warpIdx] = warp_prefix + thread_value;
            cache_active_warp_count.atomic(0).fetch_add(1u);
        };
        luisa::compute::sync_block();

        $if(threadIdx < cache_active_warp_count[0])
        {
            Var<T> warp_val  = cache[threadIdx];
            cache[threadIdx] = warp_scan_op<T>(warp_val);  // Get warp's prefix in block
            $if(threadIdx == cache_active_warp_count[0] - 1)
            {
                cache_block_sum[0] = cache[threadIdx] + warp_val;
            };
        };
        luisa::compute::sync_block();
        output_block_sum = cache_block_sum[0];
        return cache[warpIdx] + warp_prefix;
    }

    namespace sort_detail
    {
        template <typename T>
        inline void block_intrinsic_scan_exclusive(const luisa::compute::UInt& vid,
                                                   const Var<T>&               thread_value,
                                                   luisa::compute::Shared<T>&  cache_warp_sum,
                                                   luisa::compute::Shared<T>&  cache_output_prefix)
        {
            using Uint = luisa::compute::UInt;

            const luisa::compute::UInt threadIdx = vid % reduce_block_dim;
            const luisa::compute::UInt warpIdx   = threadIdx / warp_dim;
            const luisa::compute::UInt laneIdx   = threadIdx % warp_dim;

            Var<T> warp_offset = warp_scan_op<T>(thread_value);
            $if(laneIdx == 31)
            {
                cache_warp_sum[warpIdx] = warp_offset + thread_value;
            };
            luisa::compute::sync_block();

            $if(warpIdx == 0)
            {
                Var<T> warp_val           = cache_warp_sum[threadIdx];
                cache_warp_sum[threadIdx] = warp_scan_op<T>(warp_val);  // Get warp's prefix in block
            };
            luisa::compute::sync_block();
            cache_output_prefix[threadIdx] = cache_warp_sum[warpIdx] + warp_offset;
            luisa::compute::sync_block();
        }

        template <typename T>
        inline void swap_template(Var<T>& a, Var<T>& b)
        {
            Var<T> tmp = a;
            a          = b;
            b          = tmp;
        };
    }  // namespace sort_detail

    template <typename T>
    inline T block_radix_sort_8bit(const luisa::compute::UInt& vid, const Var<T>& thread_value)
    {
        using Uint = luisa::compute::UInt;

        luisa::compute::set_block_size(reduce_block_dim);
        const Uint threadIdx = vid % reduce_block_dim;
        const Uint warpIdx   = threadIdx / warp_dim;

        constexpr uint bit_size   = 8;
        constexpr uint num_pass   = sizeof(T) * 8 / bit_size;
        constexpr uint num_bucket = 1 << bit_size;

        luisa::compute::Shared<T>      cache_value(reduce_block_dim);
        luisa::compute::Shared<ushort> cache_index(reduce_block_dim);    // Sorted get original index
        luisa::compute::Shared<uint>   cache_per_bit_count(num_bucket);  // 2560 KB
        luisa::compute::Shared<uint>   cache_scan_warp_sum(warp_num);

        cache_value[threadIdx] = thread_value;
        cache_index[threadIdx] = threadIdx;
        luisa::compute::sync_block();

        for (uint pass = 0; pass < num_pass; pass++)
        {
            cache_per_bit_count[threadIdx] = 0;
            luisa::compute::sync_block();

            const Uint orig_threadIdx = cache_index[threadIdx];
            const T    orig_value     = cache_value.read(orig_threadIdx);
            const Uint thread_bit     = (orig_value >> (pass * bit_size)) & 0xFF;

            // !Invalid: will loss the order of threads with same bit
            // const Uint offset = cache_per_bit_count.atomic(thread_bit).fetch_add(1u);
            // luisa::compute::sync_block();

            Uint offset;  // TODO: (?) How to keep the order of threads with same bit without serialization
            for (uint wid = 0; wid < 8; wid++)
            {
                $if(wid == warpIdx)
                {
                    // !Still rely on specific hardware
                    offset = cache_per_bit_count.atomic(thread_bit).fetch_add(1u);
                };
                luisa::compute::sync_block();
            }

            auto& cache_per_bit_prefix = cache_per_bit_count;
            sort_detail::block_intrinsic_scan_exclusive<uint>(
                vid, cache_per_bit_count[threadIdx], cache_scan_warp_sum, cache_per_bit_prefix);
            const Uint bit_prefix = cache_per_bit_prefix.read(thread_bit);
            const Uint insert_idx = bit_prefix + offset;
            cache_index.write(insert_idx, orig_threadIdx);
            // cache_value.write(insert_idx, orig_value);
            luisa::compute::sync_block();
        }

        const Uint orig_threadIdx = cache_index[threadIdx];
        const T    sorted_value   = cache_value.read(orig_threadIdx);
        return sorted_value;
    }

    template <typename T>
    inline Var<T> block_sort_odd_even(const luisa::compute::UInt& vid, const Var<T>& thread_value, bool ascending = true)
    {
        luisa::compute::set_block_size(reduce_block_dim);
        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;

        luisa::compute::Shared<T> cache(reduce_block_dim);
        cache[threadIdx] = thread_value;
        luisa::compute::sync_block();

        luisa::compute::UInt iter = 0u;
        $while(true)
        {
            $if(iter == reduce_block_dim)
            {
                $break;
            };

            luisa::compute::UInt phase = iter & 1u;  // 0: compare (0,1),(2,3)...  1: compare (1,2),(3,4)...
            // only threads that are the left element of a pair participate
            $if((threadIdx + 1u) < reduce_block_dim)
            {
                $if(((threadIdx & 1u) == phase))
                {
                    Var<T> a = cache[threadIdx];
                    Var<T> b = cache[threadIdx + 1u];
                    if (ascending)
                    {
                        $if(a > b)
                        {
                            cache[threadIdx]      = b;
                            cache[threadIdx + 1u] = a;
                        };
                    }
                    else
                    {
                        $if(a < b)
                        {
                            cache[threadIdx]      = b;
                            cache[threadIdx + 1u] = a;
                        };
                    }
                };
            };
            luisa::compute::sync_block();
            iter = iter + 1u;
        };

        return cache[threadIdx];
    }

    template <typename T>
    inline std::pair<Var<ushort>, Var<T>> block_bitonic_sort(const luisa::compute::UInt& vid,
                                                             const Var<T>&               thread_value,
                                                             bool ascending = true)
    {
        luisa::compute::set_block_size(reduce_block_dim);
        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;

        luisa::compute::Shared<ushort> cache_key(reduce_block_dim);
        luisa::compute::Shared<T>      cache_value(reduce_block_dim);
        cache_key[threadIdx]   = threadIdx;
        cache_value[threadIdx] = thread_value;
        luisa::compute::sync_block();

        luisa::compute::UInt k = 2u;
        // luisa::compute::UInt count = 0u;
        $while(true)
        {
            $if(k > reduce_block_dim)
            {
                $break;
            };
            luisa::compute::UInt j = k >> 1u;
            $while(true)
            {
                $if(j == 0u)
                {
                    $break;
                };

                // count += 1;
                luisa::compute::UInt ixj = threadIdx ^ j;
                $if(ixj > threadIdx)
                {
                    luisa::compute::Bool tmp = ((threadIdx & k) == 0u);
                    $if(((tmp == ascending) & (cache_value[threadIdx] > cache_value[ixj]))
                        | ((tmp != ascending) & (cache_value[threadIdx] < cache_value[ixj])))
                    {
                        sort_detail::swap_template(cache_value[threadIdx], cache_value[ixj]);
                        sort_detail::swap_template(cache_key[threadIdx], cache_key[ixj]);
                    };
                };
                luisa::compute::sync_block();
                j = j >> 1u;
            };
            k = k << 1u;
        };
        // $if (vid == 0) {
        //     device_log("block_bitonic_sort iters = {}", count);
        // };
        return std::make_pair(cache_key[threadIdx], cache_value[threadIdx]);
    }

    template <typename T>
    inline void block_bitonic_sort(luisa::compute::Shared<ushort>& cache_key,
                                   luisa::compute::Shared<T>&      cache_value,
                                   const luisa::compute::UInt&     vid,
                                   const Var<T>&                   thread_value,
                                   bool                            ascending = true)
    {
        luisa::compute::set_block_size(reduce_block_dim);
        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;

        luisa::compute::UInt k = 2u;
        $while(true)
        {
            $if(k > reduce_block_dim)
            {
                $break;
            };
            luisa::compute::UInt j = k >> 1u;
            $while(true)
            {
                $if(j == 0u)
                {
                    $break;
                };
                luisa::compute::UInt ixj = threadIdx ^ j;
                $if(ixj > threadIdx)
                {
                    luisa::compute::Bool tmp = ((threadIdx & k) == 0u);
                    $if(((tmp == ascending) & (cache_value[threadIdx] > cache_value[ixj]))
                        | ((tmp != ascending) & (cache_value[threadIdx] < cache_value[ixj])))
                    {
                        sort_detail::swap_template(cache_value[threadIdx], cache_value[ixj]);
                        sort_detail::swap_template(cache_key[threadIdx], cache_key[ixj]);
                    };
                };
                luisa::compute::sync_block();
                j = j >> 1u;
            };
            k = k << 1u;
        };
    }


    template <typename T>
    inline Var<T> block_bitonic_sort_ranged(const luisa::compute::UInt& vid,
                                            const Var<T>&               thread_value,
                                            const Var<luisa::uint2>&    range,
                                            bool                        ascending = true)
    {
        luisa::compute::set_block_size(reduce_block_dim);
        const luisa::compute::UInt threadIdx = vid % reduce_block_dim;

        luisa::compute::Shared<T> cache(reduce_block_dim);
        cache[threadIdx] = thread_value;
        luisa::compute::sync_block();

        luisa::compute::UInt k = 2u;

        auto swap = [](Var<T>& a, Var<T>& b)
        {
            Var<T> tmp = a;
            a          = b;
            b          = tmp;
        };

        $while(true)
        {
            $if(k > reduce_block_dim)
            {
                $break;
            };
            luisa::compute::UInt j = k >> 1u;
            $while(true)
            {
                $if(j == 0u)
                {
                    $break;
                };
                luisa::compute::UInt ixj = threadIdx ^ j;
                $if((ixj > threadIdx) & (ixj >= range.x & ixj < range.y))
                {
                    luisa::compute::Bool tmp = (threadIdx & k) == 0u;
                    $if((tmp == ascending & cache[threadIdx] > cache[ixj])
                        | (tmp != ascending & cache[threadIdx] < cache[ixj]))
                    {
                        swap(cache[threadIdx], cache[ixj]);
                    };
                };
                luisa::compute::sync_block();
                j = j >> 1u;
            };
            k = k << 1u;
        };

        return (cache[threadIdx]);
    }

    template <typename T>
    inline luisa::compute::Shader<1, luisa::compute::BufferView<T>> generate_fill_shader(luisa::compute::Device& device,
                                                                                         const T& value)
    {
        return device.compile<1>(
            [value](Var<luisa::compute::BufferView<T>>& buffer)
            {
                const luisa::compute::UInt index = luisa::compute::dispatch_id().x;
                buffer->write(index, value);
            });
    }

}  // namespace ParallelIntrinsic


// #define set_op_aabb(a, b)    a.cols[0] = b.cols[0]; a.cols[1] = b.cols[1];
// #define get_op_aabb(a, b)    a.cols[0] = b.cols[0]; a.cols[1] = b.cols[1];
// #define reduce_op_aabb(a, b) a.cols[0] = min_vec(a.cols[0], b.cols[0]); a.cols[1] = max_vec(a.cols[1], b.cols[1]);

// #define set_op_aabb(a, b)    a = b;
// #define get_op_aabb(a, b)    a = b;
// #define reduce_op_aabb(a, b) a.cols[0] = min_vec(a.cols[0], b.cols[0]); a.cols[1] = max_vec(a.cols[1], b.cols[1]);
// #define reduce_aabb(aabb, tid, wid, dim) reduce_with_cache(aabb, tid, wid, dim, float2x3, set_op_aabb, reduce_op_aabb, get_op_aabb)

// typename SetCacheOp, typename GetCacheOp,
/*
template<typename T, typename ReduceOp, bool use_second_reduce>
class ReduceHelper
{
private:
    luisa::compute::Shader<1, luisa::compute::BufferView<T>> reduce_vert_tree_global_aabb;

public:
    void init(luisa::compute::Device& device, luisa::compute::BufferView<T>& block_value)
    {
        reduce_vert_tree_global_aabb = device.compile<1>([
            block_view = block_value.view()
        ]
        ()
        {
            luisa::compute::set_block_size(256);
            const luisa::compute::UInt vid = luisa::compute::dispatch_id().x;
            const luisa::compute::UInt threadIdx = vid % 256;
            const luisa::compute::UInt warpIdx = vid / 256;
            
            luisa::compute::Shared<T> cache_aabb(256);
            cache_aabb[threadIdx] = threadIdx;
            luisa::compute::sync_block();
            luisa::compute::UInt s = 256 >> 1;
            $while(true) 
            {
                $if (s == 0) { $break; };  
                $if(threadIdx < s) 
                {                                             
                    ReduceOp(cache_aabb[threadIdx], cache_aabb[threadIdx + s]);       
                };                                                         
                luisa::compute::sync_block();                              
                s >>= 1;                                                   
            };
            $if (warpIdx == 0) 
            {
                block_view->write(vid, cache_aabb[0]);
            };
        });
    }

    void reduce(luisa::compute::Stream& stream, luisa::compute::BufferView<T>& block_value)
    {
        stream << reduce_vert_tree_global_aabb(block_value).diaptch();
    }
};
*/

}  // namespace lcs
