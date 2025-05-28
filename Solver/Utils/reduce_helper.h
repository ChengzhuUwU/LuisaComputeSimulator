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
//             $while(true) {	                                                            \
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

        
namespace lcsv 
{

namespace ParallelIntrinsic
{

constexpr uint reduce_block_dim = 256;

template<typename T>
inline void default_reduce_op_unary(Var<T>& left, const Var<T>& right)
{
    left = left + right;
}

template<typename T, typename ReduceOp>
inline Var<T> block_reduce(const luisa::compute::UInt vid, const Var<T>& thread_value, const ReduceOp reduce_op_unary = default_reduce_op_unary<T>)
{
    using Uint = luisa::compute::UInt;
    luisa::compute::set_block_size(reduce_block_dim);
    // const luisa::compute::UInt vid = luisa::compute::dispatch_id().x;
    const luisa::compute::UInt threadIdx = vid % reduce_block_dim;
    const luisa::compute::UInt warpIdx = vid / reduce_block_dim;
    
    luisa::compute::Shared<T> cache(reduce_block_dim);
    cache[threadIdx] = thread_value;
    Var<T> block_value = thread_value;
    luisa::compute::sync_block();

    luisa::compute::UInt s = reduce_block_dim >> 1;
    $while (true) 
    {
        $if (threadIdx < s) 
        {                                             
            reduce_op_unary(cache[threadIdx], cache[threadIdx + s]);       
        };                                                         
        luisa::compute::sync_block();                              
        s >>= 1;
        $if (s == 0) { $break; };                                               
    };
    $if (threadIdx == 0)
    {
        block_value = cache[0];
    };
    return block_value;
}

template<typename T>
inline luisa::compute::Shader<1, luisa::compute::BufferView<T>> generate_fill_shader(luisa::compute::Device& device, const T& value)
{
    return device.compile<1>([value](
        Var<luisa::compute::BufferView<T>>& buffer
    ){
        const luisa::compute::UInt index = luisa::compute::dispatch_id().x;
        buffer->write(index, value);
    });
}

}


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

}