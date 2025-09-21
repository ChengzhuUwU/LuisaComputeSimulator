#pragma once

// #include <vcruntime_typeinfo.h>
// #undef max
// #undef min
#include <numeric>
#include <vector>

#ifdef LUISA_COMPUTE_SOLVER_USE_SYSTEM_PARALLEL_FOR

#if defined(LUISA_COMPUTE_SOLVER_ENABLE_TBB)
    #define LCS_PARALLEL_USE_TBB
    #include <tbb/parallel_for.h>
#elif defined(__APPLE__)
    #define LCS_PARALLEL_USE_DISPATCH
    #include <dispatch/dispatch.h>
    // #include "luisa/core/fiber.h"
    // #include "luisa/core/shared_function.h"
#elif defined(LUISA_COMPUTE_SOLVER_ENABLE_LIBDISPATCH)
    #define LCS_PARALLEL_USE_DISPATCH
    #include <dispatch/dispatch.h>
#endif

#endif


namespace CpuParallel
{

using uint = unsigned int;

// ------------------- openmp ------------------- //
// extern int CPU_THREAD_NUM;

// inline int get_cpu_thread_num() { return CPU_THREAD_NUM; }
// inline int get_cpu_process_num() { return omp_get_num_procs(); }
// inline int get_current_thread_id() { return omp_get_thread_num(); }

// #define STR(x) #x

// #define omp_barrier _Pragma(STR(omp barrier))
// #define omp_single _Pragma(STR(omp single))
// #define omp_parallel _Pragma(STR(omp parallel num_threads(CPU_THREAD_NUM)))
// #define omp_for _Pragma(STR(omp for))
// #define omp_parallel_for _Pragma(STR(omp parallel for num_threads(CPU_THREAD_NUM)))

// #define omp_parallel_for_reduction(reduction_op, value) _Pragma(STR(omp parallel for num_threads(CPU_THREAD_NUM) reduction(reduction_op:value)))
// #define omp_parallel_for_reduction_sum(sum) omp_parallel_for_reduction(+, sum)

// ------------------- tbb ------------------- //

template<typename T1, typename T2> inline T1 max_scalar(const T1& left, const T2& right) { return left > right ? left : right; }
template<typename T1, typename T2> inline T1 min_scalar(const T1& left, const T2& right) { return left < right ? left : right; }

template<typename T> static inline bool default_compate(const T& left, const T& right) { return left < right; }


#if defined (LCS_PARALLEL_USE_TBB)

template<typename FuncName>
void parallel_for(uint start_pos, uint end_pos, FuncName func, const uint blockDim = 256)
{
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;

    tbb::parallel_for(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), 
        [&](tbb::blocked_range<uint> r) 
        { 
            uint blockIdx = r.begin();
            uint startIdx = max_scalar(blockDim * blockIdx, start_pos);
            uint endIdx = min_scalar(blockDim * (blockIdx + 1), end_pos);
            for (uint index = startIdx; index < endIdx; index++) 
            {
                func(index); 
            }
        }, tbb::simple_partitioner{});
}

template<typename FuncName>
void parallel_for_each_core(uint start_core_idx, uint end_core_idx, FuncName func){
    
    tbb::parallel_for(tbb::blocked_range<uint>(start_core_idx, end_core_idx, 1), 
        [&](tbb::blocked_range<uint> r) 
        { 
            uint blockIdx = r.begin();
            func(blockIdx);
        }, 
        tbb::simple_partitioner{});
}

template<typename T, typename ParallelFunc, typename ReduceFuncBinary>
inline T parallel_for_and_reduce(uint start_pos, uint end_pos, ParallelFunc func_parallel, ReduceFuncBinary func_binary, const T zero)
{
    const uint blockDim = 256;
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;
    // parallel_reduce
    return tbb::parallel_deterministic_reduce(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), zero, 
        [&]( tbb::blocked_range<uint> r, T result ) 
        {
            uint blockIdx = r.begin();
            uint startIdx = max_scalar(blockDim * blockIdx, start_pos);
            uint endIdx = min_scalar(blockDim * (blockIdx + 1), end_pos);

            for (uint index = startIdx; index < endIdx; index++) 
            {
                T parallel_result = func_parallel(index);
                result = func_binary(result, parallel_result);
                // func_binary(result, parallel_result);
            }
            return result;  
        }, 
        func_binary, 
        tbb::simple_partitioner{} );
}

// inclusive : 包含第一个元素, func_output(index, block_prefix, parallel_result);
template<typename T, typename ParallelFunc, typename OutputFunc>
inline void parallel_for_and_scan(uint start_pos, uint end_pos, ParallelFunc func_parallel, OutputFunc func_output, const T& zero)
{

    const uint blockDim = 256;
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim;

    tbb::parallel_scan(tbb::blocked_range<uint>(start_dispatch, end_dispatch, 1), zero, 
        [&]( tbb::blocked_range<uint> r, T block_prefix, auto is_final_scan) -> T
        {

            uint start_blockIdx = r.begin();
            uint end_blockIdx = r.end() - 1;

            uint startIdx = max_scalar(blockDim * start_blockIdx, start_pos);
            uint endIdx   = min_scalar(blockDim * (end_blockIdx + 1), end_pos);

            for (uint index = startIdx; index < endIdx; index++) 
            {
                T parallel_result = func_parallel(index);
                block_prefix += parallel_result;
                if (is_final_scan) 
                {
                    func_output(index, block_prefix, parallel_result);
                }
            }
            return block_prefix;
        }, 
        [](const T& x, const T& y) -> T{return x + y;},
        tbb::simple_partitioner{} );

    // tbb::parallel_scan(tbb::blocked_range<uint>(start_pos, end_pos), zero, 
    // [&]( tbb::blocked_range<uint> r, T block_prefix, auto is_final_scan) -> T{
    //     for (auto i = r.begin(); i != r.end(); ++i) {
    //         T parallel_result = func_parallel(i);
    //         block_prefix += parallel_result;
    //         if(is_final_scan) {
    //             func_output(i, block_prefix, parallel_result);
    //         }   
    //     }
    //     return block_prefix;
    // }, 
    // [](const T& x, const T& y) -> T{return x + y;} );

}

template <typename Ptr, typename _Comp>
inline void parallel_sort(Ptr begin, Ptr end, _Comp comp = default_compate)
{
    tbb::parallel_sort(begin, end, comp);
}

#elif defined (LCS_PARALLEL_USE_DISPATCH)

namespace detail
{
    // template<class F>
    // requires(std::is_invocable_v<F, uint32_t, uint32_t, uint32_t>)
    // static void parallel_template(uint32_t start_pos, uint32_t end_pos, F &&block_func, uint32_t internal_jobs) noexcept 
    // {
    //     const uint job_count = end_pos - start_pos;
    //     const uint start_dispatch = start_pos / internal_jobs;
    //     const uint end_dispatch   = (end_pos + internal_jobs - 1) / internal_jobs;
    //     // auto thread_count = std::clamp<uint32_t>(job_count / internal_jobs, 1u, luisa::fiber::worker_thread_count());
    //     auto thread_count = end_dispatch - start_dispatch; // std::clamp<uint32_t>(job_count / internal_jobs, 1u, luisa::fiber::worker_thread_count());
    //     if (thread_count > 1) 
    //     {
    //         luisa::fiber::counter evt{thread_count};
    //         luisa::SharedFunction<void()> func
    //         {
    //             [
    //                 counter = luisa::fiber::detail::NonMovableAtomic<uint32_t>(0), 
    //                 job_count, thread_count, start_pos, end_pos,
    //                 internal_jobs, 
    //                 evt, 
    //                 // block_func = std::forward<F>(block_func)
    //                 &block_func
    //             ]() mutable noexcept
    //             {
    //                 uint32_t block_idx = 0u;
    //                 while ((block_idx = counter.value.fetch_add(1)) < thread_count) 
    //                 {
    //                     const uint startIdx = max_scalar(internal_jobs * block_idx, start_pos);
    //                     const uint endIdx = min_scalar(internal_jobs * (block_idx + 1), end_pos);
    //                     block_func(block_idx, startIdx, endIdx);
    //                 }
    //                 evt.done();
    //             }
    //         };
    //         for (uint32_t i = 0; i < thread_count; ++i)  { marl::schedule(func); }
    //         evt.wait();
    //     } 
    //     else 
    //     {
    //         block_func(0, start_pos, end_pos);
    //     }
    // }

};


template<typename ParallelFunc>
inline void parallel_for(uint start_pos, uint end_pos, ParallelFunc func_parallel, const uint blockDim = 32)
{
    uint start_dispatch = start_pos / blockDim; 
    uint end_dispatch = (end_pos + blockDim - 1) / blockDim; 
    dispatch_apply(end_dispatch - start_dispatch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), 
        ^(size_t blockIdx) 
        { 
            uint startIdx = max_scalar(blockDim * blockIdx, start_pos); 
            uint endIdx = min_scalar(blockDim * (blockIdx + 1), end_pos); 
            for (uint index = startIdx; index < endIdx; index++) 
            { 
                func_parallel(index); 
            } 
        });

    // const uint job_count = end_pos - start_pos;
    // const uint start_dispatch = start_pos / internal_jobs;
    // const uint end_dispatch   = (end_pos + internal_jobs - 1) / internal_jobs;
    // const uint thread_count = end_dispatch - start_dispatch;
    // if (thread_count > 1) 
    // {
    //     luisa::fiber::counter evt{thread_count};
    //     luisa::SharedFunction<void()> func
    //     {
    //         [
    //             counter = luisa::fiber::detail::NonMovableAtomic<uint32_t>(0), 
    //             job_count, thread_count, start_pos, end_pos,
    //             internal_jobs, 
    //             evt, 
    //             &func_parallel
    //         ]() mutable noexcept
    //         {
    //             uint32_t block_idx = 0u;
    //             while ((block_idx = counter.value.fetch_add(1)) < thread_count) 
    //             {
    //                 const uint startIdx = max_scalar(internal_jobs * block_idx, start_pos);
    //                 const uint endIdx = min_scalar(internal_jobs * (block_idx + 1), end_pos);
    //                 for (uint index = startIdx; index < endIdx; index++)
    //                 {
    //                     func_parallel(index);
    //                 }
    //             }
    //             evt.done();
    //         }
    //     };
    //     for (uint32_t i = 0; i < thread_count; ++i)  { marl::schedule(func); }
    //     evt.wait();
    // } 
    // else 
    // {
    //     for (uint index = start_pos; index < end_pos; index++)
    //     {
    //         func_parallel(index);
    //     }
    // }
}

template<typename T, typename ParallelFunc, typename ReduceFuncBinary>
inline T parallel_for_and_reduce(uint start_pos, uint end_pos, ParallelFunc func_parallel, ReduceFuncBinary func_binary, const T zero)
{
    const uint blockDim = 256;
    const uint start_dispatch = start_pos / blockDim;
    const uint end_dispatch   = (end_pos + blockDim - 1) / blockDim;
    const uint num_dispatch   = end_dispatch - start_dispatch;

    std::vector<T> partials(num_dispatch, zero);

    struct ParallelReduceContext 
    {
        ParallelFunc *func_parallel;
        ReduceFuncBinary *func_binary;
        std::vector<T> *partials;
        uint start_pos;
        uint end_pos;
        uint blockDim;
        T zero;
        uint start_dispatch;
    };

    auto parallel_reduce_func = [](void *context, size_t blockIdx) 
    {
        auto *ctx = static_cast<ParallelReduceContext *>(context);
        uint startIdx = max_scalar(ctx->blockDim * blockIdx, ctx->start_pos);
        uint endIdx = min_scalar(ctx->blockDim * (blockIdx + 1), ctx->end_pos);
        T block_sum = ctx->zero;
        for (uint index = startIdx; index < endIdx; index++) 
        {
            T local_val = (*(ctx->func_parallel))(index);
            block_sum = (*(ctx->func_binary))(block_sum, local_val);
        }
        (*(ctx->partials))[blockIdx - ctx->start_dispatch] = block_sum;
    };

    ParallelReduceContext context = { &func_parallel, &func_binary, &partials, start_pos, end_pos, blockDim, zero, start_dispatch };

    dispatch_apply_f(num_dispatch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, parallel_reduce_func);

    return std::reduce(partials.begin(), partials.end(), zero, func_binary);
}

template<typename ParallelFunc>
void parallel_for_each_core(uint start_core_idx, uint end_core_idx, ParallelFunc func_parallel)
{
    // dispatch_apply_f(end_core_idx - start_core_idx, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &func, [start_core_idx](void *context, size_t idx) noexcept {
    //     auto task = static_cast<ParallelFunc *>(context);
    //     (*task)(static_cast<uint>(start_core_idx + idx));
    // });
    // dispatch_apply_f(end_core_idx - start_core_idx, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &func, 
    //     [&](void *context, size_t blockIdx) noexcept 
    //     {
    //         auto task = static_cast<ParallelFunc *>(context);
    //         (*task)(static_cast<uint>(start_core_idx + blockIdx));
    //     });

    const uint thread_count = end_core_idx - start_core_idx;
    if (thread_count > 1) 
    {
        dispatch_apply(thread_count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), 
        ^(size_t threadIdx) 
        { 
            func_parallel(start_core_idx + threadIdx);
        });
    } 
    else 
    {
        for (uint coreIdx = start_core_idx; coreIdx < end_core_idx; coreIdx++)
        {
            func_parallel(coreIdx);
        }
    }
}

template <typename Ptr, typename _Comp>
inline void parallel_sort(Ptr begin, Ptr end, _Comp comp = default_compate)
{
    std::sort(begin, end, comp);
}

template<typename T, typename ParallelFunc, typename OutputFunc>
inline void parallel_for_and_scan(const uint start_pos, const uint end_pos,
                                  ParallelFunc func_parallel,
                                  OutputFunc func_output,
                                  const T& zero)
{
    const uint blockDim = 256;
    uint start_dispatch = start_pos / blockDim;
    uint end_dispatch   = (end_pos + blockDim - 1) / blockDim;
    uint thread_count     = end_dispatch - start_dispatch;

    if (thread_count > 1) 
    {
        std::vector<T> list_prefix_thread(end_pos - start_pos); // inclusive
        std::vector<T> list_prefix_block(thread_count); // exclusive


        struct ParallelReduceContext 
        {
            ParallelFunc *func_parallel;
            OutputFunc *func_output;
            std::vector<T> *list_prefix_thread;
            std::vector<T> *list_prefix_block;
            uint start_pos;
            uint end_pos;
            uint blockDim;
            T zero;
            uint start_dispatch;
        };

        auto parallel_reduce_func = [](void *context, size_t blockIdx) 
        {
            auto *ctx = static_cast<ParallelReduceContext *>(context);
            uint startIdx = max_scalar(ctx->blockDim * blockIdx, ctx->start_pos);
            uint endIdx = min_scalar(ctx->blockDim * (blockIdx + 1), ctx->end_pos);
            T prefix_block = ctx->zero;
            for (uint index = startIdx; index < endIdx; index++)
            {
                T val = (*(ctx->func_parallel))(index);
                prefix_block = prefix_block + val;
                (*(ctx->list_prefix_thread))[index] = prefix_block; // inclusive
            }
            (*(ctx->list_prefix_block))[blockIdx] = prefix_block;
        };

        ParallelReduceContext context = { &func_parallel, &func_output, &list_prefix_thread, &list_prefix_block, start_pos, end_pos, blockDim, zero, start_dispatch };

        dispatch_apply_f(thread_count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, parallel_reduce_func);

        std::exclusive_scan(list_prefix_block.begin(), list_prefix_block.end(), list_prefix_block.begin(), zero);

        auto parallel_output_func = [](void *context, size_t blockIdx) 
        {
            auto *ctx = static_cast<ParallelReduceContext *>(context);
            const uint startIdx = max_scalar(blockDim * blockIdx, ctx->start_pos); 
            const uint endIdx = min_scalar(blockDim * (blockIdx + 1), ctx->end_pos); 
            const T prefix1 = (*(ctx->list_prefix_block))[blockIdx];
            for (uint index = startIdx; index < endIdx; index++) 
            {
                const T curr_offset = index == startIdx ? ctx->zero : (*(ctx->list_prefix_thread))[index - 1];
                const T next_offset = (*(ctx->list_prefix_thread))[index];
                (*(ctx->func_output))(index, prefix1 + next_offset, next_offset - curr_offset);
            }    
        };

        dispatch_apply_f(thread_count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &context, parallel_output_func);

    } 
    else 
    {
        T prefix = zero;
        for (uint index = start_pos; index < end_pos; index++)
        {
            T curr_val = (func_parallel(index));
            prefix = prefix + curr_val;
            func_output(index, prefix, curr_val);
        }
    }

    // dispatch_apply_f(end_dispatch - start_dispatch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &func_output, 
    //     [&](void *context, size_t blockIdx) noexcept 
    //     {
    //         auto task_output = static_cast<ParallelFunc *>(context);
    //         uint globalBlockIdx = blockIdx + start_dispatch;
    //         uint startIdx = max_scalar(blockDim * globalBlockIdx, start_pos);
    //         uint endIdx   = min_scalar(blockDim * (globalBlockIdx + 1), end_pos);
    //         const T prefix1 = prefix_block[blockIdx];
    //         for (uint index = startIdx; index < endIdx; index++) 
    //         {
    //             const T curr_offset = index == startIdx ? zero : prefix_thread[index - 1];
    //             const T next_offset = prefix_thread[index];
    //             (*task_output)(static_cast<uint>(index, prefix1 + curr_offset, next_offset - curr_offset));
    //         }
    //     });
}

#else

template<typename FuncName>
void parallel_for(uint start_pos, uint end_pos, FuncName func, const uint blockDim = 256)
{
    for (uint index = start_pos; index < end_pos; index++) 
    {
        func(index); 
    }
}

template<typename FuncName>
void parallel_for_each_core(uint start_core_idx, uint end_core_idx, FuncName func)
{
    for (uint index = start_core_idx; index < end_core_idx; index++) 
    {
        func(index); 
    }
}

template<typename T, typename ParallelFunc, typename ReduceFuncBinary>
inline T parallel_for_and_reduce(uint start_pos, uint end_pos, ParallelFunc func_parallel, ReduceFuncBinary func_binary, const T zero)
{
    std::vector<T> thread_result(end_pos - start_pos);
    for (uint index = start_pos; index < end_pos; index++) 
    {
        thread_result[index - start_pos] = func_parallel(index); 
    }
    return std::reduce(thread_result.begin(), thread_result.end(), zero, func_binary);
}

// inclusive : 包含第一个元素, func_output(index, block_prefix, parallel_result);
template<typename T, typename ParallelFunc, typename OutputFunc>
inline void parallel_for_and_scan(uint start_pos, uint end_pos, ParallelFunc func_parallel, OutputFunc func_output, const T& zero)
{
    std::vector<T> thread_result(end_pos - start_pos);
    for (uint index = start_pos; index < end_pos; index++) 
    {
        thread_result[index - start_pos] = func_parallel(index); 
    }
    std::exclusive_scan(thread_result.begin(), thread_result.end(), thread_result.begin(), zero);
    for (uint index = start_pos; index < end_pos; index++) 
    {
        thread_result[index - start_pos] = func_output(
            index, 
            thread_result[index], 
            index == start_pos ? thread_result[0] : thread_result[index] - thread_result[index - 1]); 
    }
}

template <typename Ptr, typename _Comp>
inline void parallel_sort(Ptr begin, Ptr end, _Comp comp = default_compate)
{
    std::parallel_sort(begin, end, comp);
}

#endif


template<typename FuncName>
void single_thread_for(uint start_idx, uint end_idx, FuncName func, const uint blockDim = 32)
{
    for(uint index = start_idx; index < end_idx; index++){ func(index); }
}

template<typename T, typename ParallelFunc>
inline T single_thread_for_and_reduce_sum(uint start_pos, uint end_pos, ParallelFunc func_parallel)
{
    std::vector<T> thread_values(end_pos - start_pos);
    for (uint index = start_pos; index < end_pos; index++)
    {
        T parallel_result = func_parallel(index);
        thread_values[index - start_pos] = parallel_result;
    }
    return std::reduce(thread_values.begin(), thread_values.end(), T(), [](const T& x, const T& y) -> T{return x + y;});
}

template<typename T, typename ParallelFunc, typename ReduceFuncBinary>
inline T single_thread_for_and_reduce(uint start_pos, uint end_pos, ParallelFunc func_parallel, ReduceFuncBinary func_binary, const T zero)
{
    std::vector<T> thread_values(end_pos - start_pos);
    for (uint index = start_pos; index < end_pos; index++)
    {
        T parallel_result = func_parallel(index);
        thread_values[index - start_pos] = parallel_result;
    }
    return std::reduce(thread_values.begin(), thread_values.end(), zero, func_binary);
}


template<typename T, typename ParallelFunc>
inline T parallel_for_and_reduce_sum(uint start_pos, uint end_pos, ParallelFunc func_parallel)
{
    return parallel_for_and_reduce<T>(start_pos, end_pos, 
        func_parallel, 
        // [](T& result, const T& parallel_result) -> void { result += parallel_result; }, // func_unary
        [](const T& x, const T& y) -> T{return x + y;}, // func_binary
        T()
        ); 
}

template<typename T>
inline T parallel_reduce_sum(const T* array, const uint size)
{
    return parallel_for_and_reduce_sum<T>(0, size, [&](const uint index) { return array[index]; }); 
}
template<typename T>
inline T parallel_reduce_sum(const std::vector<T>& array)
{
    return parallel_for_and_reduce_sum<T>(0, array.size(), [&](const uint index) { return array[index]; }); 
}

// From src to dst
template <typename T>
inline void parallel_copy(const T& src, T& dst, const uint array_size)
{
    parallel_for(0, array_size, [&](const uint index)
    {
        dst[index] = src[index];
    });
}

// From src to dst
template <typename T>
inline void parallel_copy(const std::vector<T>& src, std::vector<T>& dst)
{
    const uint array_size = dst.size();
    parallel_for(0, array_size, [&](const uint index)
    {
        dst[index] = src[index];
    });
}
template <typename T1, typename T2>
inline void parallel_set(T1& dst, const uint array_size, const T2& value)
{
    parallel_for(0, array_size, [&](const uint index)
    {
        dst[index] = value;
    });
}
template <typename T>
inline void parallel_set(std::vector<T>& dst, const T& value)
{
    const uint array_size = dst.size();
    parallel_for(0, array_size, [&](const uint index)
    {
        dst[index] = value;
    });
}

// [](float& x, const float& y) -> void{ x += y; },
// [](const float& x, const float& y) -> float{ return x + y; }


} // namespace lcs