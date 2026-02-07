#pragma once

#include "Utils/cpu_parallel.h"
#include "luisa/core/spin_mutex.h"
#include "luisa/dsl/resource.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/stream.h"
#include <vector>

namespace BufferOp
{

template <typename T>
using Var = luisa::compute::Var<T>;

template <typename T>
static void buffer_copy(const std::vector<T>& src, std::vector<T>& dst)
{
    if (src.size() != dst.size())
    {
        LUISA_ERROR("Buffer size mismatch {} != {}", src.size(), dst.size());
    }
    std::memcpy(dst.data(), src.data(), sizeof(T) * src.size());
    // CpuParallel::parallel_copy(src, dst);
}
template <typename T>
static void buffer_copy(luisa::compute::Stream&          stream,
                        const luisa::compute::Buffer<T>& src,
                        luisa::compute::Buffer<T>&       dst)
{
    stream << src.copy_to(dst);
}
template <typename T>
static void buffer_upload(luisa::compute::Stream& stream, const std::vector<T>& src, luisa::compute::Buffer<T>& dst)
{
    stream << dst.copy_from(src.data());
}

template <typename T>
static void buffer_download(luisa::compute::Stream&          stream,
                            const luisa::compute::Buffer<T>& src,
                            std::vector<T>&                  dst,
                            const bool                       wait = false)
{
    stream << src.copy_to(dst.data());
    if (wait)
        stream << luisa::compute::synchronize();
}
// template <typename T>
// [[nodiscard]] static auto buffer_download(const luisa::compute::Buffer<T>& src, std::vector<T>& dst, const bool wait = false)
// {
//     if (wait)
//         return src.copy_to(dst.data()) << luisa::compute::synchronize();
//     else
//         return src.copy_to(dst.data());
// }

template <typename T>
static void buffer_add(const luisa::compute::BufferView<T>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
template <typename T>
static void buffer_add(const luisa::compute::BufferVar<T>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}

template <typename T>
static void buffer_add(std::vector<T>& buffer, const uint dest, const T& value)
{
    buffer[dest] = buffer[dest] + value;
}
template <typename T>
static void buffer_add(const std::span<T>& buffer, const uint dest, const T& value)
{
    buffer[dest] = buffer[dest] + value;
}
template <typename T>
static void atomic_buffer_add(const std::span<T>&                 buffer,
                              const std::span<luisa::spin_mutex>& mutex_ref,
                              const uint                          dest,
                              const T&                            value)
{
    // CpuParallel::spin_atomic<T>::fetch_add(buffer[dest], value);
    mutex_ref[dest].lock();
    buffer[dest] = buffer[dest] + value;
    mutex_ref[dest].unlock();
}
template <typename T>
static void atomic_buffer_add(const std::vector<T>&               buffer,
                              const std::span<luisa::spin_mutex>& mutex_ref,
                              const uint                          dest,
                              const T&                            value)
{
    // CpuParallel::spin_atomic<T>::fetch_add(buffer[dest], value);
    mutex_ref[dest].lock();
    buffer[dest] = buffer[dest] + value;
    mutex_ref[dest].unlock();
}

static void atomic_buffer_add(const luisa::compute::BufferVar<luisa::float3>& buffer,
                              const Var<uint>                                 dest,
                              const Var<luisa::float3>&                       value)
{
    buffer->atomic(dest)[0].fetch_add(value[0]);
    buffer->atomic(dest)[1].fetch_add(value[1]);
    buffer->atomic(dest)[2].fetch_add(value[2]);
}
static void atomic_buffer_add(const luisa::compute::BufferVar<luisa::float3x3>& buffer,
                              const Var<uint>                                   dest,
                              const Var<luisa::float3x3>&                       value)
{
    buffer->atomic(dest)[0][0].fetch_add(value[0][0]);
    buffer->atomic(dest)[0][1].fetch_add(value[0][1]);
    buffer->atomic(dest)[0][2].fetch_add(value[0][2]);
    buffer->atomic(dest)[1][0].fetch_add(value[1][0]);
    buffer->atomic(dest)[1][1].fetch_add(value[1][1]);
    buffer->atomic(dest)[1][2].fetch_add(value[1][2]);
    buffer->atomic(dest)[2][0].fetch_add(value[2][0]);
    buffer->atomic(dest)[2][1].fetch_add(value[2][1]);
    buffer->atomic(dest)[2][2].fetch_add(value[2][2]);
}

}  // namespace BufferOp