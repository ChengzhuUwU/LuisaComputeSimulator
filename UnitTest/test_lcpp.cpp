#include "lcpp/device/device_radix_sort.h"
#include "luisa/core/logging.h"
#include "luisa/runtime/buffer.h"
#include <algorithm>
#include <functional>
#include <lcpp/parallel_primitive.h>
#include <string_view>
#include <vector>
using namespace luisa::parallel_primitive;

template <typename T, typename FnGetDesired>
bool check_template(const T& actual, FnGetDesired fn_get_desired, const std::string_view test_name, uint size)
{
    for (uint i = 0; i < actual.size(); ++i)
    {
        auto desired = fn_get_desired(i);
        if (actual[i] != desired)
        {
            // LUISA_WARNING("Test failed at size {:5}: index {}, expected {}, got {}", size, i, desired, actual[i]);
            LUISA_WARNING("{:12} {:4} elements: Failed at index {}, expected {}, got {}", test_name, size, i, desired, actual[i]);
            return false;
        }
    }
    LUISA_INFO("{:12} {:4} elements: all values are correct.", test_name, size);
    return true;
}

void test_device_reduce(Device& device, Stream& stream)
{
    // Device-level reduce example
    DeviceReduce<> device_reduce;
    device_reduce.create(device);

    using Type4Byte = int;

    for (uint loop = 0; loop < 24; ++loop)
    {
        uint                   num_items = 1 << loop;
        Buffer<Type4Byte>      d_input   = device.create_buffer<Type4Byte>(num_items);
        Buffer<Type4Byte>      d_output  = device.create_buffer<Type4Byte>(1);
        std::vector<Type4Byte> host_input(num_items, 1);
        stream << d_input.copy_from(host_input.data()) << synchronize();
        CommandList cmdlist;
        device_reduce.Sum(cmdlist, stream, d_input.view(), d_output.view(), 1024);
        // device_reduce.Sum(cmdlist, stream, d_input.view(), d_output.view(), num_items);
        stream << cmdlist.commit() << synchronize();
        std::vector<Type4Byte> host_output(1);
        stream << d_output.copy_to(host_output.data()) << synchronize();

        check_template(
            host_output, [num_items](uint) -> Type4Byte { return num_items; }, "Device Reduce", num_items);

        d_input.release();
        d_output.release();
    }
}
void test_device_scan(Device& device, Stream& stream)
{
    DeviceScan<> device_scan;
    device_scan.create(device);
    for (uint loop = 0; loop < 24; ++loop)
    {
        uint              num_items  = 1 << loop;
        Buffer<uint>      d_keys_in  = device.create_buffer<uint>(num_items);
        Buffer<uint>      d_keys_out = device.create_buffer<uint>(num_items);
        std::vector<uint> host_keys(num_items);
        for (uint i = 0; i < num_items; ++i)
        {
            host_keys[i] = 1;
        }
        stream << d_keys_in.copy_from(host_keys.data()) << synchronize();
        CommandList cmdlist;
        device_scan.ExclusiveSum(cmdlist, stream, d_keys_in.view(), d_keys_out.view(), num_items);

        // `ExclusiveScan` should provide binary-operation
        // device_scan.ExclusiveScan(
        //     cmdlist,
        //     stream,
        //     d_keys_in.view(),
        //     d_keys_out.view(),
        //     num_items,
        //     [](UInt a, UInt b) { return a + b; },
        //     0u);
        stream << cmdlist.commit() << synchronize();
        std::vector<uint> host_keys_out(num_items);
        stream << d_keys_out.copy_to(host_keys_out.data()) << synchronize();

        d_keys_in.release();
        d_keys_out.release();

        check_template(
            host_keys_out, [](uint i) -> uint { return i; }, "Device Scan", num_items);
    }
}
void test_device_radix_sort(Device& device, Stream& stream)
{

    DeviceRadixSort<> device_radix_sort;
    device_radix_sort.create(device);
    for (uint loop = 0; loop < 24; ++loop)
    {
        uint              num_items    = 1 << loop;
        Buffer<uint>      d_keys_in    = device.create_buffer<uint>(num_items);
        Buffer<uint>      d_keys_out   = device.create_buffer<uint>(num_items);
        Buffer<uint>      d_values_in  = device.create_buffer<uint>(num_items);
        Buffer<uint>      d_values_out = device.create_buffer<uint>(num_items);
        std::vector<uint> host_keys(num_items);
        for (uint i = 0; i < num_items; ++i)
        {
            host_keys[i] = num_items - i - 1;
        }
        stream << d_keys_in.copy_from(host_keys.data()) << synchronize();
        CommandList cmdlist;
        device_radix_sort.SortKeys(cmdlist, stream, d_keys_in.view(), d_keys_out.view(), num_items);
        stream << cmdlist.commit() << synchronize();
        std::vector<uint> host_keys_out(num_items);
        stream << d_keys_out.copy_to(host_keys_out.data()) << synchronize();

        d_keys_in.release();
        d_keys_out.release();
        d_values_in.release();
        d_values_out.release();

        check_template(
            host_keys_out, [](uint i) -> uint { return i; }, "Device Radix Sort", num_items);
    }
}


int main(int argc, char* argv[])
{

    // Create device and stream
    Context ctx{argv[0]};
    auto    backend = argc > 1 ? argv[1] : "cuda";
    Device  device  = ctx.create_device("cuda");
    Stream  stream  = device.create_stream();


    // Run tests
    test_device_reduce(device, stream);
    test_device_scan(device, stream);
    test_device_radix_sort(device, stream);
    return 0;
}