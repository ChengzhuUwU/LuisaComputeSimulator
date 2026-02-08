#include "lcpp/device/device_radix_sort.h"
#include "luisa/runtime/buffer.h"
#include <algorithm>
#include <lcpp/parallel_primitive.h>
#include <vector>
using namespace luisa::parallel_primitive;
int main(int argc, char* argv[])
{

    // Create device and stream
    Context ctx{argv[0]};
    Device  device = ctx.create_device("cuda");
    Stream  stream = device.create_stream();

    if (false)
    {
        // Device-level reduce example
        DeviceReduce<> device_reduce;
        device_reduce.create(device);

        Buffer<int> input  = device.create_buffer<int>(1024);
        Buffer<int> output = device.create_buffer<int>(1);

        std::vector<int> host_input(1024, 1);
        stream << input.copy_from(host_input.data()) << synchronize();

        CommandList cmdlist;
        device_reduce.Sum(cmdlist, stream, input.view(), output.view(), 1024);
        stream << cmdlist.commit() << synchronize();

        std::vector<int> host_output(1);
        stream << output.copy_to(host_output.data()) << synchronize();
        LUISA_INFO("Sum result: {}", host_output[0]);
    }
    {
        DeviceRadixSort<> device_radix_sort;
        device_radix_sort.create(device);
        for (uint loop = 5; loop < 20; ++loop)
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

            uint idx       = 0;
            bool is_sorted = std::all_of(
                host_keys_out.begin(), host_keys_out.end(), [&idx](uint key) { return key == idx++; });

            auto print_span = [&host_keys_out](size_t start, size_t count)
            {
                LUISA_INFO(" -> Value from {:4} to {:4} : {}",
                           start,
                           start + count - 1,
                           std::span(host_keys_out).subspan(start, count));
            };
            LUISA_INFO("Sorted {} items: {} to sort", num_items, is_sorted ? "Successfully" : "Failed");
            print_span(0, 10);
            print_span(num_items - 10, 10);
        }
    }
}