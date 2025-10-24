
#include <any>
#include <luisa/luisa-compute.h>

struct Arguments
{
    luisa::compute::Buffer<uint>            buffer1;
    luisa::compute::Buffer<float>           buffer2;
    luisa::compute::Buffer<luisa::float3>   buffer3;
    luisa::compute::Buffer<luisa::float3x3> buffer4;
};
LUISA_BINDING_GROUP(Arguments, buffer1, buffer2, buffer3, buffer4){};


void test_dynamic_resize(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    using namespace luisa::compute;
    using Uint  = UInt;
    using Uint2 = UInt2;

    Arguments args;
    auto      fn_from_ref = device.compile<1>(
        [&](Var<Arguments> args, const Uint iter)
        {
            const Uint lid = dispatch_id().x;
            args.buffer1.write(lid, iter);
        });

    auto fn_lc_test = [&](const uint loop)
    {
        const uint num_leaves = 1 << loop;
        // buffer                = device.create_sparse_buffer<uint>(num_leaves);
        args.buffer1 = device.create_buffer<uint>(num_leaves);
        args.buffer2 = device.create_buffer<float>(num_leaves);
        args.buffer3 = device.create_buffer<float3>(num_leaves);
        args.buffer4 = device.create_buffer<float3x3>(num_leaves);
        stream << fn_from_ref(args, loop).dispatch(num_leaves) << synchronize();

        std::vector<uint> host_array(num_leaves);
        stream << args.buffer1.copy_to(host_array.data()) << synchronize();
        if (!std::all_of(host_array.begin(), host_array.end(), [loop](uint v) { return v == loop; }))
        {
            LUISA_ERROR("Values mismatch at loop = {}", loop);
            return -1;
        }
        LUISA_INFO("LC Test successed.");
        return 0;
    };

    const uint test_times = 10;
    for (uint i = 1; i <= test_times; i++)
    {
        if (fn_lc_test(i) != 0)
        {
            LUISA_ERROR("Test failed at loop = {}", i);
            return;
        }
    }
}

int main(int argc, char** argv)
{
    luisa::log_level_verbose();
    LUISA_INFO("Test LC features");

    // Init GPU system
#if defined(__APPLE__)
    std::string backend = "metal";
#else
    std::string backend = "cuda";
#endif
    const std::string            binary_path(argv[0]);
    luisa::compute::Context      context{binary_path};
    luisa::vector<luisa::string> device_names = context.backend_device_names(backend);
    if (device_names.empty())
    {
        LUISA_WARNING("No haredware device found.");
        exit(1);
    }
    if (argc >= 2)
    {
        backend = argv[1];
    }
    luisa::compute::Device device = context.create_device(backend, nullptr, true);
    luisa::compute::Stream stream = device.create_stream(luisa::compute::StreamTag::COMPUTE);

    test_dynamic_resize(device, stream);

    return 0;
}