
#include <iostream>
#include <luisa/luisa-compute.h>

int main(int argc, char** argv)
{
    luisa::log_level_info();
    LUISA_INFO("Test jit");

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

    using namespace luisa::compute;

    struct LBVH_DATA
    {
        Buffer<uint>   sa_sorted_get_original;
        Buffer<float3> sa_positions;
    };
    LBVH_DATA lbvh_data;
    lbvh_data.sa_sorted_get_original = device.create_buffer<uint>(1000);
    lbvh_data.sa_positions           = device.create_buffer<float3>(1000);

    stream << synchronize();

    auto fn_test_compile_refit = [&](const uint loop)
    {
        auto fn_update_vert_tree_leave_aabb = device.compile<1>(
            [sa_sorted_get_original = lbvh_data.sa_sorted_get_original.view()](
                const Var<Buffer<float3>> sa_x_start, const Var<Buffer<float3>> sa_x_end, const Float thickness)
            {
                const UInt lid = dispatch_id().x;
                UInt       vid = sa_sorted_get_original->read(lid);
            });

        const float thickness = 0.0f;
        stream
            << fn_update_vert_tree_leave_aabb(lbvh_data.sa_positions, lbvh_data.sa_positions, thickness).dispatch(1000)
            << synchronize();

        std::cout << loop << " ";
    };

    const uint test_times = 1;
    for (uint i = 1; i <= test_times; i++)
    {
        const uint loop = i;
        fn_test_compile_refit(loop);
    }

    return 0;
}