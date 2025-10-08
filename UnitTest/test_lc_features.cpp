
#include <luisa/luisa-compute.h>

int main(int argc, char** argv)
{
    luisa::log_level_info();
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


    using namespace luisa::compute;
    using Uint  = UInt;
    using Uint2 = UInt2;


    auto fn_test1 = device.compile<1>(
        [](const BufferVar<uint> sa_apply_flag)
        {
            const Uint lid = dispatch_id().x;
            // $if(lid == 0)
            // {
            //     device_log("Hello Device Print");
            // };
        });

    auto fn_lc_test = [&](const uint loop)
    {
        const uint num_leaves = 1 << loop;
        LUISA_INFO("In loop = {},", loop);

        Buffer<uint>        sa_apply_flag = device.create_buffer<uint>(num_leaves);
        luisa::vector<uint> host_apply_flag(num_leaves, 0u);

        stream << sa_apply_flag.copy_from(host_apply_flag.data()) << fn_test1(sa_apply_flag).dispatch(num_leaves)
               << sa_apply_flag.copy_to(host_apply_flag.data()) << synchronize();

        LUISA_INFO(".  LC Test successed.");
        return 0;
    };

    const uint test_times = 10;
    for (uint i = 1; i <= test_times; i++)
    {
        if (fn_lc_test(i) != 0)
        {
            LUISA_ERROR("Test failed at loop = {}", i);
            return -1;
        }
    }

    return 0;
}