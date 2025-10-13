
#include <luisa/luisa-compute.h>

struct CompressedAABB
{
    std::array<float, 3> min_bound;
    uint                 flag1;
    std::array<float, 3> max_bound;
    uint                 flag2;
    auto                 get_aabb()
    {
        std::array<luisa::float3, 2> aabb;
        aabb[0] = luisa::make_float3(min_bound[0], min_bound[1], min_bound[2]);
        aabb[1] = luisa::make_float3(max_bound[0], max_bound[1], max_bound[2]);
        return aabb;
    }
};

// clang-format off
LUISA_STRUCT(CompressedAABB, min_bound, flag1, max_bound, flag2)
{
    auto get_aabb()
    {
        luisa::compute::Var<std::array<luisa::float3, 2>> aabb;
        aabb[0] = luisa::compute::make_float3(min_bound[0], min_bound[1], min_bound[2]);
        aabb[1] = luisa::compute::make_float3(max_bound[0], max_bound[1], max_bound[2]);
        return aabb;
    }
};
// clang-format on

inline CompressedAABB make_compressed_aabb()
{
    CompressedAABB c{};
    c.min_bound = {1e30f, 1e30f, 1e30f};
    c.max_bound = {-1e30f, -1e30f, -1e30f};
    c.flag1     = 0u;
    c.flag2     = 0u;
    return c;
}
inline CompressedAABB make_compressed_aabb(const std::array<luisa::float3, 2>& aabb, const uint flag1, const uint flag2)
{
    CompressedAABB c{};
    c.min_bound = {aabb[0].x, aabb[0].y, aabb[0].z};
    c.max_bound = {aabb[1].x, aabb[1].y, aabb[1].z};
    c.flag1     = flag1;
    c.flag2     = flag2;
    return c;
}
inline luisa::compute::Var<CompressedAABB> make_compressed_aabb(const luisa::compute::Var<std::array<luisa::float3, 2>>& aabb,
                                                                const luisa::compute::UInt flag1,
                                                                const luisa::compute::UInt flag2)
{
    luisa::compute::Var<CompressedAABB> c{};
    c.min_bound = {aabb[0].x, aabb[0].y, aabb[0].z};
    c.max_bound = {aabb[1].x, aabb[1].y, aabb[1].z};
    c.flag1     = flag1;
    c.flag2     = flag2;
    return c;
}


int main(int argc, char** argv)
{
    luisa::log_level_info();
    LUISA_INFO("Test LBVH refit");

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
    using float2x3 = std::array<float3, 2>;
    using aabbData = float2x3;
    using Float2x3 = Var<float2x3>;
    using Uint     = UInt;
    using Uint2    = UInt2;


    auto fn_add_aabb = [](const Float2x3& a, const Float2x3& b) {
        return Float2x3{luisa::compute::min(a[0], b[0]), luisa::compute::max(a[1], b[1])};
    };

    auto fn_refit_tree_aabb = device.compile<1>(
        [fn_add_aabb](const BufferVar<uint>     sa_parrent,
                      const BufferVar<uint2>    sa_children,
                      const BufferVar<aabbData> sa_node_aabb,
                      const BufferVar<uint>     sa_apply_flag,
                      const BufferVar<uint>     sa_is_healthy,
                      const Uint                num_inner_nodes)
        {
            const Uint lid     = dispatch_id().x;
            Uint       current = lid + num_inner_nodes;
            Uint       parrent = sa_parrent->read(current);
            Uint       loop    = 0;

            $while(parrent != -1)
            {
                luisa::compute::sync_block();
                std::atomic_thread_fence(std::memory_order_seq_cst);

                loop += 1;
                $if(loop > 10000)
                {
                    sa_is_healthy->write(0, 0u);
                    $break;
                };
                Uint orig_flag = sa_apply_flag->atomic(parrent).fetch_add(1u);
                $if(orig_flag == 0)
                {
                    $break;
                }
                $elif(orig_flag == 1)
                {
                    sa_apply_flag->atomic(parrent).fetch_add(1u);
                    Uint2 child_of_parrent = sa_children->read(parrent);

                    // Invalid
                    Float2x3 aabb_left    = sa_node_aabb->read(child_of_parrent[0]);
                    Float2x3 aabb_right   = sa_node_aabb->read(child_of_parrent[1]);
                    auto     parrent_aabb = fn_add_aabb(aabb_left, aabb_right);
                    sa_node_aabb->write(parrent, parrent_aabb);

                    // Invalid
                    // Float2x3 aabb_left;
                    // Float2x3 aabb_right;
                    // aabb_left[0][0] = sa_node_aabb->atomic(child_of_parrent[0])[0][0].fetch_min(1e30f);
                    // aabb_left[0][1] = sa_node_aabb->atomic(child_of_parrent[0])[0][1].fetch_min(1e30f);
                    // aabb_left[0][2] = sa_node_aabb->atomic(child_of_parrent[0])[0][2].fetch_min(1e30f);
                    // aabb_left[1][0] = sa_node_aabb->atomic(child_of_parrent[0])[1][0].fetch_max(-1e30f);
                    // aabb_left[1][1] = sa_node_aabb->atomic(child_of_parrent[0])[1][1].fetch_max(-1e30f);
                    // aabb_left[1][2] = sa_node_aabb->atomic(child_of_parrent[0])[1][2].fetch_max(-1e30f);

                    // aabb_right[0][0] = sa_node_aabb->atomic(child_of_parrent[1])[0][0].fetch_min(1e30f);
                    // aabb_right[0][1] = sa_node_aabb->atomic(child_of_parrent[1])[0][1].fetch_min(1e30f);
                    // aabb_right[0][2] = sa_node_aabb->atomic(child_of_parrent[1])[0][2].fetch_min(1e30f);
                    // aabb_right[1][0] = sa_node_aabb->atomic(child_of_parrent[1])[1][0].fetch_max(-1e30f);
                    // aabb_right[1][1] = sa_node_aabb->atomic(child_of_parrent[1])[1][1].fetch_max(-1e30f);
                    // aabb_right[1][2] = sa_node_aabb->atomic(child_of_parrent[1])[1][2].fetch_max(-1e30f);
                    // auto parrent_aabb = fn_add_aabb(aabb_left, aabb_right);
                    // sa_node_aabb->atomic(parrent)[0][0].fetch_min(aabb_left[0][0]);
                    // sa_node_aabb->atomic(parrent)[0][1].fetch_min(aabb_left[0][1]);
                    // sa_node_aabb->atomic(parrent)[0][2].fetch_min(aabb_left[0][2]);
                    // sa_node_aabb->atomic(parrent)[1][0].fetch_max(aabb_right[1][0]);
                    // sa_node_aabb->atomic(parrent)[1][1].fetch_max(aabb_right[1][1]);
                    // sa_node_aabb->atomic(parrent)[1][2].fetch_max(aabb_right[1][2]);

                    current = parrent;
                    parrent = sa_parrent->read(current);
                }
                $else
                {
                    sa_is_healthy->write(0, 0u);
                    $break;
                };
                // luisa::compute::sync_block();
                // std::atomic_thread_fence(std::memory_order_seq_cst);
            };
        });

    auto fn_refit_tree_aabb_v2 = device.compile<1>(
        [fn_add_aabb](const BufferVar<uint>           sa_parrent,
                      const BufferVar<uint2>          sa_children,
                      const BufferVar<CompressedAABB> sa_node_aabb,
                      //   const BufferVar<uint>     sa_apply_flag,
                      const BufferVar<uint> sa_is_healthy,
                      const Uint            num_inner_nodes)
        {
            const Uint lid     = dispatch_id().x;
            Uint       current = lid + num_inner_nodes;
            Uint       parrent = sa_parrent->read(current);
            Uint       loop    = 0;

            $while(parrent != -1)
            {
                loop += 1;
                $if(loop > 10000)
                {
                    sa_is_healthy->write(0, 0u);
                    $break;
                };
                Uint orig_flag = sa_node_aabb->atomic(parrent).flag1.fetch_add(1u);
                $if(orig_flag == 0)
                {
                    $break;
                }
                $elif(orig_flag == 1)
                {
                    sa_node_aabb->atomic(parrent).flag1.fetch_add(1u);
                    Uint2 child_of_parrent = sa_children->read(parrent);

                    auto wait_for_write = [&](Uint child_idx)
                    {
                        UInt iter = 0;
                        $while(true)
                        {
                            auto flag = sa_node_aabb->atomic(child_idx).flag2.compare_exchange(1u, 2u);
                            $if(flag == 1u)
                            {
                                $break;
                            };
                            $if(iter > 1000)
                            {
                                // sa_is_healthy->write(0, 0u);
                                device_log("Thread {} read last flag {}", dispatch_id().x, flag);
                                $break;
                            };
                            iter += 1;
                        };
                    };

                    wait_for_write(child_of_parrent[0]);
                    wait_for_write(child_of_parrent[1]);

                    auto aabb_left       = sa_node_aabb->read(child_of_parrent[0]);
                    auto aabb_right      = sa_node_aabb->read(child_of_parrent[1]);
                    auto parrent_aabb    = fn_add_aabb(aabb_left->get_aabb(), aabb_right->get_aabb());
                    auto parrent_aabb_v2 = make_compressed_aabb(parrent_aabb, 0u, 0u);
                    sa_node_aabb->write(parrent, parrent_aabb_v2);
                    sa_node_aabb.atomic(parrent).flag2.exchange(1u);

                    current = parrent;
                    parrent = sa_parrent->read(current);
                }
                $else
                {
                    sa_is_healthy->write(0, 0u);
                    $break;
                };
                // luisa::compute::sync_block();
                // std::atomic_thread_fence(std::memory_order_seq_cst);
            };
        });

    auto fn_test_with_depth = [&](const uint depth)
    {
        const uint num_leaves = 1 << depth;
        LUISA_INFO("Tree depth = {}, num_leaves = {}, Disire for root AABB = {}",
                   depth,
                   num_leaves,
                   float2x3{float3(0.f), float3(float(num_leaves))});
        const uint num_nodes       = num_leaves * 2 - 1;
        const uint num_inner_nodes = num_leaves - 1;

        Buffer<uint>     sa_apply_flag = device.create_buffer<uint>(num_nodes);
        Buffer<uint>     sa_parrent    = device.create_buffer<uint>(num_nodes);
        Buffer<uint2>    sa_children   = device.create_buffer<uint2>(num_nodes);
        Buffer<aabbData> sa_node_aabb  = device.create_buffer<aabbData>(num_nodes);
        Buffer<uint>     sa_is_healthy = device.create_buffer<uint>(1);


        luisa::vector<uint>     host_apply_flag(num_nodes, 0u);
        luisa::vector<uint>     host_parrent(num_nodes);
        luisa::vector<uint2>    host_children(num_nodes);
        luisa::vector<aabbData> host_node_aabb(num_nodes);
        luisa::vector<uint>     host_is_healthy(1, true);


        Buffer<CompressedAABB>        sa_node_aabb_v2 = device.create_buffer<CompressedAABB>(num_nodes);
        luisa::vector<CompressedAABB> host_node_aabb_v2(num_nodes);

        // Initialize a complete binary tree
        for (uint i = 0; i < num_leaves; i++)
        {
            host_parrent[i + num_inner_nodes] = (i + num_inner_nodes - 1) / 2;
        }

        for (uint i = 0; i < num_inner_nodes; i++)
        {
            host_parrent[i]  = (i - 1) / 2;
            host_children[i] = uint2{2 * i + 1, 2 * i + 2};
        }
        host_parrent[0] = -1u;

        // Initialize leaf and inner node aabb
        for (uint i = 0; i < num_leaves; i++)
        {
            auto aabb                              = float2x3{float3(float(i), float(i), float(i)),
                                 float3(float(i + 1), float(i + 1), float(i + 1))};
            host_node_aabb[i + num_inner_nodes]    = aabb;
            host_node_aabb_v2[i + num_inner_nodes] = make_compressed_aabb(aabb, 0u, 1u);
        }
        for (uint i = 0; i < num_inner_nodes; i++)
        {
            auto default_aabb    = float2x3{float3(1e30f, 1e30f, 1e30f), float3(-1e30f, -1e30f, -1e30f)};
            host_node_aabb[i]    = default_aabb;
            host_node_aabb_v2[i] = make_compressed_aabb(default_aabb, 0u, 0u);
        }


        stream << sa_parrent.copy_from(host_parrent.data());
        stream << sa_children.copy_from(host_children.data());
        stream << sa_node_aabb.copy_from(host_node_aabb.data());
        stream << sa_node_aabb_v2.copy_from(host_node_aabb_v2.data());
        stream << sa_is_healthy.copy_from(host_is_healthy.data());

        // stream << fn_refit_tree_aabb(sa_parrent.view(),
        //                              sa_children.view(),
        //                              sa_node_aabb.view(),
        //                              sa_apply_flag.view(),
        //                              sa_is_healthy.view(),
        //                              num_inner_nodes)
        //   .dispatch(num_leaves);
        // stream << sa_node_aabb.view(0, num_inner_nodes).copy_to(host_node_aabb.data());

        stream << fn_refit_tree_aabb_v2(
                      sa_parrent.view(), sa_children.view(), sa_node_aabb_v2.view(), sa_is_healthy.view(), num_inner_nodes)
                      .dispatch(num_leaves);
        stream << sa_node_aabb_v2.view(0, num_inner_nodes).copy_to(host_node_aabb_v2.data());


        stream << sa_is_healthy.copy_to(host_is_healthy.data());
        stream << luisa::compute::synchronize();

        if (!host_is_healthy[0])
        {
            LUISA_ERROR(".  Refit LBVH failed due to unhealthy tree.");
            return -1;
        }
        // auto root_aabb = host_node_aabb[0];
        auto root_aabb = host_node_aabb_v2[0].get_aabb();

        if (luisa::any(root_aabb[0] != float3(0.f))
            || luisa::any(root_aabb[1] != float3(float(num_leaves), float(num_leaves), float(num_leaves))))
        {
            LUISA_ERROR(".  Refit LBVH failed due to incorrect root aabb: [{}, {}]", root_aabb[0], root_aabb[1]);
            return -1;
        }

        LUISA_INFO(".  Refit LBVH successed. Root aabb: [{}, {}]", root_aabb[0], root_aabb[1]);
        return 0;
    };

    struct LBVH_DATA
    {
        Buffer<uint>   sa_sorted_get_original;
        Buffer<float3> sa_positions;
    };
    LBVH_DATA lbvh_data;
    lbvh_data.sa_sorted_get_original = device.create_buffer<uint>(1000);
    lbvh_data.sa_positions           = device.create_buffer<float3>(1000);

    auto fn_test_compile_refit = [&](const uint depth)
    {
        auto fn_update_vert_tree_leave_aabb = device.compile<1>(
            [sa_sorted_get_original =
                 lbvh_data.sa_sorted_get_original.view()](const Var<luisa::compute::BufferView<float3>> sa_x_start,
                                                          const Var<luisa::compute::BufferView<float3>> sa_x_end,
                                                          const Float thickness)
            {
                const Uint lid = dispatch_id().x;
                Uint       vid = sa_sorted_get_original->read(lid);
            });

        stream << fn_update_vert_tree_leave_aabb(lbvh_data.sa_positions, lbvh_data.sa_positions, 0.0f).dispatch(1000)
               << synchronize();
    };

    // const uint depth      = 20;
    const uint test_times = 20;
    // for (uint i = 1; i <= test_times; i++)
    // {
    //     const uint depth = i + 1;
    //     if (fn_test_with_depth(depth) != 0)
    //     {
    //         LUISA_ERROR("Test failed at depth = {}", depth);
    //         return -1;
    //     }
    // }
    for (uint i = 1; i <= test_times; i++)
    {
        const uint depth = i + 1;
        fn_test_compile_refit(depth);
    }

    return 0;
}