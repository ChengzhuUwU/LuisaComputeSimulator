
#include <luisa/luisa-compute.h>

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
            host_node_aabb[i + num_inner_nodes] = aabbData{float3(float(i), float(i), float(i)),
                                                           float3(float(i + 1), float(i + 1), float(i + 1))};
        }
        for (uint i = 0; i < num_inner_nodes; i++)
        {
            host_node_aabb[i] = aabbData{float3(1e30f, 1e30f, 1e30f), float3(-1e30f, -1e30f, -1e30f)};
        }


        stream << sa_parrent.copy_from(host_parrent.data());
        stream << sa_children.copy_from(host_children.data());
        stream << sa_node_aabb.copy_from(host_node_aabb.data());
        stream << sa_is_healthy.copy_from(host_is_healthy.data());

        stream << fn_refit_tree_aabb(sa_parrent.view(),
                                     sa_children.view(),
                                     sa_node_aabb.view(),
                                     sa_apply_flag.view(),
                                     sa_is_healthy.view(),
                                     num_inner_nodes)
                      .dispatch(num_leaves);

        stream << sa_node_aabb.view(0, num_inner_nodes).copy_to(host_node_aabb.data());
        stream << sa_is_healthy.copy_to(host_is_healthy.data());
        stream << luisa::compute::synchronize();
        ;
        if (!host_is_healthy[0])
        {
            LUISA_ERROR(".  Refit LBVH failed due to unhealthy tree.");
            return -1;
        }
        if (luisa::any(host_node_aabb[0][0] != float3(0.f))
            || luisa::any(host_node_aabb[0][1] != float3(float(num_leaves), float(num_leaves), float(num_leaves))))
        {
            LUISA_ERROR(".  Refit LBVH failed due to incorrect root aabb: [{}, {}]",
                        host_node_aabb[0][0],
                        host_node_aabb[0][1]);
            return -1;
        }

        LUISA_INFO(".  Refit LBVH successed. Root aabb: [{}, {}]", host_node_aabb[0][0], host_node_aabb[0][1]);
        return 0;
    };

    // const uint depth      = 20;
    const uint test_times = 10;
    for (uint i = 1; i <= test_times; i++)
    {
        const uint depth = i + 1;
        if (fn_test_with_depth(depth) != 0)
        {
            LUISA_ERROR("Test failed at depth = {}", depth);
            return -1;
        }
    }

    return 0;
}