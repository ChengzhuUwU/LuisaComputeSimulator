#include "precond_cg.h"
#include "Core/scalar.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

namespace lcsv 
{

template<typename T>
void buffer_add(luisa::compute::BufferView<T> buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}
template<typename T>
void buffer_add(Var<luisa::compute::BufferView<T>>& buffer, const Var<uint> dest, const Var<T>& value)
{
    buffer->write(dest, buffer->read(dest) + value);
}

void ConjugateGradientSolver::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;
    
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};
    // auto& sa_cgX = sim_data->sa_cgX;
    // auto& sa_cgB = sim_data->sa_cgB;
    // auto& sa_cgA_diag = sim_data->sa_cgA_diag;
    // auto& sa_cgA_offdiag = sim_data->sa_cgA_offdiag;
    
    // auto& sa_cgMinv = sim_data->sa_cgMinv;
    // auto& sa_cgP = sim_data->sa_cgP;
    // auto& sa_cgQ = sim_data->sa_cgQ;
    // auto& sa_cgR = sim_data->sa_cgR;
    // auto& sa_cgZ = sim_data->sa_cgZ;

    fn_reset_float3 = device.compile<1>([](Var<luisa::compute::BufferView<float3>> buffer)
    {
        buffer->write(dispatch_id().x, luisa::compute::make_float3(0.0f));
    });
    fn_reset_float = device.compile<1>([](Var<luisa::compute::BufferView<float>> buffer)
    {
        buffer->write(dispatch_id().x, Float(0.0f));
    });
    fn_reset_uint = device.compile<1>([](Var<luisa::compute::BufferView<uint>> buffer)
    {
        buffer->write(dispatch_id().x, Uint(0u));
    });
    fn_apply_dx = device.compile<1>([sa_x = sim_data->sa_x.view(), sa_x_iter_start = sim_data->sa_x_iter_start.view(), sa_dx = sim_data->sa_cgX.view()](const Float alpha)
    {
        const UInt vid = dispatch_id().x;
        Float3 x_new = sa_x_iter_start->read(vid) + alpha * sa_dx->read(vid);
        sa_x->write(vid, x_new);
    });

    // 0 : old_dot_rr
    // 1 : new_dot_rz
    // 2 : alpha 
    // 3 : beta
    // 4 : new_dot_rr
    // 
    // 6 : init energy
    // 7 : new energy

    auto fn_save_dot_rr = [sa_convergence = sim_data->sa_convergence.view()](const Float dot_rr) 
    { 
        const Float normR = sqrt_scalar(dot_rr);
        // Save current rTr
        sa_convergence->write(4, normR);

        // Save current rTr to convergent list
        Uint iteration_idx = as<Uint>(sa_convergence->read(8));
        sa_convergence->write(10 + iteration_idx, normR); 
        sa_convergence->write(8, as<Float>(iteration_idx + 1));
    };
    auto fn_read_rz = [sa_convergence = sim_data->sa_convergence.view()]() 
    { 
        return sa_convergence->read(1);
    };
    auto fn_update_dot_rz = [sa_convergence = sim_data->sa_convergence.view()](const Float dot_rz) 
    { 
        sa_convergence->write(1, dot_rz);
    };

    auto fn_save_alpha = [sa_convergence = sim_data->sa_convergence.view(), fn_read_rz](const Float dot_pq) 
    { 
        Float delta = fn_read_rz();
        Float alpha = select(dot_pq == 0.0f, Float(0.0f), delta / dot_pq); // alpha = delta / dot(p, q)
        sa_convergence->write(2, alpha);
    };
    auto fn_read_alpha = [sa_convergence = sim_data->sa_convergence.view()]() 
    { 
        return sa_convergence->read(2);
    };
    
    auto fn_save_beta = [sa_convergence = sim_data->sa_convergence.view(), fn_read_rz](const Float dot_rz_old, const Float dot_rz) 
    { 
        // Float delta_old = fn_read_rz();
        Float delta_old = dot_rz_old;
        Float beta = select(delta_old == 0.0f, Float(0.0f), dot_rz / delta_old);
        sa_convergence->write(3, beta);
    };
    auto fn_read_beta = [sa_convergence = sim_data->sa_convergence.view()]() 
    { 
        return sa_convergence->read(3);
    };

    // PCG kernels
    fn_pcg_init = device.compile<1>(
        [
            sa_cgB = sim_data->sa_cgB.view(),
            sa_cgQ = sim_data->sa_cgQ.view(),
            sa_cgR = sim_data->sa_cgR.view(),
            sa_cgP = sim_data->sa_cgP.view(),
            sa_block_result = sim_data->sa_block_result.view()
        ]()
    {
        const UInt vid = dispatch_id().x;
        Float dot_rr = 0.0f;
        {
            Float3 b = sa_cgB->read(vid);
            Float3 q = sa_cgQ->read(vid);
            Float3 r = b - q;
            sa_cgR->write(vid, r);
            sa_cgP->write(vid, make_float3(0.0f));
            sa_cgQ->write(vid, make_float3(0.0f));
    
            dot_rr = dot_vec(r, r);
        };
        dot_rr = ParallelIntrinsic::block_intrinsic_reduce(vid, dot_rr, ParallelIntrinsic::warp_reduce_op_sum<float>);

        $if (vid % 256 == 0)
        {
            const Uint blockIdx = vid / 256;
            sa_block_result->write(blockIdx, dot_rr);
        };
    }, default_option);

    fn_pcg_init_second_pass = device.compile<1>(
        [
            sa_block_result = sim_data->sa_block_result.view(),
            sa_convergence = sim_data->sa_convergence.view()
        ]()
        {
            const UInt vid = dispatch_id().x;

            Float dot_rr = 0.0f;
            {
                dot_rr = sa_block_result->read(vid);
            };
            dot_rr = ParallelIntrinsic::block_intrinsic_reduce(vid, dot_rr, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if (vid == 0)
            {
                sa_convergence->write(0, dot_rr); // rTr_0
                sa_convergence->write(1, 0.0f); // rTz
                sa_convergence->write(2, 0.0f); // alpha
                sa_convergence->write(3, 0.0f); // beta

                sa_convergence->write(8, as<Float>(Uint(0))); // iteration count
                sa_convergence->write(9, dot_rr);
            };
        }
    );


    fn_dot_pq = device.compile<1>(
        [
            sa_cgP = sim_data->sa_cgP.view(),
            sa_cgQ = sim_data->sa_cgQ.view(),
            sa_block_result = sim_data->sa_block_result.view()
        ]()
        {
            const UInt vid = dispatch_id().x;

            Float dot_pq = 0.0f;
            {
                Float3 p = sa_cgP->read(vid);
                Float3 q = sa_cgQ->read(vid);
                dot_pq = dot_vec(p, q);
            };
            
            dot_pq = ParallelIntrinsic::block_intrinsic_reduce(vid, dot_pq, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if (vid % 256 == 0)
            {
                sa_block_result->write(vid / 256, dot_pq);
            };
        }
    );

    // Write 2 <- alpha
    fn_dot_pq_second_pass = device.compile<1>(
        [
            sa_block_result = sim_data->sa_block_result.view(),
            &fn_save_alpha
        ]()
        {
            const UInt vid = dispatch_id().x;

            Float dot_pq = 0.0f;
            {
                dot_pq = sa_block_result->read(vid);
            };
                
            dot_pq = ParallelIntrinsic::block_intrinsic_reduce(vid, dot_pq, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if (vid == 0) { fn_save_alpha(dot_pq); };
        }
    );



    fn_pcg_update_p = device.compile<1>(
        [
        sa_cgP = sim_data->sa_cgP.view(),
        sa_cgZ = sim_data->sa_cgZ.view(),
        sa_convergence = sim_data->sa_convergence.view(),
        &fn_read_beta
        ]()
    {
        const UInt vid = dispatch_id().x;
        const Float beta = fn_read_beta();
        const Float3 p = sa_cgP->read(vid);
        sa_cgP->write(vid, sa_cgZ->read(vid) + beta * p);
    }, default_option);

    fn_pcg_step = device.compile<1>(
        [
        sa_cgX = sim_data->sa_cgX.view(),
        sa_cgR = sim_data->sa_cgR.view(),
        sa_cgP = sim_data->sa_cgP.view(),
        sa_cgQ = sim_data->sa_cgQ.view(),
        &fn_read_alpha
        ]()
    {
        const UInt vid = dispatch_id().x;
        const Float alpha = fn_read_alpha();
        sa_cgX->write(vid, sa_cgX->read(vid) + alpha * sa_cgP->read(vid));
        sa_cgR->write(vid, sa_cgR->read(vid) - alpha * sa_cgQ->read(vid));
    }, default_option);



    // Preconditioner
    fn_pcg_make_preconditioner = device.compile<1>(
        [
            sa_cgA_diag = sim_data->sa_cgA_diag.view(),
            sa_cgMinv = sim_data->sa_cgMinv.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view()
        ]()
    {
        const UInt vid = dispatch_id().x;
        Float3x3 diagA = sa_cgA_diag->read(vid);
        Float3x3 inv_M = inverse(diagA);
        sa_cgMinv->write(vid, inv_M);
    }, default_option);

    fn_pcg_apply_preconditioner = device.compile<1>(
        [
            sa_cgR = sim_data->sa_cgR.view(),
            sa_cgZ = sim_data->sa_cgZ.view(),
            sa_cgMinv = sim_data->sa_cgMinv.view(),
            sa_block_result = sim_data->sa_block_result.view()
        ]() 
    {
        const UInt vid = dispatch_id().x;
        const Float3 r = sa_cgR->read(vid);
        const Float3x3 inv_M = sa_cgMinv->read(vid);
        Float3 z = inv_M * r;
        sa_cgZ->write(vid, z);

        Float dot_rz = dot_vec(r, z);
        Float dot_rr = dot_vec(r, r);
        Float2 dot_rr_rz = makeFloat2(dot_rr, dot_rz);
        dot_rr_rz = ParallelIntrinsic::block_intrinsic_reduce(vid, dot_rr_rz, ParallelIntrinsic::warp_reduce_op_sum<float2>);
        $if (vid % 256 == 0)
        {
            const Uint blockIdx = vid / 256;
            sa_block_result->write(2 * blockIdx + 0, dot_rr_rz[0]);
            sa_block_result->write(2 * blockIdx + 1, dot_rr_rz[1]);
        };
    }, default_option);

    // Write 1 <- dot_rz (replace)
    // Write 3 <- beta
    fn_pcg_apply_preconditioner_second_pass = device.compile<1>(
        [
            sa_block_result = sim_data->sa_block_result.view(),
            &fn_update_dot_rz, &fn_save_dot_rr, &fn_save_beta, &fn_read_rz
        ]()
        {
            const UInt vid = dispatch_id().x;

            Float dot_rr = sa_block_result->read(2 * vid + 0);
            Float dot_rz = sa_block_result->read(2 * vid + 1);
            Float2 dot_rr_rz = makeFloat2(dot_rr, dot_rz);

            dot_rr_rz = ParallelIntrinsic::block_intrinsic_reduce(vid, dot_rr_rz, ParallelIntrinsic::warp_reduce_op_sum<float2>);

            $if (vid == 0)
            {
                dot_rr = dot_rr_rz[0];
                dot_rz = dot_rr_rz[1];

                const Float dot_rz_old = fn_read_rz();
                fn_save_beta(dot_rz_old, dot_rz);
                fn_update_dot_rz(dot_rz);
                fn_save_dot_rr(dot_rr);
            };
        }
    );
}
void ConjugateGradientSolver::host_solve(
    luisa::compute::Stream& stream, 
    std::function<void(const std::vector<float3>&, std::vector<float3>&)> func_spmv,
    std::function<double(const std::vector<float3>&, const std::vector<float3>&)> func_compute_energy
)
{
    std::vector<float3>& sa_cgX = host_sim_data->sa_cgX;
    std::vector<float3>& sa_cgB = host_sim_data->sa_cgB;
    std::vector<float3x3>& sa_cgA_diag = host_sim_data->sa_cgA_diag;
    std::vector<float3x3>& sa_cgA_offdiag = host_sim_data->sa_cgA_offdiag; // Row-major for simplier SpMV
 
    std::vector<float3x3>& sa_cgMinv = host_sim_data->sa_cgMinv;
    std::vector<float3>& sa_cgP = host_sim_data->sa_cgP;
    std::vector<float3>& sa_cgQ = host_sim_data->sa_cgQ;
    std::vector<float3>& sa_cgR = host_sim_data->sa_cgR;
    std::vector<float3>& sa_cgZ = host_sim_data->sa_cgZ;
    
    std::vector<float3>& sa_x = host_sim_data->sa_x;
    std::vector<float3>& sa_x_iter_start = host_sim_data->sa_x_iter_start;
    std::vector<float3>& sa_x_tilde = host_sim_data->sa_x_tilde;
    auto host_apply_dx = [&](const float alpha)
    {
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            sa_x[vid] = sa_x_iter_start[vid] + alpha * sa_cgX[vid];
        });
    };
    
    auto fast_dot = [](const std::vector<float3>& left_ptr, const std::vector<float3>& right_ptr) -> float
    {
        return CpuParallel::parallel_for_and_reduce_sum<float>(0, left_ptr.size(), [&](const uint vid)
        {
            return luisa::dot(left_ptr[vid], right_ptr[vid]);
        });
    };
    auto fast_norm = [](const std::vector<float3>& ptr) -> float
    {
        float tmp = CpuParallel::parallel_for_and_reduce_sum<float>(0, ptr.size(), [&](const uint vid)
        {
            return luisa::dot(ptr[vid], ptr[vid]);
        });
        return sqrt(tmp);
    };
    auto fast_infinity_norm = [](const std::vector<float3>& ptr) -> float // Min value in array
    {
        return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
        {
            return luisa::length(ptr[vid]);
        }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
    };
    
    auto get_dot_rz_rr = [&]() -> float2 // [0] = r^T z, [1] = r^T r
    {
        return CpuParallel::parallel_for_and_reduce_sum<float2>(0, sa_cgR.size(), [&](const uint vid) -> float2
        {
            float3 r = sa_cgR[vid];
            float3 z = sa_cgZ[vid];
            return luisa::make_float2(luisa::dot(r, z), luisa::dot(r, r));
        });
    };
    auto read_beta = [](const uint vid, std::vector<float>& sa_converage) -> float
    {
        float delta_old = sa_converage[0];
        float delta = sa_converage[2];
        float beta = delta_old == 0.0f ? 0.0f : delta / delta_old;
        if (vid == 0)  
        { 
            sa_converage[1] = 0; 
            uint iteration_idx = uint(sa_converage[8]);
            sa_converage[9 + iteration_idx] = delta;
            sa_converage[8] = float(iteration_idx + 1); 
        }
        return beta;
    };
    auto save_dot_pq = [](const uint blockIdx, std::vector<float>& sa_converage, const float dot_pq) -> void
    {
        sa_converage[1] = dot_pq; /// <= reduce
        if (blockIdx == 0)
        {
            float delta_old = sa_converage[2];
            float delta_old_old = sa_converage[0];
            sa_converage[2] = 0;
            sa_converage[0] = delta_old;
            sa_converage[4] = delta_old_old;
        }
    };
    auto read_alpha = [](std::vector<float>& sa_converage) -> float
    {
        float delta = sa_converage[0];
        float dot_pq = sa_converage[1];
        float alpha = dot_pq == 0.0f ? 0.0f : delta / dot_pq;
        return alpha;
    };
    auto save_dot_rz = [](const uint blockIdx, std::vector<float>& sa_converage, const float dot_rz) -> void
    {
        sa_converage[2] = dot_rz; /// <= reduce
    };

    const uint num_verts = mesh_data->num_verts;
    
    auto pcg_make_preconditioner_jacobi = [&]()
    {
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();

        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            float3x3 diagA = sa_cgA_diag[vid];
            float3x3 inv_M = luisa::inverse(diagA);

            // Not available
            // const bool is_fixed = sa_is_fixed[vid];
            // if (is_fixed)
            // {
            //     inv_M = luisa::make_float3x3(0.0f);
            // }

            // float3x3 inv_M = luisa::make_float3x3(
            //     luisa::make_float3(1.0f / diagA[0][0], 0.0f, 0.0f), 
            //     luisa::make_float3(0.0f, 1.0f / diagA[1][1], 0.0f), 
            //     luisa::make_float3(0.0f, 0.0f, 1.0f / diagA[2][2])
            // );
            sa_cgMinv[vid] = inv_M;
        });
    };
    auto pcg_apply_preconditioner_jacobi = [&]()
    {
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const float3 r = sa_cgR[vid];
            const float3x3 inv_M = sa_cgMinv[vid];
            float3 z = inv_M * r;
            sa_cgZ[vid] = z;
        });
    };

    auto pcg_init = [&]()
    {
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const float3 b = sa_cgB[vid];
            const float3 q = sa_cgQ[vid];
            const float3 r = b - q;  // r = b - q = b - A * x
            sa_cgR[vid] = r;
            sa_cgP[vid] = Zero3;
            sa_cgQ[vid] = Zero3;
        });
    };
    auto pcg_update_p = [&](const float beta)
    {
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            const float3 p = sa_cgP[vid];
            sa_cgP[vid] = sa_cgZ[vid] + beta * p;
        });
    };
    auto pcg_step = [&](const float alpha)
    {
        CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
        {
            sa_cgX[vid] = sa_cgX[vid] + alpha * sa_cgP[vid];
            sa_cgR[vid] = sa_cgR[vid] - alpha * sa_cgQ[vid];
        });
    };

    auto& sa_convergence = host_mesh_data->sa_pcg_convergence;
    std::fill(sa_convergence.begin(), sa_convergence.end(), 0.0f);

    // func_spmv(sa_cgX, sa_cgQ);
    CpuParallel::parallel_set(sa_cgQ, makeFloat3(0.0f));

    pcg_init();
    
    pcg_make_preconditioner_jacobi();

    float normR_0 = 0.0f;
    float normR = 0.0f;

    uint iter = 0;
    for (iter = 0; iter < lcsv::get_scene_params().pcg_iter_count; iter++)
    {
        lcsv::get_scene_params().current_pcg_it = iter;

        // if (get_scene_params().print_system_energy)
        // {
        //     update_position_for_energy();
        //     compute_system_energy(it);
        //     compute_pcg_residual(it);
        // }

        pcg_apply_preconditioner_jacobi();

        float2 dot_rr_rz = get_dot_rz_rr(); 
        float dot_rz = dot_rr_rz[0];
        normR = std::sqrt(dot_rr_rz[1]); if (iter == 0) normR_0 = normR;
        save_dot_rz(0, sa_convergence, dot_rz);

        if (luisa::isnan(dot_rz) || luisa::isinf(dot_rz)) { luisa::log_error("Exist NAN/INF in PCG iteration"); exit(0); }
        if (normR < 5e-3 * normR_0 || dot_rz == 0.0f) 
        {
            break;
        }

        const float beta = read_beta(0, sa_convergence);
        pcg_update_p(beta);
    
        func_spmv(sa_cgP, sa_cgQ);
        float dot_pq = fast_dot(sa_cgP, sa_cgQ);
        save_dot_pq(0, sa_convergence, dot_pq);   
        
        const float alpha = read_alpha(sa_convergence);

        luisa::log_info("   In pcg iter {:3} : rTr = {}, beta = {}, alpha = {}", 
                iter, normR, beta, alpha);
        
        pcg_step(alpha);
    }

    constexpr bool print_energy = true; double curr_energy = 0.0;
    if constexpr (print_energy)
    {
        host_apply_dx(1.0f);
        curr_energy = func_compute_energy(sa_x, sa_x_iter_start);
    }
    const float infinity_norm = fast_infinity_norm(host_sim_data->sa_cgX);
    if (luisa::isnan(infinity_norm) || luisa::isinf(infinity_norm))
    {
        luisa::log_error("cgX exist NAN/INF value : {}", infinity_norm);
    }
    luisa::log_info("  In non-linear iter {:2}, PCG : iter-count = {:3}, rTr error = {:6.5f}, max_element(p) = {:6.5f}, energy = {}", 
        get_scene_params().current_nonlinear_iter,
        iter, normR / normR_0, infinity_norm, print_energy ? luisa::format("{:6.5f}", curr_energy) : ""
    );
            
    /*
    for (uint iter = 0; iter < lcsv::get_scene_params().pcg_iter_count; iter++)
    {
        lcsv::get_scene_params().current_pcg_it = iter;
        pcg_apply_preconditioner_jacobi();
        delta_old = delta;
        float2 dot_rr_rz = get_dot_rz_rr(); delta = dot_rr_rz[0];
        float normR = std::sqrt(dot_rr_rz[1]); if (iter == 0) normR_0 = normR;
        float beta = delta_old == 0.0f ? 0.0f : delta / delta_old;
        pcg_update_p(beta);
        pcg_spmv(sa_cgP, sa_cgQ);
        float dot_pq = fast_dot(sa_cgP, sa_cgQ);
        if (normR < 5e-3 * normR_0) 
        {
            break;
        }
        const float alpha = dot_pq == 0.0f ? 0.0f : delta / dot_pq;
        pcg_step(alpha);   
    }
    */
}
void ConjugateGradientSolver::device_solve( // TODO: input sa_x
    luisa::compute::Stream& stream, 
    std::function<void(const luisa::compute::Buffer<float3>&, luisa::compute::Buffer<float3>&)> func_spmv,
    std::function<double(const luisa::compute::Buffer<float3>&, const luisa::compute::Buffer<float3>&)> func_compute_energy
)
{
    auto host_infinity_norm = [](const std::vector<float3>& ptr) -> float // Min value in array
    {
        return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
        {
            return luisa::length(ptr[vid]);
        }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
    };
    
    std::vector<float3>& host_x = host_sim_data->sa_x;
    std::vector<float3>& host_x_iter_start = host_sim_data->sa_x_iter_start;
    std::vector<float3>& host_x_tilde = host_sim_data->sa_x_tilde;
    std::vector<float3>& host_cgX = host_sim_data->sa_cgX;

    auto host_apply_dx = [&](const float alpha)
    {
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            host_x[vid] = host_x_iter_start[vid] + alpha * host_cgX[vid];
        });
    };
    auto device_apply_dx = [&](const float alpha)
    {
        stream << fn_apply_dx(alpha).dispatch(mesh_data->num_verts);
    };
    
    // auto device_pcg = [&]()
    const uint num_verts = host_mesh_data->num_verts;
    const uint num_edges = host_mesh_data->num_edges;
    const uint num_faces = host_mesh_data->num_faces;
    const uint num_blocks_verts = get_dispatch_block(num_verts, 256);

    stream 
        << fn_reset_float(sim_data->sa_convergence).dispatch(sim_data->sa_convergence.size())
        ;

    // pcg_spmv(sim_data->sa_cgX, sim_data->sa_cgQ);

    stream 
        // << sim_data->sa_cgR.copy_from(sim_data->sa_cgB) // Cause cgX is set to zero...
        // << mp_buffer_filler->fill(device, sim_data->sa_cgQ, makeFloat3(0.0f))
        << fn_reset_float3(sim_data->sa_cgQ).dispatch(num_verts)
        << fn_pcg_init().dispatch(num_verts) 
        << fn_pcg_init_second_pass().dispatch(num_blocks_verts)
        << fn_pcg_make_preconditioner().dispatch(num_verts)
        
        // << sim_data->sa_convergence.copy_to(host_sim_data->sa_convergence.data())
        // << sim_data->sa_cgB.copy_to(host_sim_data->sa_cgB.data())
        // << sim_data->sa_cgR.copy_to(host_sim_data->sa_cgR.data())
        // << sim_data->sa_cgP.copy_to(host_sim_data->sa_cgP.data())
        // << luisa::compute::synchronize();
        ;
    
    // luisa::log_info("   PCG init info: rTr = {} / {}, bTb = {}, pTp = {}", 
    //     host_norm(host_sim_data->sa_cgR), host_sim_data->sa_convergence[4],
    //     host_norm(host_sim_data->sa_cgB),
    //     host_norm(host_sim_data->sa_cgP)
    // );

    float normR_0 = 0.0f;
    float normR = 0.0f; float beta = 0.0f; float alpha = 0.0f;

    uint iter = 0;
    for (iter = 0; iter < lcsv::get_scene_params().pcg_iter_count; iter++)
    {
        lcsv::get_scene_params().current_pcg_it = iter;

        stream 
            << fn_pcg_apply_preconditioner().dispatch(num_verts)
            << fn_pcg_apply_preconditioner_second_pass().dispatch(num_blocks_verts) // Compute beta
            << sim_data->sa_convergence.view(4, 1).copy_to(&normR)
            << luisa::compute::synchronize();

        // 0 : old_dot_rr
        // 1 : new_dot_rz
        // 2 : alpha 
        // 3 : beta
        // 4 : new_dot_rr
        // 
        // 6 : init energy
        // 7 : new energy

        if (iter == 0) normR_0 = normR;
        if (normR < 5e-3 * normR_0) 
        {
            break;
        }

        stream 
            << fn_pcg_update_p().dispatch(num_verts);

        func_spmv(sim_data->sa_cgP, sim_data->sa_cgQ);
        stream 
            << fn_dot_pq().dispatch(num_verts)
            << fn_dot_pq_second_pass().dispatch(num_blocks_verts) // Compute alpha

            // << sim_data->sa_cgB.copy_to(host_sim_data->sa_cgB.data())
            // << sim_data->sa_cgP.copy_to(host_sim_data->sa_cgP.data())
            // << sim_data->sa_cgQ.copy_to(host_sim_data->sa_cgQ.data())
            // << sim_data->sa_cgR.copy_to(host_sim_data->sa_cgR.data())
            // << sim_data->sa_cgZ.copy_to(host_sim_data->sa_cgZ.data())
            // << sim_data->sa_convergence.view(2, 1).copy_to(&alpha)
            // << sim_data->sa_convergence.view(3, 1).copy_to(&beta)

            << fn_pcg_step().dispatch(num_verts)

            // << luisa::compute::synchronize()
            ;

        // luisa::log_info("   In pcg iter {:3} : bTb = {}, sqrt(rTr) = {}, beta = {}, alpha = {}, pTq = {}, rTz = {}", 
        //         iter, 
        //         host_dot(host_sim_data->sa_cgB, host_sim_data->sa_cgB),
        //         normR, beta, alpha,
        //         host_dot(host_sim_data->sa_cgP, host_sim_data->sa_cgQ),
        //         host_dot(host_sim_data->sa_cgR, host_sim_data->sa_cgZ) );
    }

    stream 
        << sim_data->sa_cgX.copy_to(host_sim_data->sa_cgX.data())
        << luisa::compute::synchronize();

    constexpr bool print_energy = true; double curr_energy = 0.0;
    if constexpr (print_energy)
    {
        host_apply_dx(1.0f);
        curr_energy = func_compute_energy(sim_data->sa_x, sim_data->sa_x_tilde);
    }
    luisa::log_info("  In non-linear iter {:2}, PCG : iter-count = {:3}, rTr error = {:6.5f}, max_element(p) = {:6.5f}, energy = {}", 
        get_scene_params().current_nonlinear_iter,
        iter, normR / normR_0, host_infinity_norm(host_sim_data->sa_cgX), print_energy ? luisa::format("{:6.5f}", curr_energy) : ""
    );
}

using EigenFloat3x3 = Eigen::Matrix<float, 3, 3>;
using EigenFloat6x6 = Eigen::Matrix<float, 6, 6>;
using EigenFloat9x9 = Eigen::Matrix<float, 9, 9>;
using EigenFloat12x12 = Eigen::Matrix<float, 12, 12>;
using EigenFloat3   = Eigen::Matrix<float, 3, 1>;
using EigenFloat4   = Eigen::Matrix<float, 4, 1>;

static inline EigenFloat3 float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
static inline EigenFloat4 float4_to_eigen4(const float4& input) { EigenFloat4 vec; vec << input[0], input[1], input[2], input[3]; return vec; };
static inline float3 eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };
static inline float4 eigen4_to_float4(const EigenFloat4& input) { return luisa::make_float4(input(0, 0), input(1, 0), input(2, 0), input(3, 0)); };

void ConjugateGradientSolver::eigen_solve(
    const Eigen::SparseMatrix<float>& eigen_cgA, 
    Eigen::VectorXf& eigen_cgX,
    const Eigen::VectorXf& eigen_cgB, 
    std::function<double(const std::vector<float3>&, const std::vector<float3>&)> func_compute_energy
)
{
    std::vector<float3>& host_x = host_sim_data->sa_x;
    std::vector<float3>& host_x_iter_start = host_sim_data->sa_x_iter_start;
    std::vector<float3>& host_x_tilde = host_sim_data->sa_x_tilde;
    std::vector<float3>& host_cgX = host_sim_data->sa_cgX;
    auto host_apply_dx = [&](const float alpha)
    {
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {
            host_x[vid] = host_x_iter_start[vid] + alpha * host_cgX[vid];
        });
    };
    auto fast_infinity_norm = [](const std::vector<float3>& ptr) -> float // Min value in array
    {
        return CpuParallel::parallel_for_and_reduce(0, ptr.size(), [&](const uint vid)
        {
            return luisa::length(ptr[vid]);
        }, [](const float left, const float right) { return max_scalar(left, right); }, -1e9f); 
    };

    constexpr bool print_energy = true; double curr_energy = 0.0;

    auto eigen_iter_solve = [&]()
    {
        // Solve cgA * dx = cg_b_vec for dx using Conjugate Gradient
        Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower> solver; // Eigen::IncompleteCholesky<float>

        // solver.setMaxIterations(128);
        solver.setTolerance(1e-2f);
        solver.compute(eigen_cgA);

        // 计算Jacobi预条件子的对角线逆
        // Eigen::VectorXf eigen_cgR = eigen_cgB - eigen_cgA * eigen_cgX;
        // Eigen::VectorXf eigen_cgM_inv(eigen_cgR.rows());
        // for (int i = 0; i < eigen_cgR.rows(); ++i) {
        //     float diag = eigen_cgA.coeff(i, i);
        //     eigen_cgM_inv[i] = (std::abs(diag) > 1e-12f) ? (1.0f / diag) : 0.0f;
        // }
        // Eigen::VectorXf eigen_cgZ = eigen_cgR.cwiseProduct(eigen_cgM_inv);
        // Eigen::VectorXf eigen_cgQ = eigen_cgA * eigen_cgZ;
        // luisa::log_info("initB = {}, initR = {}, initM = {}, initZ = {}, initQ = {}",
        //     eigen_cgB.norm(), eigen_cgR.norm(), eigen_cgM_inv.norm(), eigen_cgZ.norm(), eigen_cgQ.norm());

        solver._solve_impl(eigen_cgB, eigen_cgX);
        if (solver.info() != Eigen::Success) { luisa::log_error("Eigen: Solve failed in {} iterations", solver.iterations()); }
        else 
        {
            CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
            {
                host_cgX[vid] = eigen3_to_float3(eigen_cgX.segment<3>(3 * vid));
            });
            if constexpr (print_energy)
            {
                host_apply_dx(1.0f);
                curr_energy = func_compute_energy(host_sim_data->sa_x, host_sim_data->sa_x_tilde);
            }

            luisa::log_info("  In non-linear iter {:2}, Eigen-PCG : iter-count = {}, rTr error = {:6.5f}, max_element(p) = {:6.5f}, energy = {:6.5f}", 
                get_scene_params().current_nonlinear_iter, solver.iterations(),
                solver.error(), fast_infinity_norm(host_cgX), curr_energy); // from normR_0 -> normR
        }
    };
    auto eigen_decompose_solve = [&]()
    {
        // Solve cgA * dx = cg_b_vec for dx using SimplicialLDLT decomposition
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
        solver.compute(eigen_cgA);
        if (solver.info() != Eigen::Success)
        {
            luisa::log_error("Eigen: SimplicialLDLT decomposition failed!");
            return;
        }
        solver._solve_impl(eigen_cgB, eigen_cgX);
        if (solver.info() != Eigen::Success)
        {
            luisa::log_error("Eigen: SimplicialLDLT solve failed!");
            return;
        }
        else
        {
            float error = (eigen_cgB - eigen_cgA * eigen_cgX).norm();
            CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
            {
                host_cgX[vid] = eigen3_to_float3(eigen_cgX.segment<3>(3 * vid));
            });
            if constexpr (print_energy)
            {
                host_apply_dx(1.0f);
                curr_energy = func_compute_energy(host_sim_data->sa_x, host_sim_data->sa_x_tilde);
            }

            luisa::log_info("  In non-linear iter {:2}, Eigen-Decompose : rTr error = {:6.5f}, max_element(p) = {:6.5f}, energy = {:6.5f}", 
                get_scene_params().current_nonlinear_iter, 
                error, fast_infinity_norm(host_cgX), curr_energy); // from normR_0 -> normR
        }
    };

    eigen_iter_solve();

}


} // namespace lcsv 