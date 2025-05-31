#include "SimulationSolver/descent_solver.h"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include <luisa/dsl/sugar.h>

struct TestTypeAAA
{
    luisa::float3 a;
    luisa::float3 b;
    luisa::float3 c;
    // [[nodiscard]] friend auto operator*(const luisa::compute::Var<TestTypeAAA> left, const luisa::compute::Var<TestTypeAAA> right) noexcept { return left.a * right.a + left.b * right.b + left.c * right.c; }
    // [[nodiscard]] friend auto operator+(const luisa::compute::Var<TestTypeAAA> left, const luisa::compute::Var<TestTypeAAA> right) noexcept { return left.a + right.a + left.b + right.b + left.c + right.c; }
    // [[nodiscard]] friend auto operator-(const luisa::compute::Var<TestTypeAAA> left, const luisa::compute::Var<TestTypeAAA> right) noexcept { return left.a - right.a + left.b - right.b + left.c - right.c; }
    // [[nodiscard]] friend auto operator/(const luisa::compute::Var<TestTypeAAA> left, const luisa::compute::Var<TestTypeAAA> right) noexcept { return left.a / right.a + left.b / right.b + left.c / right.c; }
};
LUISA_STRUCT(TestTypeAAA, a, b, c) {};
[[nodiscard]] auto operator*(const luisa::compute::Var<TestTypeAAA>& left, const luisa::compute::Var<TestTypeAAA>& right) noexcept 
{ 
    return left.a * right.a + left.b * right.b + left.c * right.c; 
}

namespace lcsv {
using lcsv::get_scene_params;

void DescentSolver::test_luisa()
{

    using namespace luisa::compute;

    auto makeTypeAAA = [&](const Float3& a, const Float3& b, const Float3& c)
    {
        // Float3x3 left1 = make_float3x3(1.0f);
        // Float3x3 right1 = make_float3x3(2.0f);
        // return left1 * right1;
        auto left = Var<TestTypeAAA>{a, b, c};;
        auto right = Var<TestTypeAAA>{a, b, c};;
        return left * right;
    };

    // Float3 vec1 = make_float3(0.0f);
    // auto left = Var<TestTypeAAA>{vec1, vec1, vec1};;
    // auto right = Var<TestTypeAAA>{vec1, vec1, vec1};;
    // auto value = left * right;
    // luisa::log_info("print left {} {} {}", left.a, left.b, left.c);
    // luisa::log_info("print right {} {} {}", right.a, right.b, right.c);
    // luisa::log_info("print mat1 {} {} {}", value.x, value.y, value.z);

    // Float3 vec1 = make_float3(0.0f);
    // Float4 vec2 = make_float4(0.0f);
    // auto mat1 = Float4x3{vec1, vec1, vec1, vec1};
    // auto mat2 = Float3x4{vec2, vec2, vec2};
    // // auto result = mat1 * mat2;
    // luisa::log_info("print mat1 {}", vec1);
    // luisa::log_info("print mat2 {}", vec2);

}
void DescentSolver::reset_constrains(luisa::compute::Stream& stream)
{
    


    // auto fn_reset_template = [&](Buffer<float>& buffer)
    // {
    //     parallel_set(
    //         buffer.data(), 
    //         buffer.size(), 
    //         0.0f);
    // };

    

    // fn_reset_template(xpbd_data->sa_lambda_stretch_mass_spring);
    // fn_reset_template(xpbd_data->sa_lambda_bending);
}
void DescentSolver::reset_collision_constrains(luisa::compute::Stream& stream)
{

}




void DescentSolver::compile(luisa::compute::Device& device)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    auto makeHf1 = [](const Float3& force, const Float3x3& hessian) 
    {
        return makeFloat4x3(force, hessian[0], hessian[1], hessian[2]);
    };
    auto extractHf1 = [&](Float3& force, Float3x3& hessian, BufferView<float4x3> sa_Hf, const Uint vid)
    {
        Float4x3 hf = sa_Hf->read(vid);
        force = hf.cols[0];
        hessian[0] = hf.cols[1];
        hessian[1] = hf.cols[2];
        hessian[2] = hf.cols[3];
    };

    auto writeHf = [](const Float3& force, const Float3x3& hessian, BufferView<float> sa_Hf, const Uint vid) 
    {
        const Uint offset = vid * 12;
        sa_Hf->write(offset + 0, force[0]);
        sa_Hf->write(offset + 1, force[1]);
        sa_Hf->write(offset + 2, force[2]);
        sa_Hf->write(offset + 3, hessian[0][0]);
        sa_Hf->write(offset + 4, hessian[0][1]);
        sa_Hf->write(offset + 5, hessian[0][2]);
        sa_Hf->write(offset + 6, hessian[1][0]);
        sa_Hf->write(offset + 7, hessian[1][1]);
        sa_Hf->write(offset + 8, hessian[1][2]);
        sa_Hf->write(offset + 9, hessian[2][0]);
        sa_Hf->write(offset + 10, hessian[2][1]);
        sa_Hf->write(offset + 11, hessian[2][2]);
    };
    auto extractHf = [](Float3& force, Float3x3& hessian, BufferView<float> sa_Hf, const Uint vid)
    {
        const Uint offset = vid * 12;
        force[0] = sa_Hf->read(offset + 0);
        force[1] = sa_Hf->read(offset + 1);
        force[2] = sa_Hf->read(offset + 2);
        hessian[0][0] = sa_Hf->read(offset + 3);
        hessian[0][1] = sa_Hf->read(offset + 4);
        hessian[0][2] = sa_Hf->read(offset + 5);
        hessian[1][0] = sa_Hf->read(offset + 6);
        hessian[1][1] = sa_Hf->read(offset + 7);
        hessian[1][2] = sa_Hf->read(offset + 8);
        hessian[2][0] = sa_Hf->read(offset + 9);
        hessian[2][1] = sa_Hf->read(offset + 10);
        hessian[2][2] = sa_Hf->read(offset + 11);
    };
    
    fn_predict_position = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_v = sim_data->sa_v.view(),
            sa_x_start = sim_data->sa_x_step_start.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view()
        ](const Float substep_dt)
    {
        const UInt vid = dispatch_id().x;
        const Float3 gravity(0, -9.8f, 0);
        Float3 x_prev = sa_x_start->read(vid);
        Float3 v_prev = sa_v->read(vid);
        Float3 outer_acceleration = gravity;
        Float3 v_pred = v_prev + substep_dt * outer_acceleration;
        $if (sa_is_fixed->read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
        const Float3 x_pred = x_prev + substep_dt * v_pred;
        sa_x->write(vid, x_pred);
        sa_x_tilde->write(vid, x_pred);
    }, default_option);

    fn_update_velocity = device.compile<1>(
        [
            sa_x = sim_data->sa_x.view(),
            sa_v = sim_data->sa_v.view(),
            sa_iter_start_position = sim_data->sa_x_step_start.view(),
            sa_iter_position = sim_data->sa_x.view(),
            sa_velocity_start = sim_data->sa_v_step_start.view(),
            sa_vert_velocity = sim_data->sa_v.view(),
            sa_x_start = sim_data->sa_x_step_start.view()
        ](const Float substep_dt, const Bool fix_scene, const Float damping)
        {
            const UInt vid = dispatch_id().x;

            Float3 x_k_init = sa_iter_start_position->read(vid);
            Float3 x_k = sa_iter_position->read(vid);

            Float3 dx = x_k - x_k_init;
            Float3 vel = dx / substep_dt;

            $if (fix_scene) 
            {
                dx = Zero3;
                vel = Zero3;
                sa_iter_position->write(vid, sa_x_start->read(vid));
                return;
            };

            vel *= exp(-damping * substep_dt);

            sa_vert_velocity->write(vid, vel);
            sa_velocity_start->write(vid, vel);
            sa_iter_start_position->write(vid, x_k);
        }
    );

    fn_evaluate_inertia = device.compile<1>(
        [
            sa_Hf = sim_data->sa_Hf.view(),
            sa_Hf1 = sim_data->sa_Hf1.view(),
            sa_x = sim_data->sa_x.view(),
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_x_start = sim_data->sa_x_step_start.view(),
            sa_v = sim_data->sa_v.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view()
        , writeHf, makeHf1](const Float substep_dt)
        {
            const UInt vid = dispatch_id().x;
            const Float3 gravity(0, -9.8f, 0);
            const Float h = substep_dt;
            const Float h_2_inv = 1.f / (h * h);

            Float3 x_k = sa_x->read(vid);
            Float3 x_tilde = sa_x_tilde->read(vid);

            auto is_fixed = sa_is_fixed->read(vid);
            Float mass = sa_vert_mass->read(vid);
            Float3x3 mat = make_float3x3(1.0f) * mass * h_2_inv;

            // Float3 outer_force = mass * gravity;

            $if (is_fixed != 0)
            {
                mat = make_float3x3(1.0f) * float(1E9);
            };
            Float3 gradient = -mass * h_2_inv * (x_k - x_tilde) ; // + outer_force;
            
            Float4x3 Hf = makeHf1(gradient, mat); sa_Hf1->write(vid, Hf);
            writeHf(gradient, mat, sa_Hf, vid);
        }
    );

    auto outer_product = [](const Float3& left, const Float3& right)
    {
        return make_float3x3(
            left * right[0],
            left * right[1],
            left * right[2]
        );
    };

    // for (uint vid = 0; vid < 10; vid++)
    // {
    //     const auto& vert_adj_edge = host_mesh_data->vert_adj_edges[vid];
    //     for (uint j = 0; j < vert_adj_edge.size(); j++)
    //     {
    //         const uint adj_eid = vert_adj_edge[j];
    //         auto edge = host_mesh_data->sa_edges[adj_eid];
    //         float3 vert_pos[2] = {
    //             host_mesh_data->sa_rest_x[edge[0]],
    //             host_mesh_data->sa_rest_x[edge[1]],
    //         };
    //         float3 diff = vert_pos[1] - vert_pos[0];
    //         float l = max(length(diff), Epsilon);
    //         luisa::log_info("vid {}'s {}th adj {}: rest length = {}", vid, j, adj_eid, l);
    //     }
    // }

    fn_evaluate_stretch_spring = device.compile<1>(
        [
            sa_Hf = sim_data->sa_Hf.view(),
            sa_Hf1 = sim_data->sa_Hf1.view(),
            sa_iter_position = sim_data->sa_x.view(),
            sa_start_position = sim_data->sa_x_step_start.view(),
            sa_vert_adj_edges_csr = mesh_data->sa_vert_adj_edges_csr.view(),
            sa_edges = mesh_data->sa_edges.view(),
            sa_rest_length = mesh_data->sa_edges_rest_state_length.view()
        , extractHf, writeHf, outer_product, makeHf1](const Float stiffness_stretch)
        {
            const Uint vid = dispatch_id().x;
            const Uint curr_prefix = sa_vert_adj_edges_csr->read(vid);
            const Uint next_prefix = sa_vert_adj_edges_csr->read(vid + 1);
            const Uint num_adj = next_prefix - curr_prefix;
            
            // Float4x3 hf = make_float4x3(make_float3(0.0f), make_float3(0.0f), make_float3(0.0f), make_float3(0.0f));
            Float3 force = make_float3(0.0f);
            Float3x3 hessian = make_float3x3(0.0f);
            $for (j, num_adj)
            {
                const Uint adj_eid = sa_vert_adj_edges_csr->read(curr_prefix + j);
                Uint2 edge = sa_edges->read(adj_eid);

                Float3 vert_pos[2] = {
                    sa_iter_position->read(edge[0]),
                    sa_iter_position->read(edge[1]),
                };
                Float3 force_0;
                Float3x3 He = make_float3x3(0.0f);
                
                Float3 diff = vert_pos[1] - vert_pos[0];
                Float l = max(length(diff), Epsilon);
                const Float L = sa_rest_length->read(adj_eid);
                const Float stiffness_stretch_spring = stiffness_stretch;

                Float l0 = L;
                Float C = l - l0;
                // if (C > Epsilon)
                
                Float3 dir = diff / l;
                force_0 = stiffness_stretch_spring * dir * C;

            
                // Float3x3 xxT = outer_product(diff, diff);
                Float3x3 ddT = outer_product(dir, dir);
                Float x_inv = 1.f / l;
                // Float x_squared_inv = x_inv * x_inv;
                // He = stiffness_stretch_spring * x_squared_inv * xxT + stiffness_stretch_spring * max(1.0f - L * x_inv, 0.0f) * (make_float3x3(1.0f) - x_squared_inv * xxT) ;
                He = stiffness_stretch_spring * ddT + stiffness_stretch_spring * max(1.0f - L * x_inv, 0.0f) * (make_float3x3(1.0f) - ddT) ;

                force += lcsv::select(vid == edge[0], force_0, -force_0);
                hessian += He;

            };

            Float4x3 hf = sa_Hf1->read(vid);
            hf.cols[0] += force;
            hf.cols[1] += hessian[0];
            hf.cols[2] += hessian[1];
            hf.cols[3] += hessian[2];
            sa_Hf1->write(vid, hf);

            Float3 orig_force; Float3x3 orig_hessian; extractHf(orig_force, orig_hessian, sa_Hf, vid);
            writeHf(orig_force + force, orig_hessian + hessian, sa_Hf, vid);
        }
    );
    
    /*
    fn_evaluate_bending = device.compile<1>(
        [
            sa_Hf = xpbd_data->sa_Hf.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_vert_adj_bending_edges = mesh_data->sa_vert_adj_bending_edges.view(),
            sa_bending_edges = mesh_data->sa_bending_edges.view(),
            sa_bending_edge_Q = mesh_data->sa_bending_edges_Q.view(),
            stiffness_bending = get_scene_params().get_stiffness_quadratic_bending()
        ]
        {
            const UInt vid = dispatch_id().x;
            const uint curr_prefix = sa_vert_adj_bending_edges->read(vid);
            const uint next_prefix = sa_vert_adj_bending_edges->read(vid + 1);
            const uint num_adj_edges = next_prefix - curr_prefix;

            Float4x3 hf = Zero4x3;
            for (uint j = 0; j < num_adj_edges; j++)
            {
                const uint adj_eid = sa_vert_adj_bending_edges->read(curr_prefix + j);
                Int4 edge = sa_bending_edges->read(adj_eid);

                Float3 vert_pos[4] = {
                    sa_iter_position->read(edge[0]),
                    sa_iter_position->read(edge[1]),
                    sa_iter_position->read(edge[2]),
                    sa_iter_position->read(edge[3])
                };

                Float4x4 Q = sa_bending_edge_Q->read(adj_eid);
                const uint offset = vid == edge[0] ? 0 : vid == edge[1] ? 1 : vid == edge[2] ? 2 : vid == edge[3] ? 3 : -1u;

                Float3 force = Zero3;
                Float3x3 hessian = Zero3x3;
                for (uint jj = 0; jj < 4; jj++) 
                {
                    force -= get(Q, offset, jj) * vert_pos[jj];
                }
                force *= stiffness_bending;
                hessian = stiffness_bending * get(Q, offset, offset) * Identity3x3;
                hf += makeHf(force, hessian);
            }
            sa_Hf->write(vid, sa_Hf->read(vid) + hf);
        }
    );
    */

    fn_step = device.compile<1>(
        [
            sa_Hf = sim_data->sa_Hf.view(),
            sa_Hf1 = sim_data->sa_Hf1.view(),
            sa_iter_position = sim_data->sa_x.view()
        , extractHf]
        {
            const UInt vid = dispatch_id().x;

            Float3 f;
            Float3x3 H;
            extractHf(f, H, sa_Hf, vid);

            // Float4x3 Hf = sa_Hf1->read(vid);
            // Float3 f = Hf.cols[0];
            // Float3x3 H = make_float3x3(
            //     Hf.cols[1],
            //     Hf.cols[2],
            //     Hf.cols[3]
            // );

            Float det = luisa::compute::determinant(H);
            $if (luisa::compute::abs(det) > Epsilon) 
            {
                // det
                Float3x3 H_inv = luisa::compute::inverse(H);
                Float3 dx = H_inv * f;
                // dx = 0.3f * dx;
                // dx *= 0.3f;
                sa_iter_position->write(vid, sa_iter_position->read(vid) + dx);
            };
        }
    );
}
void DescentSolver::collision_detection(luisa::compute::Stream& stream)
{
    // TODO
}
void DescentSolver::predict_position(luisa::compute::Stream& stream)
{

}
void DescentSolver::update_velocity(luisa::compute::Stream& stream)
{

}
// void CpuSolver::compute_energy(const Buffer<float3>& curr_position)
void compute_energy()
{
    if (!lcsv::get_scene_params().print_system_energy) return;
    // // luisa::log_info("buffer size = {}", curr_position.size());

    // double energy = 0.0;
    // double energy_inertia = 0.f, energy_stretch = 0.f, energy_bending = 0.f;

    // // Inertia
    // {
    //     energy_inertia = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
    //     {
    //         return Constrains::Energy::compute_energy_inertia(vid, 
    //             curr_position.data(), 
    //             &lcsv::get_scene_params(), 
    //             mesh_data->sa_is_fixed.data(), 
    //             mesh_data->sa_vert_mass.data(), 
    //             xpbd_data->sa_x_start.data(), 
    //             xpbd_data->sa_v_start.data());
    //     });
    // }
    
    // // Stretch 
    // {
    //     const float stiffness = lcsv::get_scene_params().stiffness_stretch_spring;
    //     energy_stretch = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_edges, [&](const uint eid)
    //     {
    //         return Constrains::Energy::compute_energy_stretch_mass_spring(
    //             eid, curr_position.data(), 
    //             xpbd_data->sa_merged_edges.data(), 
    //             xpbd_data->sa_merged_edges_rest_length.data(), 
    //             stiffness);
    //     });
    // }

    // // Bending
    // if (lcsv::get_scene_params().use_bending)
    // {   
    //     const auto bending_type = 
    //         (lcsv::get_scene_params().use_vbd_solver // Our VBD solver only add quadratic bending implementation
    //         || lcsv::get_scene_params().use_quadratic_bending_model) ?  
    //         Constrains::BendingTypeQuadratic : Constrains::BendingTypeDAB;
    //     const bool use_xpbd_solver = lcsv::get_scene_params().use_xpbd_solver;

    //     const float stiffness_bending_quadratic = lcsv::get_scene_params().get_stiffness_quadratic_bending();
    //     const float stiffness_bending_DAB = lcsv::get_scene_params().get_stiffness_DAB_bending();

    //     energy_bending = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_bending_edges, [&](const uint eid)
    //     {
    //         float energy = 0.f;
    //         Constrains::Energy::compute_energy_bending(bending_type, eid, curr_position.data(), 
    //             xpbd_data->sa_merged_bending_edges.data(), 
    //             nullptr,
    //             nullptr, 
    //             xpbd_data->sa_merged_bending_edges_Q.data(),
    //             xpbd_data->sa_merged_bending_edges_angle.data(), 
    //             stiffness_bending_DAB, 
    //             stiffness_bending_quadratic, 
    //             use_xpbd_solver
    //         );
    //         return energy;
    //     });
    // }
    
    // // Obstacle Collisoin
    // float energy_obs_collision = 0.0f;

    // // Self Collision
    // float energy_self_collision = 0.0f;

    // double total_energy = energy_inertia + energy_stretch + energy_bending + energy_obs_collision + energy_self_collision;

    // mesh_data->sa_system_energy[energy_idx++] = total_energy;
}

void DescentSolver::physics_step_xpbd(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcsv::SolverInterface::physics_step_prev_operation();
}
void DescentSolver::physics_step_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcsv::SolverInterface::physics_step_prev_operation(); 
    // Get frame start position and velocity
    CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
    {
        host_sim_data->sa_x[vid] = host_mesh_data->sa_x_frame_outer[vid];
        host_sim_data->sa_v[vid] = host_mesh_data->sa_v_frame_outer[vid];
    });
    
    // Upload to GPU
    stream << sim_data->sa_x.copy_from(host_sim_data->sa_x.data())
           << sim_data->sa_v.copy_from(host_sim_data->sa_v.data())
           << sim_data->sa_x_step_start.copy_from(host_sim_data->sa_x.data())
           << sim_data->sa_v_step_start.copy_from(host_sim_data->sa_v.data())
           << mp_buffer_filler->fill(device, mesh_data->sa_system_energy, 0.0f)
           << luisa::compute::synchronize();
    
    // const uint num_substep = lcsv::get_scene_params().print_xpbd_convergence ? 1 : lcsv::get_scene_params().num_substep;
    const uint num_substep = lcsv::get_scene_params().num_substep;
    const uint nonlinear_iter_count = lcsv::get_scene_params().nonlinear_iter_count;
    const float substep_dt = lcsv::get_scene_params().get_substep_dt();

    // energy_idx = 0;
    for (uint substep = 0; substep < num_substep; substep++)
    {
        stream << fn_predict_position(substep_dt).dispatch(mesh_data->num_verts) << luisa::compute::synchronize();

        for (uint iter = 0; iter < nonlinear_iter_count; iter++)
        {
            stream 
                << fn_evaluate_inertia(substep_dt).dispatch(mesh_data->num_verts)
                << fn_evaluate_stretch_spring(1e4).dispatch(mesh_data->num_verts)
                << fn_step().dispatch(mesh_data->num_verts) << luisa::compute::synchronize();
        }
        // const Float substep_dt, const Bool fix_scene, const Float damping
        stream << fn_update_velocity(lcsv::get_scene_params().get_substep_dt(), false, lcsv::get_scene_params().damping_cloth).dispatch(mesh_data->num_verts) << luisa::compute::synchronize();
    }
    
    stream << luisa::compute::synchronize();
    
    // Copy to host (if use GPU)
    {
        stream  << sim_data->sa_x.copy_to(host_sim_data->sa_x.data())
                << sim_data->sa_v.copy_to(host_sim_data->sa_v.data())
                << luisa::compute::synchronize();
    }
    
    // Return frame end position and velocity
    CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
    {
        host_mesh_data->sa_x_frame_outer[vid] = host_sim_data->sa_x[vid];
        host_mesh_data->sa_v_frame_outer[vid] = host_sim_data->sa_v[vid];
    });
    lcsv::SolverInterface::physics_step_post_operation(); 
}
void DescentSolver::physics_step_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    lcsv::SolverInterface::physics_step_prev_operation(); 
    // Get frame start position and velocity
    CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
    {
        // TODO: Move copy into SolverInterface
        host_sim_data->sa_x[vid] = host_mesh_data->sa_x_frame_outer[vid];
        host_sim_data->sa_v[vid] = host_mesh_data->sa_v_frame_outer[vid];
        host_sim_data->sa_x_step_start[vid] = host_mesh_data->sa_x_frame_outer[vid];
        host_sim_data->sa_v_step_start[vid] = host_mesh_data->sa_v_frame_outer[vid];
    });
    std::fill(host_mesh_data->sa_system_energy.begin(), host_mesh_data->sa_system_energy.end(), 0.0f);
    
    const uint num_substep = lcsv::get_scene_params().num_substep;
    const uint nonlinear_iter_count = lcsv::get_scene_params().nonlinear_iter_count;
    const float substep_dt = lcsv::get_scene_params().get_substep_dt();
    const bool print_energy = lcsv::get_scene_params().print_system_energy;

    auto predict_position = [&](const float substep_dt)
    {
        auto* sa_x = host_sim_data->sa_x.data();
        auto* sa_x_tilde = host_sim_data->sa_x_tilde.data();
        auto* sa_v = host_sim_data->sa_v.data();
        auto* sa_x_start = host_sim_data->sa_x_step_start.data();
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();

        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            const float3 gravity(0, -9.8f, 0);
            float3 x_0 = sa_x_start[vid];
            float3 v_0 = sa_v[vid];
            float3 outer_acceleration = gravity;
            float3 v_pred = v_0 + substep_dt * outer_acceleration;
            if (sa_is_fixed[vid] != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
            const float3 x_pred = x_0 + substep_dt * v_pred;
            sa_x_tilde[vid] = x_pred;
            
            // sa_x[vid] = x_pred;
            sa_x[vid] = x_0;
            {
            	// Adaptive Init
            	// Float3 a_t = (v_0 - v_prev) / h;
            	// float len_outer_accelaration = length_vec(outer_acceleration);

            	// float a_t_ext = dot_vec(a_t, outer_acceleration / len_outer_accelaration);
            	// float a_hat = a_t_ext > len_outer_accelaration ? 1.f 
            	// 			: a_t_ext < 0 ?                      0.f
            	// 			: a_t_ext / len_outer_accelaration;
            	// x_k += a_hat * h * h * outer_acceleration;
            }

            
        });
    };
    auto update_velocity = [&](const float substep_dt, const bool fix_scene, const float damping)
    {
        auto* sa_iter_position = host_sim_data->sa_x.data();
        auto* sa_iter_start_position = host_sim_data->sa_x_step_start.data();
        auto* sa_vert_velocity = host_sim_data->sa_v.data();
        auto* sa_velocity_start = host_sim_data->sa_v_step_start.data();

        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            float3 x_k_init = sa_iter_start_position[vid];
            float3 x_k = sa_iter_position[vid];

            float3 dx = x_k - x_k_init;
            float3 vel = dx / substep_dt;

            if (fix_scene) 
            {
                dx = Zero3;
                vel = Zero3;
                sa_iter_position[vid] = sa_iter_start_position[vid];
                return;
            };

            vel *= exp(-damping * substep_dt);

            sa_vert_velocity[vid] = vel;
            sa_velocity_start[vid] = vel;
            sa_iter_start_position[vid] = x_k;
        });
    };
    auto evaluate_inertia = [&](const float substep_dt)
    {
        auto* sa_x = host_sim_data->sa_x.data();
        auto* sa_v = host_sim_data->sa_v.data();
        auto* sa_x_start = host_sim_data->sa_x_step_start.data();
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();
        auto* sa_vert_mass = host_mesh_data->sa_vert_mass.data();
        auto* sa_Hf1 = host_sim_data->sa_Hf1.data();

        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            const float3 gravity(0, -9.8f, 0);
            const float h = substep_dt;
            const float h_2_inv = 1.f / (h * h);

            float3 x_k = sa_x[vid];
            float3 x_0 = sa_x_start[vid];
            float3 v_0 = sa_v[vid];

            auto is_fixed = sa_is_fixed[vid];
            float mass = sa_vert_mass[vid];
            float3x3 mat = luisa::make_float3x3(1.0f) * mass * h_2_inv;
            float3 outer_force = mass * gravity;

            if (is_fixed != 0)
            {
                mat = mat + luisa::make_float3x3(1.0f) * float(1E9);
            };
            float3 gradient = -mass * h_2_inv * (x_k - x_0 - v_0 * h) + outer_force;
            float4x3 Hf = float4x3{gradient, mat[0], mat[1], mat[2]};
            sa_Hf1[vid] = Hf;
        });
    };
    auto evaluate_spring = [&](const float stiffness_stretch)
    {
        auto* sa_iter_position = host_sim_data->sa_x.data();
        auto* sa_vert_adj_edges_csr = host_mesh_data->sa_vert_adj_edges_csr.data();
        auto* sa_edges = host_mesh_data->sa_edges.data();
        auto* sa_rest_length = host_mesh_data->sa_edges_rest_state_length.data();
        auto* sa_Hf1 = host_sim_data->sa_Hf1.data();

        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {   
            const uint curr_prefix = sa_vert_adj_edges_csr[vid];
            const uint next_prefix = sa_vert_adj_edges_csr[vid + 1];
            const uint num_adj = next_prefix - curr_prefix;
            
            float4x3 hf{Zero3, Zero3, Zero3, Zero3};
            for (uint j = 0; j < num_adj; j++)
            {
                const uint adj_eid = sa_vert_adj_edges_csr[curr_prefix + j];
                auto edge = sa_edges[adj_eid];

                float3 vert_pos[2] = {
                    sa_iter_position[edge[0]],
                    sa_iter_position[edge[1]],
                };
                float3 force[2] = {Zero3, Zero3};
                float3x3 He = luisa::make_float3x3(0.0f);
        
                
                const float L = sa_rest_length[adj_eid];
                const float stiffness_stretch_spring = stiffness_stretch;

                float3 diff = vert_pos[1] - vert_pos[0];
                float l = max_scalar(length_vec(diff), Epsilon);
                float l0 = L;
                float C = l - l0;
                // if (C > Epsilon)
                
                float3 dir = diff / l;
                force[0] = stiffness_stretch_spring * dir * C;
                force[1] = -force[0];

                
                float3x3 xxT = outer_product(diff, diff);
                float x_inv = 1.f / l;
                float x_squared_inv = x_inv * x_inv;
                He = stiffness_stretch_spring * x_squared_inv * xxT + stiffness_stretch_spring * max_scalar(1 - L * x_inv, 0.0f) * (luisa::make_float3x3(1.0f) - x_squared_inv * xxT) ;

                const uint offset = vid == edge[0] ? 0 : vid == edge[1] ? 1 : -1u;
                hf.cols[0] += force[offset];
                hf.cols[1] += He.cols[0];
                hf.cols[2] += He.cols[1];
                hf.cols[3] += He.cols[2];
            }

            
            sa_Hf1[vid] += hf;
        });
    };
    auto step = [&]()
    {
        CpuParallel::parallel_for(0, mesh_data->num_verts, [&](const uint vid)
        {   
            float4x3 Hf = host_sim_data->sa_Hf1[vid];
            float3 f = Hf.cols[0];
            float3x3 H = make_float3x3(
                Hf.cols[1],
                Hf.cols[2],
                Hf.cols[3]
            );

            float det = luisa::determinant(H);
            if (luisa::abs(det) > Epsilon) 
            {
                // det
                float3x3 H_inv = luisa::inverse(H);
                float3 dx = H_inv * f;
                dx *= 0.3f;
                host_sim_data->sa_x[vid] += dx;
            };
        });
    };
    
    for (uint substep = 0; substep < num_substep; substep++)
    {
        predict_position(substep_dt);

        if (print_energy) luisa::log_info("Frame {} start   position energy = {}", lcsv::get_scene_params().current_frame, host_compute_energy(host_sim_data->sa_x_step_start, host_sim_data->sa_x_tilde));
        if (print_energy) luisa::log_info("Frame {} predict position energy = {}", lcsv::get_scene_params().current_frame ,host_compute_energy(host_sim_data->sa_x, host_sim_data->sa_x_tilde));

        for (uint iter = 0; iter < nonlinear_iter_count; iter++)
        {
            evaluate_inertia(substep_dt);
            
            evaluate_spring(1e4);

            step();

            if (print_energy) luisa::log_info("    Non-linear iter {:2} energy = {}", iter, host_compute_energy(host_sim_data->sa_x, host_sim_data->sa_x_tilde));
        }
        update_velocity(substep_dt, false, lcsv::get_scene_params().damping_cloth);
    }

    // Return frame end position and velocity
    CpuParallel::parallel_for(0, host_sim_data->sa_x.size(), [&](const uint vid)
    {
        host_mesh_data->sa_x_frame_outer[vid] = host_sim_data->sa_x[vid];
        host_mesh_data->sa_v_frame_outer[vid] = host_sim_data->sa_v[vid];
    });
    lcsv::SolverInterface::physics_step_post_operation(); 
}
void DescentSolver::solve_constraints_VBD(luisa::compute::Stream& stream)
{
    // auto& iter_position = xpbd_data->sa_x;

    // if (lcsv::get_scene_params().print_xpbd_convergence && lcsv::get_scene_params().current_it == 0) 
    // { 
    //     compute_energy(iter_position); 
    // }

    // for (uint cluster = 0; cluster < xpbd_data->num_clusters_per_vertex_bending; cluster++)
    // {
    //     const uint next_prefix = xpbd_data->clusterd_per_vertex_bending[cluster + 1];
    //     const uint curr_prefix = xpbd_data->clusterd_per_vertex_bending[cluster];
    //     const uint num_verts_cluster = next_prefix - curr_prefix;

    //     vbd_evaluate_inertia(iter_position, cluster);

    //     vbd_evaluate_stretch_spring(iter_position, cluster);
        
    //     vbd_evaluate_bending(iter_position, cluster);
        
    //     vbd_step(iter_position, cluster);
    // }

    // if (lcsv::get_scene_params().print_xpbd_convergence) 
    // { 
    //     compute_energy(iter_position); 
    // }
}

/*
auto shader_predict_position = device.compile<1>([&]
{
    const UInt vid = dispatch_id().x;
    const Float3 gravity(0, -9.8f, 0);
    Float3 x_prev = xpbd_data->sa_x_start->read(vid);
    Float3 v_prev = xpbd_data->sa_v->read(vid);
    Float3 outer_acceleration = gravity; 
    const float substep_dt = lcsv::get_scene_params().get_substep_dt();
    Float3 v_pred = v_prev + substep_dt * outer_acceleration; 
    $if(mesh_data->sa_is_fixed->read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
    const Float3 x_pred = x_prev + substep_dt * v_pred;
    xpbd_data->sa_x->write(vid, x_pred);
}, {
    .enable_debug_info = use_debug_info
});

luisa::compute::Shader<1, luisa::compute::Buffer<luisa::Vector<float, 3>>,
                    luisa::compute::Buffer<luisa::Vector<float, 3>>,
                    luisa::compute::Buffer<luisa::Vector<float, 3>>, bool,
                    luisa::compute::Buffer<luisa::Vector<float, 3>>,
                    luisa::compute::Buffer<float>,
                    luisa::compute::Buffer<unsigned int>, float, bool> fn_predict_position;

fn_predict_position = device.compile<1>([&](
    PTR(float3) sa_iter_position, PTR(float3) sa_vert_velocity, PTR(float3) sa_iter_start_position, 
    const Bool predict_for_collision, PTR(float3) sa_next_position,
    PTR(float) sa_vert_mass, 
    PTR(uint) sa_is_fixed,
    const Float substep_dt,
    const Bool fix_scene)
{
    const Uint vid = dispatch_id().x;
    const Float3 gravity = make_float3(0, -9.8f, 0);
    const Float3 x_prev = sa_iter_start_position.read(vid);
    const Float3 v_prev = sa_vert_velocity.read(vid);
    Float3 outer_acceleration = gravity; 
    Float3 v_pred = v_prev + substep_dt * outer_acceleration; 
    $if (sa_is_fixed.read(vid) != 0) { outer_acceleration = Zero3; v_pred = Zero3; };
    const Float3 x_pred = x_prev + substep_dt * v_pred;
    sa_iter_position.write(vid, x_pred);
});
*/

} // namespace lcsv
