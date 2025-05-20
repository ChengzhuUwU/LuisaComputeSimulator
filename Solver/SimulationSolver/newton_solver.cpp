#include "SimulationSolver/descent_solver.h"
#include "SimulationSolver/newton_solver.h"
#include "Core/float_nxn.h"
#include "Core/scalar.h"
#include "Core/xbasic_types.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include <luisa/dsl/sugar.h>
#include <Eigen/Sparse>

namespace lcsv 
{

void NewtonSolver::compile(luisa::compute::Device& device)
{
    const bool use_debug_info = false;
    using namespace luisa::compute;

    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    auto fn_init_force = device.compile<1>(
        [
            sa_x = xpbd_data->sa_x.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_x_start = xpbd_data->sa_x_start.view(),
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
    }, default_option);

    auto fn_init_hessian = device.compile<1>(
        [
            sa_x = xpbd_data->sa_x.view(),
            sa_v = xpbd_data->sa_v.view(),
            sa_iter_start_position = xpbd_data->sa_x_start.view(),
            sa_iter_position = xpbd_data->sa_x.view(),
            sa_velocity_start = xpbd_data->sa_v_start.view(),
            sa_vert_velocity = xpbd_data->sa_v.view(),
            sa_x_start = xpbd_data->sa_x_start.view()
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
}
void NewtonSolver::physics_step_newton_CPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    // Input
    {
        lcsv::SolverInterface::physics_step_prev_operation(); 
        CpuParallel::parallel_for(0, host_xpbd_data->sa_x.size(), [&](const uint vid)
        {
            host_xpbd_data->sa_x[vid] = host_mesh_data->sa_x_frame_start[vid];
            host_xpbd_data->sa_v[vid] = host_mesh_data->sa_v_frame_start[vid];
            host_xpbd_data->sa_x_start[vid] = host_mesh_data->sa_x_frame_start[vid];
            host_xpbd_data->sa_v_start[vid] = host_mesh_data->sa_v_frame_start[vid];
        });
        std::fill(host_mesh_data->sa_system_energy.begin(), host_mesh_data->sa_system_energy.end(), 0.0f);
    }


    std::vector<float3> sa_x_tilde;
    std::vector<float3> sa_x;
    std::vector<float3> sa_v;
    std::vector<float3> sa_v_start;
    std::vector<float3> sa_x_start;

    

    std::vector<float3> sa_b;
    std::vector<float3x3> sa_diagA;
    std::vector<float3x3> sa_offdiagA; // Row-major for simplier SpMV

    constexpr bool use_eigen = false;
    Eigen::SparseMatrix<float> cgB;
    Eigen::SparseMatrix<float> cgA;
    cgB.resize(1, mesh_data->num_verts * 3);
    cgA.resize(mesh_data->num_verts * 3, mesh_data->num_verts * 3);
    cgB.resizeNonZeros(mesh_data->num_verts * 3);
    cgA.resizeNonZeros(mesh_data->num_verts * 9 + mesh_data->num_edges * 9 * 2);
    
    const uint curr_frame = get_scene_params().current_frame;
    if (curr_frame == 0)
    {
        const uint num_verts = host_mesh_data->num_verts;
        const uint num_edges = host_mesh_data->num_edges;
        const uint num_faces = host_mesh_data->num_faces;
        sa_b.resize(num_verts);
        sa_diagA.resize(num_verts);
        sa_offdiagA.resize(num_edges * 2);
    }

    // Init vert
    {
        // auto* sa_x = host_xpbd_data->sa_x.data();
        // auto* sa_v = host_xpbd_data->sa_v.data();
        // auto* sa_x_start = host_xpbd_data->sa_x_start.data();
        auto* sa_is_fixed = host_mesh_data->sa_is_fixed.data();
        auto* sa_vert_mass = host_mesh_data->sa_vert_mass.data();
        CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](const uint vid)
        {
            const float3 gravity(0, -9.8f, 0);
            const float h = lcsv::get_scene_params().implicit_dt;
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
            float3 gradient = -mass * h_2_inv * (x_k - x_0 - v_0 * h) + outer_force;;
            if constexpr (use_eigen)
            {
                cgB.block<3, 3>(      0, 3 * vid) = gradient;
                cgA.block<3, 3>(3 * vid, 3 * vid) = mat;
            }
            else 
            {
                sa_b[vid] = gradient;
                sa_diagA[vid] = mat;
            }
            
        }); 
    }

    // Evaluate Spring
    {
        auto* sa_iter_position = host_xpbd_data->sa_x.data();
        auto* sa_edges = host_mesh_data->sa_edges.data();
        auto* sa_rest_length = host_mesh_data->sa_edges_rest_state_length.data();
        const float stiffness_stretch = 1e4;
        
        for (uint cluster_idx = 0; cluster_idx < host_xpbd_data->num_clusters_stretch_mass_spring; cluster_idx++) 
        {
            const uint curr_prefix = host_xpbd_data->prefix_stretch_mass_spring[cluster_idx];
            const uint next_prefix = host_xpbd_data->prefix_stretch_mass_spring[cluster_idx + 1];
            const uint num_elements_clustered = next_prefix - curr_prefix;

            CpuParallel::parallel_for(0, num_elements_clustered, [&](const uint i)
            {
                const uint eid = curr_prefix + i;
                uint2 edge = sa_edges[eid];

                float3 vert_pos[2] = {
                    sa_iter_position[edge[0]],
                    sa_iter_position[edge[1]],
                };
                float3 force[2] = {Zero3, Zero3};
                float3x3 He = luisa::make_float3x3(0.0f);
        
                
                const float L = sa_rest_length[eid];
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
                
                sa_b[edge[0]] = sa_b[edge[0]] + force[0];
                sa_b[edge[1]] = sa_b[edge[1]] + force[1];
                sa_diagA[edge[1]] = sa_diagA[edge[1]] + He;
                sa_diagA[edge[1]] = sa_diagA[edge[1]] + He;
                sa_offdiagA[eid * 2 + 0] = -1.0f * He;
                sa_offdiagA[eid * 2 + 1] = -1.0f * He;
            }, 32);
        }
    }

    // Solve
    {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<float>> solver(cgA);

    }
    

    // Output
    {
        CpuParallel::parallel_for(0, host_xpbd_data->sa_x.size(), [&](const uint vid)
        {
            host_mesh_data->sa_x_frame_end[vid] = host_xpbd_data->sa_x[vid];
            host_mesh_data->sa_v_frame_end[vid] = host_xpbd_data->sa_v[vid];
        });
        lcsv::SolverInterface::physics_step_post_operation(); 
    }
}
void NewtonSolver::physics_step_newton_GPU(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    
}



} // namespace lcsv 