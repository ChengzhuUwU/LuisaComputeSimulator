#include "SimulationSolver/solver_interface.h"
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/friction_kernel.hpp"
#include "Core/affine_position.h"
#include "Core/scalar.h"
#include "Energy/bending_energy.h"
#include "Energy/stretch_energy.h"
#include "Initializer/init_collision_data.h"
#include "Initializer/init_sim_data.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "SimulationCore/scene_params.h"
#include "MeshOperation/mesh_reader.h"
#include "luisa/dsl/builtin.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/stream.h"
#include <numeric>

namespace lcs
{

void SolverInterface::set_data_pointer(SolverData& solver_data, SolverHelper& solver_helper)
{
    // Data pointer
    this->host_mesh_data = &solver_data.host_mesh_data;
    this->host_sim_data  = &solver_data.host_sim_data;

    this->mesh_data = &solver_data.mesh_data;
    this->sim_data  = &solver_data.sim_data;

    this->lbvh_data_face = &solver_data.lbvh_data_face;
    this->lbvh_data_edge = &solver_data.lbvh_data_edge;

    this->host_collision_data = &solver_data.host_collision_data;
    this->collision_data      = &solver_data.collision_data;

    // Tool class pointer
    this->lbvh_face             = &solver_helper.lbvh_face;
    this->lbvh_edge             = &solver_helper.lbvh_edge;
    this->device_parallel       = &solver_helper.device_parallel;
    this->buffer_filler         = &solver_helper.buffer_filler;
    this->narrow_phase_detector = &solver_helper.narrow_phase_detector;
    pcg_solver                  = &solver_helper.pcg_solver;
}
void SolverInterface::init_data(luisa::compute::Device&                   device,
                                luisa::compute::Stream&                   stream,
                                std::vector<lcs::Initializer::WorldData>& world_data)
{
    set_data_pointer(solver_data, solver_helper);

    // Init data
    {
        lcs::Initializer::init_mesh_data(world_data, host_mesh_data);
        lcs::Initializer::upload_mesh_buffers(device, stream, host_mesh_data, mesh_data);
    }

    {
        lcs::Initializer::init_sim_data(world_data, host_mesh_data, host_sim_data);
        lcs::Initializer::upload_sim_buffers(device, stream, host_sim_data, sim_data);
        lcs::Initializer::resize_pcg_data(device, stream, host_mesh_data, host_sim_data, sim_data);
    }

    {
        lcs::Initializer::init_collision_data(world_data, host_mesh_data, host_sim_data, host_collision_data);
        lcs::Initializer::upload_collision_buffers(device, stream, host_sim_data, sim_data, host_collision_data, collision_data);
    }

    {
        lbvh_data_face->allocate(device, host_mesh_data->num_faces, lcs::LBVHTreeTypeFace);
        lbvh_data_edge->allocate(device, host_mesh_data->num_edges, lcs::LBVHTreeTypeEdge);
        lcs::Initializer::init_lbvh_data(device, stream, lbvh_data_face);
        lcs::Initializer::init_lbvh_data(device, stream, lbvh_data_edge);
        // lbvh_cloth_vert.unit_test(device, stream);
    }

    {
        host_collision_data->allocate_basic_buffers(device,
                                                    host_mesh_data->num_verts,
                                                    host_mesh_data->num_faces,
                                                    host_mesh_data->num_edges,
                                                    host_sim_data->num_dof);

        collision_data->allocate_basic_buffers(device,
                                               host_mesh_data->num_verts,
                                               host_mesh_data->num_faces,
                                               host_mesh_data->num_edges,
                                               host_sim_data->num_dof);

        host_collision_data->resize_collision_data_list(device,
                                                        host_mesh_data->num_verts,
                                                        host_mesh_data->num_faces,
                                                        host_mesh_data->num_edges,
                                                        host_sim_data->num_dof,
                                                        false,
                                                        true);

        collision_data->resize_collision_data_list(device,
                                                   host_mesh_data->num_verts,
                                                   host_mesh_data->num_faces,
                                                   host_mesh_data->num_edges,
                                                   host_sim_data->num_dof,
                                                   true,
                                                   true);
    }
}
void SolverInterface::compile(AsyncCompiler& compiler)
{
    compile_compute_energy(compiler);

    {
        lbvh_face->set_lbvh_data(lbvh_data_face);
        lbvh_edge->set_lbvh_data(lbvh_data_edge);
        lbvh_face->compile(compiler);
        lbvh_edge->compile(compiler);
    }

    LUISA_INFO("JIT Compiling Narrow Phase Detector...");
    {
        narrow_phase_detector->set_collision_data(host_collision_data, collision_data);
        narrow_phase_detector->compile(compiler);
        auto tmp_stream = compiler.device().create_stream();
        narrow_phase_detector->unit_test(compiler.device(), tmp_stream);
    }

    LUISA_INFO("JIT Compiling Solver...");
    {
        pcg_solver->set_data(host_mesh_data, mesh_data, host_sim_data, sim_data);
        pcg_solver->compile(compiler);
    }
}


template <typename T>
static void buffer_copy(const std::vector<T>& src, std::vector<T>& dst)
{
    if (src.size() != dst.size())
    {
        LUISA_ERROR("Buffer size mismatch {} != {}", src.size(), dst.size());
    }
    std::memcpy(dst.data(), src.data(), sizeof(T) * src.size());
}
void SolverInterface::physics_step_prev_operation()
{
    // Set target position velocity for fixed verts
    for (uint index = 0; index < per_vertex_animations.size(); index++)
    {
        const auto&  animate  = per_vertex_animations[index];
        const uint   vid      = animate.vertex_id;
        const float3 curr_pos = host_sim_data->sa_q_outer[vid];
        const float3 target_pos =
            luisa::make_float3(animate.translation[0], animate.translation[1], animate.translation[2]);
        const float3 desire_vel          = (target_pos - curr_pos) / lcs::get_scene_params().implicit_dt;
        host_sim_data->sa_q_v_outer[vid] = desire_vel;
    }

    // Now we only have fixed animation
    const uint prefix_dof_abd = host_sim_data->num_verts_soft;
    for (uint body_idx = 0; body_idx < host_sim_data->num_affine_bodies; body_idx++)
    {
        const bool is_fixed = host_sim_data->sa_q_is_fixed[prefix_dof_abd + 4 * body_idx] != 0;
        if (is_fixed)
        {
            const uint dof_idx = prefix_dof_abd + body_idx * 4;

            float4x3 rest_q;
            rest_q[0] = host_sim_data->sa_rest_q[dof_idx + 0];
            rest_q[1] = host_sim_data->sa_rest_q[dof_idx + 1];
            rest_q[2] = host_sim_data->sa_rest_q[dof_idx + 2];
            rest_q[3] = host_sim_data->sa_rest_q[dof_idx + 3];

            float4x3 curr_q;
            curr_q[0] = host_sim_data->sa_q_outer[dof_idx + 0];
            curr_q[1] = host_sim_data->sa_q_outer[dof_idx + 1];
            curr_q[2] = host_sim_data->sa_q_outer[dof_idx + 2];
            curr_q[3] = host_sim_data->sa_q_outer[dof_idx + 3];

            auto desire_vel = (rest_q - curr_q) / lcs::get_scene_params().implicit_dt;
            host_sim_data->sa_q_v_outer[dof_idx + 0] = desire_vel[0];
            host_sim_data->sa_q_v_outer[dof_idx + 1] = desire_vel[1];
            host_sim_data->sa_q_v_outer[dof_idx + 2] = desire_vel[2];
            host_sim_data->sa_q_v_outer[dof_idx + 3] = desire_vel[3];
        }
    }

    buffer_copy(host_sim_data->sa_q_outer, host_sim_data->sa_q_step_start);
    buffer_copy(host_sim_data->sa_q_v_outer, host_sim_data->sa_q_v);
}
void SolverInterface::physics_step_post_operation()
{
    buffer_copy(host_sim_data->sa_x, host_sim_data->sa_x_outer);
    buffer_copy(host_sim_data->sa_v, host_sim_data->sa_v_outer);
    buffer_copy(host_sim_data->sa_q, host_sim_data->sa_q_outer);
    buffer_copy(host_sim_data->sa_q_v, host_sim_data->sa_q_v_outer);
    per_vertex_animations.clear();
    per_body_animations.clear();
}

void SolverInterface::restart_system()
{
    buffer_copy(host_sim_data->sa_rest_x, host_sim_data->sa_x_outer);
    buffer_copy(host_sim_data->sa_rest_v, host_sim_data->sa_v_outer);
    buffer_copy(host_sim_data->sa_rest_q, host_sim_data->sa_q_outer);
    buffer_copy(host_sim_data->sa_rest_q_v, host_sim_data->sa_q_v_outer);
}

void SolverInterface::get_simulation_results_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions)
{
    const auto& sim_result_positions = host_sim_data->sa_x_outer;
    for (uint meshIdx = 0; meshIdx < host_mesh_data->num_meshes; meshIdx++)
    {
        CpuParallel::parallel_for(
            0,
            host_mesh_data->prefix_num_verts[meshIdx + 1] - host_mesh_data->prefix_num_verts[meshIdx],
            [&](const uint vid)
            {
                auto pos = sim_result_positions[vid + host_mesh_data->prefix_num_verts[meshIdx]];
                output_positions[meshIdx][vid] = {pos.x, pos.y, pos.z};
            },
            32);
    }
}
void SolverInterface::update_pinned_verts_position(const uint meshIdx,
                                                   const uint local_vid,
                                                   const std::array<float, 3>& pinned_verts_target_position)
{
    const uint prefix = host_mesh_data->prefix_num_verts[meshIdx];
    const uint vid    = prefix + local_vid;
    per_vertex_animations.push_back(
        {vid, {pinned_verts_target_position[0], pinned_verts_target_position[1], pinned_verts_target_position[2]}});
}
void SolverInterface::update_pinned_body_state(const uint                  body_id,
                                               const std::array<float, 3>& translation,
                                               const std::array<float, 4>& rotation)
{
    Animation::PerBodyAnimation tmp;
    tmp.set_translation(translation[0], translation[1], translation[2]);
    tmp.set_rotation(rotation[0], rotation[1], rotation[2], rotation[3]);
    per_body_animations.push_back(tmp);
}


void SolverInterface::save_current_frame_state_to_host(const uint frame, const std::string& addition_str)
{
    // save_current_frame_state();
    std::vector<float3> sa_q_frame_saved(host_sim_data->sa_q_outer);
    std::vector<float3> sa_qv_frame_saved(host_sim_data->sa_q_v_outer);

    const auto filename = luisa::format("frame_{}{}.state", frame, addition_str);

    std::string full_directory = std::string(LCSV_RESOURCE_PATH) + "/SimulationState/";

    {
        std::filesystem::path dir_path(full_directory);
        if (!std::filesystem::exists(dir_path))
        {
            try
            {
                std::filesystem::create_directories(dir_path);
                LUISA_INFO("Created directory: {}", dir_path.string());
            }
            catch (const std::filesystem::filesystem_error& e)
            {
                LUISA_ERROR("Error creating directory: {}", e.what());
                return;
            }
        }
    }

    std::string full_path = full_directory;
    full_path += std::string_view{filename};
    std::ofstream file(full_path, std::ios::out);


    if (file.is_open())
    {
        file << "o State" << std::endl;
        for (uint vid = 0; vid < host_sim_data->num_dof; vid++)
        {
            const auto vertex = sa_q_frame_saved[vid];
            file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
        }
        file << "o Velocity" << std::endl;
        for (uint vid = 0; vid < host_sim_data->num_dof; vid++)
        {
            const auto vel = sa_qv_frame_saved[vid];
            file << "v " << vel.x << " " << vel.y << " " << vel.z << std::endl;
        }

        file.close();
        LUISA_INFO("State file saved: {}", full_path);
    }
    else
    {
        LUISA_ERROR("Unable to open file: {}", full_path);
    }
}
static float3 fn_apply_template(const std::vector<uint>&   sa_x_to_dof_map,
                                const std::vector<float3>& sa_scaled_model_x,
                                const std::vector<float3>& input_q,
                                const uint                 vid)
{

    float3     new_dx;
    const uint map_info = sa_x_to_dof_map[vid];
    const uint dof_idx  = map_info & (~Attributions::RIGID_BODY_FLAG);
    if ((map_info & Attributions::RIGID_BODY_FLAG) == 0)  // Soft body
    {
        new_dx = input_q[dof_idx];
    }
    else  // Rigid body
    {
        const float3 rest_x = sa_scaled_model_x[vid];
        float3       p;
        float3x3     A;
        p      = input_q[dof_idx + 0];
        A[0]   = input_q[dof_idx + 1];
        A[1]   = input_q[dof_idx + 2];
        A[2]   = input_q[dof_idx + 3];
        new_dx = A * rest_x + p;  // Affine position
    };
    return new_dx;
};
void SolverInterface::load_saved_state_from_host(const uint frame, const std::string& addition_str)
{
    const auto filename = luisa::format("frame_{}{}.state", frame, addition_str);

    std::string full_directory = std::string(LCSV_RESOURCE_PATH) + std::string("/SimulationState/");
    std::string full_path      = full_directory;
    full_path += std::string_view{filename};

    std::ifstream file(full_path, std::ios::in);
    if (!file.is_open())
    {
        LUISA_ERROR("Unable to open state file: {}", full_path);
        return;
    }

    // std::vector<float3> sa_x_frame_saved(host_mesh_data->sa_x_frame_outer.size());
    // std::vector<float3> sa_v_frame_saved(host_mesh_data->sa_v_frame_outer.size());
    std::vector<float3> sa_q_frame_saved(host_sim_data->sa_q_outer.size());
    std::vector<float3> sa_qv_frame_saved(host_sim_data->sa_q_v_outer.size());

    std::string line;
    enum Section
    {
        None,
        Q,
        Qv,
    };
    Section current_section = None;
    uint    index           = 0;

    while (std::getline(file, line))
    {
        // LUISA_INFO("Reading line: {}", line);
        if (line.empty())
            continue;
        if (line.rfind("o State", 0) == 0)
        {
            current_section = Q;
            index           = 0;
            continue;
        }
        if (line.rfind("o Velocity", 0) == 0)
        {
            current_section = Qv;
            index           = 0;
            continue;
        }
        if (line[0] == 'v' && (current_section == Q || current_section == Qv))
        {
            std::istringstream iss(line.substr(1));
            float              x, y, z;
            iss >> x >> y >> z;
            if (current_section == Q)
            {
                if (index < host_sim_data->num_dof)
                    sa_q_frame_saved[index] = {x, y, z};
                else
                {
                    LUISA_INFO("Count of loaded q vertices exceeds the number of state q in the sim data, stopping load.");
                    file.close();
                    return;
                }
                index++;
            }
            else if (current_section == Qv)
            {
                if (index < host_sim_data->num_dof)
                    sa_qv_frame_saved[index] = {x, y, z};
                else
                {
                    LUISA_INFO("Count of loaded qv vertices exceeds the number of state q_v in the sim data, stopping load.");
                    file.close();
                    return;
                }
                index++;
            }
        }
    }
    file.close();

    buffer_copy(sa_q_frame_saved, host_sim_data->sa_q_outer);
    buffer_copy(sa_qv_frame_saved, host_sim_data->sa_q_v_outer);

    CpuParallel::parallel_for(0,
                              host_sim_data->num_verts_total,
                              [&](uint vid)
                              {
                                  float3 saved_x = fn_apply_template(host_sim_data->sa_x_to_dof_map,
                                                                     host_sim_data->sa_scaled_model_x,
                                                                     sa_q_frame_saved,
                                                                     vid);
                                  float3 saved_v = fn_apply_template(host_sim_data->sa_x_to_dof_map,
                                                                     host_sim_data->sa_scaled_model_x,
                                                                     sa_qv_frame_saved,
                                                                     vid);
                                  host_sim_data->sa_x_outer[vid] = saved_x;
                                  host_sim_data->sa_v_outer[vid] = saved_v;
                              });

    LUISA_INFO("State file loaded: {}", full_path);
}
void SolverInterface::save_mesh_to_obj(const uint frame, const std::string& addition_str)
{
    const auto& position_buffer = host_sim_data->sa_x_outer;

    // , lcs::get_scene_params().current_frame
    const auto filename = luisa::format("frame_{}{}.obj", frame, addition_str);

    std::string full_directory = std::string(LCSV_RESOURCE_PATH) + std::string("/OutputMesh/");

    {
        std::filesystem::path dir_path(full_directory);
        if (!std::filesystem::exists(dir_path))
        {
            try
            {
                std::filesystem::create_directories(dir_path);
                std::cout << "Created directory: " << dir_path << std::endl;
            }
            catch (const std::filesystem::filesystem_error& e)
            {
                std::cerr << "Error creating directory: " << e.what() << std::endl;
                return;
            }
        }
    }

    std::string full_path = full_directory;
    full_path += std::string_view{filename};
    std::ofstream file(full_path, std::ios::out);

    if (file.is_open())
    {
        file << "# Simulated Reulst" << std::endl;

        uint glocal_vert_id_prefix = 0;
        uint glocal_mesh_id_prefix = 0;

        // Cloth Part
        // if (lcs::get_scene_params().draw_cloth)
        {
            const uint num_clothes = host_mesh_data->prefix_num_verts.size() - 1;
            for (uint clothIdx = 0; clothIdx < num_clothes; clothIdx++)
            {
                const uint curr_prefix_num_verts = host_mesh_data->prefix_num_verts[clothIdx];
                const uint next_prefix_num_verts = host_mesh_data->prefix_num_verts[clothIdx + 1];
                const uint curr_prefix_num_faces = host_mesh_data->prefix_num_faces[clothIdx];
                const uint next_prefix_num_faces = host_mesh_data->prefix_num_faces[clothIdx + 1];

                {
                    file << "o mesh_" << (glocal_mesh_id_prefix + clothIdx) << std::endl;
                    for (uint vid = 0; vid < next_prefix_num_verts - curr_prefix_num_verts; vid++)
                    {
                        const auto vertex = position_buffer[curr_prefix_num_verts + vid];
                        file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
                    }

                    for (uint fid = 0; fid < next_prefix_num_faces - curr_prefix_num_faces; fid++)
                    {
                        const auto vid_prefix = glocal_vert_id_prefix + 1;
                        const auto f          = host_mesh_data->sa_faces[curr_prefix_num_faces + fid];
                        file << "f " << vid_prefix + f.x << " " << vid_prefix + f.y << " "
                             << vid_prefix + f.z << std::endl;
                    }
                }
            }
            glocal_vert_id_prefix += host_mesh_data->num_verts;
            glocal_mesh_id_prefix += 1;
        }

        file.close();
        std::cout << "OBJ file saved: " << full_path << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file: " << full_path << std::endl;
    }
}

constexpr bool print_detail = false;

constexpr uint offset_inertia          = 0;
constexpr uint offset_ground_collision = 1;
constexpr uint offset_ground_friction  = 7;
constexpr uint offset_stretch_spring   = 2;
constexpr uint offset_stretch_face     = 3;
constexpr uint offset_bending          = 4;
constexpr uint offset_abd_inertia      = 5;
constexpr uint offset_abd_ortho        = 6;

void SolverInterface::compile_compute_energy(AsyncCompiler& compiler)
{
    using namespace luisa::compute;
    const bool                   use_debug_info = false;
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    compiler.compile<1>(fn_reset_float,
                        [](Var<BufferView<float>> buffer) { buffer->write(dispatch_x(), 0.0f); });

    compiler.compile<1>(
        fn_calc_energy_inertia,
        [sa_q_tilde = sim_data->sa_q_tilde.view(), sa_system_energy = sim_data->sa_system_energy.view()](
            Var<Constitutions::SoftInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt)
        {
            auto& soft_inertia_indices = constraint.constraint_indices;
            // auto& sa_x_tilde             = constraint->sa_x_tilde;
            auto& sa_vert_mass           = constraint->sa_soft_vert_mass;
            auto& sa_stiffness_dirichlet = constraint->sa_stiffness_dirichlet;

            const Uint index = dispatch_id().x;
            const Uint vid   = soft_inertia_indices.read(index);

            Float energy = 0.0f;

            {
                Float3 x_new   = sa_q->read(vid);
                Float3 x_tilde = sa_q_tilde->read(vid);
                Float  mass    = sa_vert_mass->read(vid);
                // Bool        is_fixed       = sa_is_fixed->read(vid);
                const Float squared_inv_dt = 1.0f / (substep_dt * substep_dt);
                energy = squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);
                // $if(is_fixed)
                {
                    // Dirichlet boundary energy
                    Float stiffness_dirichlet = sa_stiffness_dirichlet->read(vid);
                    energy                    = stiffness_dirichlet * energy;
                };
                if constexpr (print_detail)
                    device_log("vid {}, inertia energy {} (invdt2 = {}, |dx|2 = {}, diff = {}) mass = {}",
                               vid,
                               energy,
                               squared_inv_dt,
                               (length_squared_vec(x_new - x_tilde)),
                               x_new - x_tilde,
                               mass);
            };


            energy = ParallelIntrinsic::block_intrinsic_reduce(vid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(vid % 256 == 0)
            {
                // sa_system_energy->write(vid / 256, energy);
                sa_system_energy->atomic(offset_inertia).fetch_add(energy);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_calc_energy_ground_collision,
        [sa_rest_vert_area              = mesh_data->sa_rest_vert_area.view(),
         sa_is_fixed                    = mesh_data->sa_is_fixed.view(),
         sa_contact_active_verts_offset = sim_data->sa_contact_active_verts_offset.view(),
         sa_contact_active_verts_d_hat  = sim_data->sa_contact_active_verts_d_hat.view(),
         sa_contact_active_verts_friction_coeff = sim_data->sa_contact_active_verts_friction_coeff.view(),
         sa_x_step_start  = sim_data->sa_x_step_start.view(),
         sa_system_energy = sim_data->sa_system_energy.view()](
            Var<BufferView<float3>> sa_x, Float floor_y, Bool use_ground_collision, Float stiffness, Uint collision_type)
        {
            const Uint vid = dispatch_id().x;

            Float energy_repulsive = 0.0f;
            Float energy_friction  = 0.0f;
            Bool  is_fixed         = sa_is_fixed->read(vid) != 0;

            $if(use_ground_collision & !is_fixed)
            {
                Float d_hat     = sa_contact_active_verts_d_hat->read(vid);
                Float thickness = sa_contact_active_verts_offset->read(vid);
                Float area      = sa_rest_vert_area->read(vid);
                Float stiff     = stiffness * area;

                Float3 normal = make_float3(0.0f, 1.0f, 0.0f);

                Float3 x_k = sa_x->read(vid);
                Float3 x_0 = sa_x_step_start->read(vid);

                // Repulsion Part
                Float curr_dist = x_k.y - floor_y;
                $if(curr_dist - thickness < d_hat)
                {
                    $if(collision_type == 0)
                    {
                        Float C          = curr_dist - d_hat - thickness;
                        energy_repulsive = 0.5f * stiff * C * C;
                    }
                    $else
                    {
                        energy_repulsive = stiff * ipc::barrier(curr_dist - thickness, d_hat);
                    };
                };

                // Friction Part
                Float init_dist = x_0.y - floor_y;
                $if(init_dist - thickness < d_hat)
                {
                    Float3 rel_dx = x_k - x_0;

                    Float k1 = 0.0f;
                    $if(collision_type == 0)
                    {
                        Float C = init_dist - thickness - d_hat;
                        k1      = stiff * C;
                    }
                    $else
                    {
                        k1 = stiff * ipc::barrier_first_derivative(init_dist - thickness, d_hat);
                    };

                    Float friction_mu  = sa_contact_active_verts_friction_coeff->read(vid);
                    Float friction_eps = Friction::ando_barrier::friction_eps;

                    auto lambda = -k1 * friction_mu;
                    energy_friction =
                        Friction::ipc_barrier::compute_friction_energy(lambda, normal, rel_dx, friction_eps);
                };
            };

            Float2 energy =
                ParallelIntrinsic::block_intrinsic_reduce(vid,
                                                          make_float2(energy_repulsive, energy_friction),
                                                          ParallelIntrinsic::warp_reduce_op_sum<float2>);
            $if(vid % 256 == 0)
            {
                sa_system_energy->atomic(offset_ground_collision).fetch_add(energy.x);
                sa_system_energy->atomic(offset_ground_friction).fetch_add(energy.y);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_calc_energy_spring,
        [sa_system_energy = sim_data->sa_system_energy.view()](
            Var<Constitutions::StretchSpring<Buffer>> constraint, Var<BufferView<float3>> sa_x, Float stiffness_spring)
        {
            auto& sa_edges                    = constraint->constraint_indices;
            auto& sa_edge_rest_state_length   = constraint->sa_stretch_spring_rest_state_length;
            auto& sa_stretch_spring_stiffness = constraint->sa_stretch_spring_stiffness;

            const Uint eid    = dispatch_id().x;
            Float      energy = 0.0f;
            {
                const Uint2 edge             = sa_edges->read(eid);
                const Float rest_edge_length = sa_edge_rest_state_length->read(eid);
                Float3      diff             = sa_x->read(edge[1]) - sa_x->read(edge[0]);
                Float       orig_lengthsqr   = length_squared_vec(diff);
                Float       l                = sqrt_scalar(orig_lengthsqr);
                Float       l0               = rest_edge_length;
                Float       C                = l - l0;
                // if (C > 0.0f)
                energy = 0.5f * sa_stretch_spring_stiffness->read(eid) * C * C;
            };
            energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(eid % 256 == 0)
            {
                // sa_system_energy->write(eid / 256, energy);
                sa_system_energy->atomic(offset_stretch_spring).fetch_add(energy);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_calc_energy_stretch_face,
        [sa_system_energy = sim_data->sa_system_energy.view()](Var<Constitutions::StretchFace<Buffer>> constraint,
                                                               Var<BufferView<float3>> sa_x)
        {
            auto& sa_faces                   = constraint->constraint_indices;
            auto& sa_stretch_faces_rest_area = constraint->sa_stretch_faces_rest_area;
            auto& sa_stretch_faces_Dm_inv    = constraint->sa_stretch_faces_Dm_inv;
            auto& sa_stretch_faces_mu_lambda = constraint->sa_stretch_faces_mu_lambda;

            const Uint fid    = dispatch_id().x;
            Float      energy = 0.0f;
            {
                const Uint3 face   = sa_faces->read(fid);
                Float3 vert_pos[3] = {sa_x->read(face[0]), sa_x->read(face[1]), sa_x->read(face[2])};

                Float2x2 Dm_inv = sa_stretch_faces_Dm_inv->read(fid);
                Float    area   = sa_stretch_faces_rest_area->read(fid);

                Float2 mu_lambda    = sa_stretch_faces_mu_lambda->read(fid);
                Float  mu_cloth     = mu_lambda[0];
                Float  lambda_cloth = mu_lambda[1];

                energy = StretchEnergy::compute_energy(
                    vert_pos[0], vert_pos[1], vert_pos[2], Dm_inv, mu_cloth, lambda_cloth, area);
            };
            energy = ParallelIntrinsic::block_intrinsic_reduce(fid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if(fid % 256 == 0)
            {
                sa_system_energy->atomic(offset_stretch_face).fetch_add(energy);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_calc_energy_bending,
        [sa_system_energy = sim_data->sa_system_energy.view()](
            Var<Constitutions::BendingEdge<Buffer>> constraint, Var<BufferView<float3>> sa_x, Float scaling)
        {
            auto& sa_edges                    = constraint->constraint_indices;
            auto& sa_bending_edges_rest_angle = constraint->sa_bending_edges_rest_angle;
            auto& sa_bending_edges_rest_area  = constraint->sa_bending_edges_rest_area;
            auto& sa_bending_edges_stiffness  = constraint->sa_bending_edges_stiffness;
            // auto& sa_bending_edges_Q          = constraint->sa_bending_edges_Q;

            const Uint eid    = dispatch_id().x;
            Float      energy = 0.0f;
            {
                const Uint4 edge = sa_edges->read(eid);

                Float3 vert_pos[4] = {
                    sa_x.read(edge[0]),
                    sa_x.read(edge[1]),
                    sa_x.read(edge[2]),
                    sa_x.read(edge[3]),
                };
                Float rest_angle = sa_bending_edges_rest_angle->read(eid);
                Float angle = BendingEnergy::compute_theta(vert_pos[0], vert_pos[1], vert_pos[2], vert_pos[3]);
                Float delta_angle = angle - rest_angle;
                Float area        = sa_bending_edges_rest_area->read(eid);
                energy = 0.5f * sa_bending_edges_stiffness->read(eid) * scaling * area * delta_angle * delta_angle;

                // const Float4x4 m_Q = sa_bending_edges_Q->read(eid);
                // for (uint ii = 0; ii < 4; ii++)
                // {
                //     for (uint jj = 0; jj < 4; jj++)
                //     {
                //         // E_b = 1/2 (x^T)Qx = 1/2 Sigma_ij Q_ij <x_i, x_j>
                //         energy += m_Q[ii][jj] * dot(vert_pos[ii], vert_pos[jj]);
                //     }
                // }
                // energy = 0.5f * stiffness_bending * energy;
            };
            energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(eid % 256 == 0)
            {
                sa_system_energy->atomic(offset_bending).fetch_add(energy);
            };
        },
        default_option);


    compiler.compile<1>(
        fn_calc_energy_abd_inertia,
        [sa_q_tilde = sim_data->sa_q_tilde.view(), sa_system_energy = sim_data->sa_system_energy.view()](
            Var<Constitutions::AbdInertia<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q, Float substep_dt)
        {
            auto& sa_affine_bodies       = constraint.constraint_indices;
            auto& sa_vert_mass           = constraint.sa_affine_bodies_mass_matrix;
            auto& sa_stiffness_dirichlet = constraint.sa_stiffness_dirichlet;

            const Uint  body_idx    = dispatch_id().x;
            const Uint4 affine_body = sa_affine_bodies->read(body_idx);

            Float energy = 0.0f;
            {
                const Float h              = substep_dt;
                const Float squared_inv_dt = 1.0f / (h * h);
                // Bool        is_fixed       = sa_is_fixed->read(body_idx);
                Float stiffness_dirichlet = sa_stiffness_dirichlet->read(body_idx);

                auto   mass_matrix = sa_vert_mass->read(body_idx);
                Float3 delta[4]    = {
                    sa_q.read(affine_body[0]) - sa_q_tilde->read(affine_body[0]),
                    sa_q.read(affine_body[1]) - sa_q_tilde->read(affine_body[1]),
                    sa_q.read(affine_body[2]) - sa_q_tilde->read(affine_body[2]),
                    sa_q.read(affine_body[3]) - sa_q_tilde->read(affine_body[3]),
                };

                for (uint ii = 0; ii < 4; ii++)
                {
                    for (uint jj = 0; jj < 4; jj++)
                    {
                        Float mass = mass_matrix[ii][jj];
                        energy += squared_inv_dt * dot(delta[ii], delta[jj]) * mass / (2.0f);
                    }
                }

                // $if(is_fixed)
                {
                    energy *= stiffness_dirichlet;
                };
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(
                body_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(body_idx % 256 == 0)
            {
                sa_system_energy->atomic(offset_abd_inertia).fetch_add(energy);
            };
        },
        default_option);

    compiler.compile<1>(
        fn_calc_energy_abd_ortho,
        [sa_system_energy = sim_data->sa_system_energy.view()](
            Var<Constitutions::AbdOrthogonality<luisa::compute::Buffer>> constraint, Var<BufferView<float3>> sa_q)
        {
            auto& abd_ortho_indices = constraint.constraint_indices;
            auto& abd_kappa         = constraint->abd_kappa;
            auto& abd_volume        = constraint->abd_volume;

            const Uint body_idx = dispatch_id().x;

            const Uint3 indices = abd_ortho_indices->read(body_idx);

            Float energy = 0.0f;
            {
                Float3x3 A;
                A[0] = sa_q->read(indices[0]);
                A[1] = sa_q->read(indices[1]);
                A[2] = sa_q->read(indices[2]);
                // Float3   p;
                // AffineBodyDynamics::extract_Ap_from_q(sa_q, body_idx, A, p);
                // device_log("A = {}, AAT = {}", A, A * transpose(A));
                for (uint ii = 0; ii < 3; ii++)
                {
                    for (uint jj = 0; jj < 3; jj++)
                    {
                        Float term = dot(A[ii], A[jj]) - (ii == jj ? 1.0f : 0.0f);
                        energy += term * term;
                    }
                }
                Float stiffness_ortho = abd_kappa->read(body_idx);
                Float volume          = abd_volume->read(body_idx);
                energy *= stiffness_ortho * volume;
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(
                body_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
            $if(body_idx % 256 == 0)
            {
                sa_system_energy->atomic(offset_abd_ortho).fetch_add(energy);
            };
        },
        default_option);
}
void SolverInterface::device_compute_elastic_energy(luisa::compute::Stream&        stream,
                                                    std::map<std::string, double>& energy_list)
{
    const luisa::compute::Buffer<float3>& curr_x = sim_data->sa_x;
    const luisa::compute::Buffer<float3>& curr_q = sim_data->sa_q;

    stream << fn_reset_float(sim_data->sa_system_energy).dispatch(8);

    const auto& soft_inertia_constitution = sim_data->get_soft_inertia_data();
    if (soft_inertia_constitution.is_valid())
    {
        stream << fn_calc_energy_inertia(soft_inertia_constitution, curr_x, get_scene_params().get_substep_dt())
                      .dispatch(host_sim_data->num_verts_soft);
    }

    const auto& abd_inertia_data = sim_data->get_abd_inertia_data();
    if (abd_inertia_data.is_valid())
    {
        // const auto abd_q_span = curr_q.view(host_sim_data->num_verts_soft, host_sim_data->num_affine_bodies * 4);
        stream << fn_calc_energy_abd_inertia(abd_inertia_data, curr_q, get_scene_params().get_substep_dt())
                      .dispatch(abd_inertia_data.get_num_indices());
    }

    const auto& abd_ortho_data = sim_data->get_abd_orthogonality_data();
    if (abd_ortho_data.is_valid())
    {
        stream << fn_calc_energy_abd_ortho(abd_ortho_data, curr_q).dispatch(abd_ortho_data.get_num_indices());
    }


    if (get_scene_params().use_floor)
    {
        stream << fn_calc_energy_ground_collision(curr_x,
                                                  get_scene_params().floor.y,
                                                  get_scene_params().use_floor,
                                                  get_scene_params().stiffness_collision,
                                                  get_scene_params().contact_energy_type)
                      .dispatch(mesh_data->num_verts);
    }

    const auto& stretch_spring_constitution = sim_data->get_stretch_spring_data();
    if (stretch_spring_constitution.is_valid())
    {
        stream << fn_calc_energy_spring(stretch_spring_constitution, curr_x, get_scene_params().stiffness_spring)
                      .dispatch(stretch_spring_constitution.get_num_indices());
    }

    const auto& stretch_face_constitution = sim_data->get_stretch_face_data();
    if (stretch_face_constitution.is_valid())
    {
        stream << fn_calc_energy_stretch_face(stretch_face_constitution, curr_x)
                      .dispatch(stretch_face_constitution.get_num_indices());
    }

    const auto& bending_edge_constitution = sim_data->get_bending_edge_data();
    if (bending_edge_constitution.is_valid())
    {
        stream << fn_calc_energy_bending(bending_edge_constitution, curr_x, get_scene_params().get_bending_stiffness_scaling())
                      .dispatch(bending_edge_constitution.get_num_indices());
    }

    auto& host_energy = host_sim_data->sa_system_energy;
    stream << sim_data->sa_system_energy.view(0, 8).copy_to(host_energy.data()) << luisa::compute::synchronize();

    // float total_energy = std::reduce(&host_energy[0], &host_energy[8], 0.0f);
    // if (get_scene_params().print_system_energy)
    // {
    //     LUISA_INFO("    Energy {} = inertia {} + ground {} + stretch {} + bending {}",
    //                total_energy,
    //                host_energy[offset_inertia],
    //                host_energy[offset_ground_collision],
    //                host_energy[offset_stretch_spring],
    //                host_energy[offset_bending]);
    // }
    energy_list.insert(std::make_pair("Inertia Soft Body", host_energy[offset_inertia]));
    energy_list.insert(std::make_pair("Inertia Rigid Body", host_energy[offset_abd_inertia]));
    energy_list.insert(std::make_pair("Ground Collision", host_energy[offset_ground_collision]));
    energy_list.insert(std::make_pair("Ground Friction", host_energy[offset_ground_friction]));
    energy_list.insert(std::make_pair("Stretch Spring", host_energy[offset_stretch_spring]));
    energy_list.insert(std::make_pair("Stretch Face", host_energy[offset_stretch_face]));
    energy_list.insert(std::make_pair("Cloth Bending", host_energy[offset_bending]));
    energy_list.insert(std::make_pair("ABD Orthogonality", host_energy[offset_abd_ortho]));
    // auto total_energy = std::accumulate(energy_list.begin(),
    //                                     energy_list.end(),
    //                                     0.0,
    //                                     [](double sum, const auto& pair) { return sum + pair.second; });
    // return total_energy;
    // return energy_inertia + energy_goundcollision + energy_spring;
};

}  // namespace lcs