#include "SimulationCore/solver_interface.h"
#include "Core/scalar.h"
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

void SolverInterface::physics_step_prev_operation()
{

}
void SolverInterface::physics_step_post_operation()
{
    
}

void SolverInterface::restart_system()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        auto rest_pos = host_mesh_data->sa_rest_x[vid];
        host_mesh_data->sa_x_frame_outer[vid] = rest_pos;

        auto rest_vel = host_mesh_data->sa_rest_v[vid];
        host_mesh_data->sa_v_frame_outer[vid] = rest_vel;
    });
}
void SolverInterface::save_current_frame_state()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        host_mesh_data->sa_x_frame_saved[vid] = host_mesh_data->sa_x_frame_outer[vid];
        host_mesh_data->sa_v_frame_saved[vid] = host_mesh_data->sa_v_frame_outer[vid];
    });
}
void SolverInterface::save_current_frame_state_to_host(const uint frame, const std::string& addition_str)
{
    save_current_frame_state();

    const std::string filename = std::format("frame_{}{}.state", frame, addition_str);

    std::string full_directory = std::string(LCSV_RESOURCE_PATH) + std::string("/SimulationState/");
    
    {
        std::filesystem::path dir_path(full_directory);
        if (!std::filesystem::exists(dir_path)) 
        {
            try 
            {
                std::filesystem::create_directories(dir_path);
                luisa::log_info("Created directory: {}", dir_path.string());
            } 
            catch (const std::filesystem::filesystem_error& e) 
            {
                luisa::log_error("Error creating directory: {}", e.what());
                return;
            }
        }
    }

    std::string full_path = full_directory + filename;
    std::ofstream file(full_path, std::ios::out);

    
    if (file.is_open()) 
    {
        file << "o position" << std::endl;
        for (uint vid = 0; vid < host_mesh_data->num_verts; vid++) 
        {
            const auto vertex = host_mesh_data->sa_x_frame_saved[vid];
            file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
        }
        file << "o velocity" << std::endl;
        for (uint vid = 0; vid < host_mesh_data->num_verts; vid++) 
        {
            const auto vel = host_mesh_data->sa_v_frame_saved[vid];
            file << "v " << vel.x << " " << vel.y << " " << vel.z << std::endl;
        }
     
        file.close();
        luisa::log_info("State file saved: {}", full_path);
    } 
    else 
    {
        luisa::log_error("Unable to open file: {}", full_path);
    }
}
void SolverInterface::load_saved_state()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        auto saved_pos = host_mesh_data->sa_x_frame_saved[vid];
        host_mesh_data->sa_x_frame_outer[vid] = saved_pos;

        auto saved_vel = host_mesh_data->sa_v_frame_saved[vid];
        host_mesh_data->sa_v_frame_outer[vid] = saved_vel;
    });
}
void SolverInterface::load_saved_state_from_host(const uint frame, const std::string& addition_str)
{
    const std::string filename = std::format("frame_{}{}.state", frame, addition_str);

    std::string full_directory = std::string(LCSV_RESOURCE_PATH) + std::string("/SimulationState/");
    std::string full_path = full_directory + filename;

    std::ifstream file(full_path, std::ios::in);
    if (!file.is_open()) 
    {
        luisa::log_error("Unable to open state file: {}", full_path);
        return;
    }

    std::string line;
    enum Section { None, Position, Velocity };
    Section current_section = None;
    uint pos_vid = 0, vel_vid = 0;

    while (std::getline(file, line)) 
    {
        if (line.empty()) continue;
        if (line.rfind("o position", 0) == 0) 
        {
            current_section = Position;
            pos_vid = 0;
            continue;
        }
        if (line.rfind("o velocity", 0) == 0) 
        {
            current_section = Velocity;
            vel_vid = 0;
            continue;
        }
        if (line[0] == 'v' && (current_section == Position || current_section == Velocity)) 
        {
            std::istringstream iss(line.substr(1));
            float x, y, z;
            iss >> x >> y >> z;
            if (current_section == Position) 
            {
                if (pos_vid < host_mesh_data->num_verts) host_mesh_data->sa_x_frame_saved[pos_vid] = {x, y, z};
                pos_vid++;
            } 
            else if (current_section == Velocity) 
            {
                if (vel_vid < host_mesh_data->num_verts) host_mesh_data->sa_v_frame_saved[vel_vid] = {x, y, z};
                vel_vid++;
            }
        }
    }
    file.close();

    if (pos_vid != host_mesh_data->num_verts || vel_vid != host_mesh_data->num_verts)
    {
        luisa::log_error("numVerts read {} does NOT match numVerts of current mesh {}", pos_vid, host_mesh_data->num_verts);
    }

    load_saved_state();

    luisa::log_info("State file loaded: {}", full_path);

}
void SolverInterface::save_mesh_to_obj(const uint frame, const std::string& addition_str)
{
    // , lcs::get_scene_params().current_frame
    const std::string filename = std::format("frame_{}{}.obj", frame, addition_str);

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

    std::string full_path = full_directory + filename;
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
                    for (uint vid = 0; vid < next_prefix_num_verts - curr_prefix_num_verts; vid++) {
                        const auto vertex = host_mesh_data->sa_x_frame_outer[curr_prefix_num_verts + vid];
                        file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
                    }
    
                    for (uint fid = 0; fid < next_prefix_num_faces - curr_prefix_num_faces; fid++) {
                        const auto vid_prefix = glocal_vert_id_prefix + 1;
                        const auto f = host_mesh_data->sa_faces[curr_prefix_num_faces + fid];
                        file << "f " << vid_prefix + f.x << " " << vid_prefix + f.y << " " << vid_prefix + f.z << std::endl;
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

// Evaluate Energy
double SolverInterface::host_compute_elastic_energy(const std::vector<float3>& curr_x)
{
    auto compute_energy_inertia = [](
        const uint vid, 
        const std::vector<float3>& sa_x, 
        const std::vector<float3>& sa_x_tilde,
        const std::vector<float> sa_vert_mass, 
        const std::vector<uint> sa_is_fixed, 
        const float substep_dt, const float stiffness_dirichlet)
    {
        const float squared_inv_dt = 1.0f / (substep_dt * substep_dt);
        float3 x_new = sa_x[vid];
        float3 x_tilde = sa_x_tilde[vid];
        float mass = sa_vert_mass[vid];
        bool is_fixed = sa_is_fixed[vid];
        float energy = squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);;
        if (is_fixed)
        {
            // Dirichlet boundary energy
            // energy = stiffness_dirichlet * squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);
            energy += stiffness_dirichlet * length_squared_vec(x_new - x_tilde) / (2.0f);
        }
        else 
        {
        }
        return energy;
    };
    auto compute_energy_goundcollision = [](
        const uint vid, 
        const std::vector<float3>& sa_x, 
        const std::vector<uint>& sa_is_fixed, 
        const std::vector<float>& sa_rest_vert_area, 
        const float3& floor,
        const bool use_floor,
        const float d_hat, const float thickness)
    {
        if (!use_floor) return 0.0f;
        if (sa_is_fixed[vid]) return 0.0f;
        float3 x_k = sa_x[vid];
        float diff = x_k.y - floor.y;
        if (diff < d_hat + thickness)
        {
            float C = d_hat + thickness - diff;
            float area = sa_rest_vert_area[vid];
            float stiff = 1e7 * area;
            return 0.5f * stiff * C * C;
        }
        else 
        {
            return 0.0f;
        }
    };
    auto compute_energy_spring = [](
        const uint eid, 
        const std::vector<float3>& sa_x, 
        const std::vector<uint2>& sa_edges,
        const std::vector<float> sa_edge_rest_state_length, 
        const float stiffness_spring)
    {
        const uint2 edge = sa_edges[eid];
        const float rest_edge_length = sa_edge_rest_state_length[eid];
        float3 diff = sa_x[edge[1]] - sa_x[edge[0]];
        // float orig_lengthsqr = length_squared_vec(diff);
        // float l = sqrt_scalar(orig_lengthsqr);
        float l = max_scalar(length_vec(diff), Epsilon);
        float l0 = rest_edge_length;
        float C = l - l0;
        float energy = 0.0f;
        // if (C > 0.0f)
            energy = 0.5f * stiffness_spring * C * C;
        return energy;
    };
    auto compute_energy_bending = [](
        const uint eid, 
        const std::vector<float3>& sa_x, 
        const std::vector<uint4>& sa_bending_edges,
        const std::vector<float4x4> sa_bending_edges_Q, 
        const float stiffness_bending)
    {
        const uint4 edge = sa_bending_edges[eid];
        const float4x4 m_Q = sa_bending_edges_Q[eid];
        float3 vert_pos[4] = {
            sa_x[edge[0]],
            sa_x[edge[1]],
            sa_x[edge[2]],
            sa_x[edge[3]],
        };
        float energy = 0.f;
        for (uint ii = 0; ii < 4; ii++) 
        {
            for (uint jj = 0; jj < 4; jj++) 
            {
                // E_b = 1/2 (x^T)Qx = 1/2 Sigma_ij Q_ij <x_i, x_j>
                energy += m_Q[ii][jj] * luisa::dot(vert_pos[ii], vert_pos[jj]); 
            }
        }
        energy = 0.5f * stiffness_bending * energy;
        return energy;
    };

    double energy_inertia = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
    {
        return compute_energy_inertia(vid, 
            curr_x, 
            host_sim_data->sa_x_tilde, 
            host_mesh_data->sa_vert_mass, 
            host_mesh_data->sa_is_fixed,
            get_scene_params().get_substep_dt(),
            get_scene_params().stiffness_dirichlet
        );
    });
    double energy_goundcollision = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
    {
        return compute_energy_goundcollision(vid, 
            curr_x,
            host_mesh_data->sa_is_fixed, 
            host_mesh_data->sa_rest_vert_area, 
            get_scene_params().floor, get_scene_params().use_floor, 
            get_scene_params().d_hat, get_scene_params().thickness);;
    });
    double energy_spring = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_edges, [&](const uint eid)
    {
        return compute_energy_spring(eid, 
            curr_x, 
            host_sim_data->sa_stretch_springs, 
            host_sim_data->sa_stretch_spring_rest_state_length, 
            get_scene_params().stiffness_spring);
    });
    double energy_bending = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_bending_edges, [&](const uint eid)
    {
        return compute_energy_bending(eid, 
            curr_x, 
            host_sim_data->sa_bending_edges, 
            host_sim_data->sa_bending_edges_Q, 
            get_scene_params().get_stiffness_quadratic_bending());
    });
    // luisa::log_info("    Energy = inertia {} + ground {} + stretch {}", energy_inertia, energy_goundcollision, energy_spring);
    return energy_inertia + energy_goundcollision + energy_spring + energy_bending;
};

constexpr uint offset_inertia = 0;
constexpr uint offset_ground_collision = 1;
constexpr uint offset_stretch_spring = 2;
constexpr uint offset_bending = 3;

void SolverInterface::compile_compute_energy(luisa::compute::Device& device)
{
    using namespace luisa::compute;
    const bool use_debug_info = false;
    luisa::compute::ShaderOption default_option = {.enable_debug_info = false};

    fn_reset_float = device.compile<1>([](Var<BufferView<float>> buffer)
    {
        buffer->write(dispatch_x(), 0.0f);
    });

    fn_calc_energy_inertia = device.compile<1>(
        [
            sa_x_tilde = sim_data->sa_x_tilde.view(),
            sa_vert_mass = mesh_data->sa_vert_mass.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_system_energy = sim_data->sa_system_energy.view()
        ](
            Var<BufferView<float3>> sa_x, 
            Float substep_dt,
            Float stiffness_dirichlet
        )
    {
        const Uint vid = dispatch_id().x;

        Float energy = 0.0f;

        {
            Float3 x_new = sa_x->read(vid);
            Float3 x_tilde = sa_x_tilde->read(vid);
            Float mass = sa_vert_mass->read(vid);
            Bool is_fixed = sa_is_fixed->read(vid);
            const Float squared_inv_dt = 1.0f / (substep_dt * substep_dt);
            energy = squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);
            $if (is_fixed)
            {
                // Dirichlet boundary energy
                // energy = stiffness_dirichlet * squared_inv_dt * length_squared_vec(x_new - x_tilde) * mass / (2.0f);
                energy = stiffness_dirichlet * length_squared_vec(x_new - x_tilde) / (2.0f);
            }
            $else
            {
                
            };
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(vid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        $if(vid % 256 == 0)
        {
            // sa_system_energy->write(vid / 256, energy);
            sa_system_energy->atomic(offset_inertia).fetch_add(energy);
        };
    }, default_option);

    fn_calc_energy_ground_collision = device.compile<1>(
        [
            sa_rest_vert_area = mesh_data->sa_rest_vert_area.view(),
            sa_is_fixed = mesh_data->sa_is_fixed.view(),
            sa_system_energy = sim_data->sa_system_energy.view()
        ](
            Var<BufferView<float3>> sa_x, 
            Float floor_y,
            Bool use_ground_collision,
            Float stiffness,
            Float d_hat,
            Float thickness
        )
    {
        const Uint vid = dispatch_id().x;

        Float energy = 0.0f;
        Bool is_fixed = sa_is_fixed->read(vid) != 0;
        $if (use_ground_collision & !is_fixed)
        {
            Float3 x_k = sa_x->read(vid);
            Float diff = x_k.y - floor_y;
            $if (diff < d_hat + thickness)
            {
                Float C = d_hat + thickness - diff;
                Float area = sa_rest_vert_area->read(vid);
                Float stiff = stiffness * area;
                energy = 0.5f * stiff * C * C;
            };
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(vid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        $if(vid % 256 == 0)
        {
            // sa_system_energy->write(vid / 256, energy);
            sa_system_energy->atomic(offset_ground_collision).fetch_add(energy);
        };
    }, default_option);

    fn_calc_energy_spring = device.compile<1>(
        [
            sa_edges = sim_data->sa_stretch_springs.view(),
            sa_edge_rest_state_length = sim_data->sa_stretch_spring_rest_state_length.view(),
            sa_system_energy = sim_data->sa_system_energy.view()
        ](
            Var<BufferView<float3>> sa_x,
            Float stiffness_spring
        )
    {
        const Uint eid = dispatch_id().x;
        Float energy = 0.0f;
        {
            const Uint2 edge = sa_edges->read(eid);
            const Float rest_edge_length = sa_edge_rest_state_length->read(eid);
            Float3 diff = sa_x->read(edge[1]) - sa_x->read(edge[0]);
            Float orig_lengthsqr = length_squared_vec(diff);
            Float l = sqrt_scalar(orig_lengthsqr);
            Float l0 = rest_edge_length;
            Float C = l - l0;
            // if (C > 0.0f)
                energy = 0.5f * stiffness_spring * C * C;
        };
        energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        $if (eid % 256 == 0)
        {
            // sa_system_energy->write(eid / 256, energy);
            sa_system_energy->atomic(offset_stretch_spring).fetch_add(energy);
        };
    }, default_option);

    fn_calc_energy_bending = device.compile<1>(
        [
            sa_edges = sim_data->sa_bending_edges.view(),
            sa_bending_edges_Q = sim_data->sa_bending_edges_Q.view(),
            sa_system_energy = sim_data->sa_system_energy.view()
        ](
            Var<BufferView<float3>> sa_x,
            Float stiffness_bending
        )
    {
        const Uint eid = dispatch_id().x;
        Float energy = 0.0f;
        {
            const Uint4 edge = sa_edges->read(eid);
            const Float4x4 m_Q = sa_bending_edges_Q->read(eid);
            Float3 vert_pos[4] = {
                sa_x.read(edge[0]),
                sa_x.read(edge[1]),
                sa_x.read(edge[2]),
                sa_x.read(edge[3]),
            };
            for (uint ii = 0; ii < 4; ii++) 
            {
                for (uint jj = 0; jj < 4; jj++) 
                {
                    // E_b = 1/2 (x^T)Qx = 1/2 Sigma_ij Q_ij <x_i, x_j>
                    energy += m_Q[ii][jj] * dot(vert_pos[ii], vert_pos[jj]); 
                }
            }
            energy = 0.5f * stiffness_bending * energy;
        };
        energy = ParallelIntrinsic::block_intrinsic_reduce(eid, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        $if (eid % 256 == 0)
        {
            sa_system_energy->atomic(offset_bending).fetch_add(energy);
        };
    }, default_option);
}
double SolverInterface::device_compute_elastic_energy(luisa::compute::Stream& stream, const luisa::compute::Buffer<float3>& curr_x)
{
    stream 
        << fn_reset_float(sim_data->sa_system_energy).dispatch(8)
        << fn_calc_energy_inertia(curr_x, get_scene_params().get_substep_dt(), get_scene_params().stiffness_dirichlet).dispatch(mesh_data->num_verts)
        << fn_calc_energy_ground_collision(curr_x, get_scene_params().floor.y, get_scene_params().use_floor, 1e7f, get_scene_params().d_hat, get_scene_params().thickness).dispatch(mesh_data->num_verts)
        << fn_calc_energy_spring(curr_x, get_scene_params().stiffness_spring).dispatch(host_sim_data->sa_stretch_springs.size())
        << fn_calc_energy_bending(curr_x, get_scene_params().get_stiffness_quadratic_bending()).dispatch(host_sim_data->sa_bending_edges.size())
        ;

    auto& host_energy = host_sim_data->sa_system_energy;
    stream 
        << sim_data->sa_system_energy.view(0, 8).copy_to(host_energy.data())
        << luisa::compute::synchronize();

    float total_energy = std::reduce(&host_energy[0], &host_energy[8], 0.0f);
    if (get_scene_params().print_system_energy)
    {
        luisa::log_info("    Energy {} = inertia {} + ground {} + stretch {} + bending {}", 
            total_energy,
            host_energy[offset_inertia], 
            host_energy[offset_ground_collision], 
            host_energy[offset_stretch_spring], 
            host_energy[offset_bending]);
    }
    return total_energy;
    // return energy_inertia + energy_goundcollision + energy_spring;
};

} // namespace lcs