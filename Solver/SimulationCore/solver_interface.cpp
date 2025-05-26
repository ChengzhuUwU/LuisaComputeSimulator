#include "SimulationCore/solver_interface.h"
#include "Core/scalar.h"
#include "Utils/cpu_parallel.h"
#include "SimulationCore/scene_params.h"
#include "MeshOperation/mesh_reader.h"

namespace lcsv 
{

void SolverInterface::physics_step_prev_operation()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        // host_mesh_data->sa_x_frame_start[vid] = host_mesh_data->sa_x_frame_end[vid];
        // host_mesh_data->sa_v_frame_start[vid] = host_mesh_data->sa_v_frame_end[vid];
    });
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
    // , lcsv::get_scene_params().current_frame
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
        // if (lcsv::get_scene_params().draw_cloth)
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
double SolverInterface::host_compute_energy(const std::vector<float3>& curr_x)
{
    auto compute_energy_inertia = [](
        const uint vid, 
        const std::vector<float3>& sa_x, 
        const std::vector<float3>& sa_x_tilde,
        const std::vector<float> sa_vert_mass, 
        const float substep_dt)
    {
        float3 x_new = sa_x[vid];
        float3 x_tilde = sa_x_tilde[vid];
        float mass = sa_vert_mass[vid];
        return length_squared_vec(x_new - x_tilde) * mass / (2 * substep_dt * substep_dt);
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
        float orig_lengthsqr = length_squared_vec(diff);
        float l = sqrt_scalar(orig_lengthsqr);
        float l0 = rest_edge_length;
        float C = l - l0;
        float energy = 0.0f;
        // if (C > 0.0f)
            energy = 0.5f * stiffness_spring * C * C;
        return energy;
    };

    double energy_inertia = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_verts, [&](const uint vid)
    {
        return compute_energy_inertia(vid, 
            curr_x, 
            host_sim_data->sa_x_tilde, 
            host_mesh_data->sa_vert_mass, 
            get_scene_params().get_substep_dt());
    });
    double energy_spring = CpuParallel::parallel_for_and_reduce_sum<double>(0, mesh_data->num_edges, [&](const uint eid)
    {
        return compute_energy_spring(eid, 
            curr_x, 
            host_mesh_data->sa_edges, 
            host_mesh_data->sa_edges_rest_state_length, 
            1e4);
    });
    // luisa::log_info("    Energy {} = inertia {} + stretch {}", energy_inertia + energy_spring, energy_inertia, energy_spring);
    return energy_inertia + energy_spring;
};

} // namespace lcsv