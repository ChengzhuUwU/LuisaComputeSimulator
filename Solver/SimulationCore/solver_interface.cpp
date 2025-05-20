#include "SimulationCore/solver_interface.h"
#include "Utils/cpu_parallel.h"
#include "MeshOperation/mesh_reader.h"

namespace lcsv 
{

void SolverInterface::physics_step_prev_operation()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        host_mesh_data->sa_x_frame_start[vid] = host_mesh_data->sa_x_frame_end[vid];
        host_mesh_data->sa_v_frame_start[vid] = host_mesh_data->sa_v_frame_end[vid];
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
        host_mesh_data->sa_x_frame_start[vid] = rest_pos;
        host_mesh_data->sa_x_frame_end[vid] = rest_pos;

        auto rest_vel = host_mesh_data->sa_rest_v[vid];
        host_mesh_data->sa_v_frame_start[vid] = rest_vel;
        host_mesh_data->sa_v_frame_end[vid] = rest_vel;
    });
}
void SolverInterface::save_current_frame_state()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        host_mesh_data->sa_x_frame_saved[vid] = host_mesh_data->sa_x_frame_end[vid];
        host_mesh_data->sa_v_frame_saved[vid] = host_mesh_data->sa_v_frame_end[vid];
    });
    
}
void SolverInterface::load_saved_state()
{
    CpuParallel::parallel_for(0, host_mesh_data->num_verts, [&](uint vid)
    {
        auto saved_pos = host_mesh_data->sa_x_frame_saved[vid];
        host_mesh_data->sa_x_frame_start[vid] = saved_pos;
        host_mesh_data->sa_x_frame_end[vid] = saved_pos;

        auto saved_vel = host_mesh_data->sa_v_frame_saved[vid];
        host_mesh_data->sa_v_frame_start[vid] = saved_vel;
        host_mesh_data->sa_v_frame_end[vid] = saved_vel;
    });
}
void SolverInterface::save_mesh_to_obj(const uint frame, const std::string& addition_str)
{
    // , lcsv::get_scene_params().current_frame
    const std::string filename = std::format("frame_{}{}.obj", frame, addition_str);

    const std::string SELF_RESOURCES_PATH = "/Users/huohuo/Desktop/Project/LuisaComputeSolver/Resources";
    std::string full_directory = std::string(SELF_RESOURCES_PATH) + std::string("/OutputMesh/");
    
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
            // for (uint clothIdx = 0; clothIdx < cloth_data.num_cloths; clothIdx++) 
            const uint clothIdx = 0;
            {
                file << "o mesh_" << (glocal_mesh_id_prefix + clothIdx) << std::endl;
                for (uint vid = 0; vid < host_mesh_data->num_verts; vid++) {
                    const auto vertex = host_mesh_data->sa_x_frame_end[vid];
                    file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
                }

                for (uint fid = 0; fid < host_mesh_data->num_faces; fid++) {
                    const auto vid_prefix = glocal_vert_id_prefix + 1;
                    const auto f = host_mesh_data->sa_faces[fid];
                    file << "f " << vid_prefix + f.x << " " << vid_prefix + f.y << " " << vid_prefix + f.z << std::endl;
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


} // namespace lcsv