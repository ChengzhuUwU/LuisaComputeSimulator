#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"

namespace lcsv::Initializater 
{


inline void resize_collision_data(
    luisa::compute::Device& device, 
    lcsv::MeshData<std::vector>* mesh_data, 
    lcsv::CollisionDataCCD<luisa::compute::Buffer>* collision_data)
{
    const uint num_verts = mesh_data->num_verts;
    const uint num_edges = mesh_data->num_edges;
    
    const uint per_element_count_BP = 64;
    const uint per_element_count_NP = 32;
    
    collision_data->broad_phase_collision_count = device.create_buffer<uint>(4); 
    collision_data->narrow_phase_collision_count = device.create_buffer<uint>(4); 
    
    collision_data->broad_phase_list_vf = device.create_buffer<uint>(per_element_count_BP * num_verts); 
    collision_data->broad_phase_list_ee = device.create_buffer<uint>(per_element_count_BP * num_edges); 
    
    collision_data->narrow_phase_indices_vv = device.create_buffer<uint2>(per_element_count_NP * num_verts); 
    collision_data->narrow_phase_indices_ve = device.create_buffer<uint3>(per_element_count_NP * num_verts); 
    collision_data->narrow_phase_indices_vf = device.create_buffer<uint4>(per_element_count_NP * num_verts); 
    collision_data->narrow_phase_indices_ee = device.create_buffer<uint4>(per_element_count_NP * num_edges); 
    
    collision_data->per_vert_num_broad_phase_vf = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_num_broad_phase_ee = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_num_narrow_phase_vv = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_num_narrow_phase_ve = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_num_narrow_phase_vf = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_num_narrow_phase_ee = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_prefix_narrow_phase_vv = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_prefix_narrow_phase_ve = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_prefix_narrow_phase_vf = device.create_buffer<uint>(num_verts); 
    collision_data->per_vert_prefix_narrow_phase_ee = device.create_buffer<uint>(num_verts); 

    collision_data->collision_indirect_cmd_buffer_broad_phase = device.create_indirect_dispatch_buffer(2); 
    collision_data->collision_indirect_cmd_buffer_narrow_phase = device.create_indirect_dispatch_buffer(4); 

    const uint bytes = 
        collision_data->broad_phase_list_vf.size_bytes() * 4 +
        collision_data->broad_phase_list_ee.size_bytes() * 4 +
        collision_data->narrow_phase_indices_vv.size_bytes() * 4 +
        collision_data->narrow_phase_indices_ve.size_bytes() * 4 +
        collision_data->narrow_phase_indices_vf.size_bytes() * 4 +
        collision_data->narrow_phase_indices_ee.size_bytes() * 4 +
        collision_data->per_vert_num_broad_phase_vf.size_bytes() * 4 +
        collision_data->per_vert_num_broad_phase_ee.size_bytes() * 4 +
        collision_data->per_vert_num_narrow_phase_vv.size_bytes() * 4 +
        collision_data->per_vert_num_narrow_phase_ve.size_bytes() * 4 +
        collision_data->per_vert_num_narrow_phase_vf.size_bytes() * 4 +
        collision_data->per_vert_num_narrow_phase_ee.size_bytes() * 4 +
        collision_data->per_vert_prefix_narrow_phase_vv.size_bytes() * 4 +
        collision_data->per_vert_prefix_narrow_phase_ve.size_bytes() * 4 +
        collision_data->per_vert_prefix_narrow_phase_vf.size_bytes() * 4 +
        collision_data->per_vert_prefix_narrow_phase_ee.size_bytes() * 4;
    luisa::log_info("Allocated collision buffer size {} MB", bytes / (1024 * 1024));
    luisa::log_info("Allocated collision buffer size {} GB", bytes / (1024 * 1024 * 1024));
}


}