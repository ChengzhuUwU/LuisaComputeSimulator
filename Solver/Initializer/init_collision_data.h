#pragma once

#include "SimulationCore/base_mesh.h"
#include "SimulationCore/simulation_data.h"
#include "Initializer/initializer_utils.h"

namespace lcsv::Initializater 
{

template<template<typename...> typename BufferType>
inline void resize_collision_data(
    luisa::compute::Device& device, 
    lcsv::MeshData<std::vector>* mesh_data, 
    lcsv::CollisionDataCCD<BufferType>* collision_data)
{
    const uint num_verts = mesh_data->num_verts;
    const uint num_edges = mesh_data->num_edges;
    
    const uint per_element_count_BP = 64;
    const uint per_element_count_NP = 32;
    
    lcsv::Initializater::resize_buffer(device, collision_data->broad_phase_collision_count, 4); 
    lcsv::Initializater::resize_buffer(device, collision_data->narrow_phase_collision_count, 4); 
    lcsv::Initializater::resize_buffer(device, collision_data->broad_phase_list_vf, per_element_count_BP * num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->broad_phase_list_ee, per_element_count_BP * num_edges); 
    lcsv::Initializater::resize_buffer(device, collision_data->narrow_phase_indices_vv, per_element_count_NP * num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->narrow_phase_indices_ve, per_element_count_NP * num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->narrow_phase_indices_vf, per_element_count_NP * num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->narrow_phase_indices_ee, per_element_count_NP * num_edges); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_num_broad_phase_vf, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_num_broad_phase_ee, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_num_narrow_phase_vv, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_num_narrow_phase_ve, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_num_narrow_phase_vf, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_num_narrow_phase_ee, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_prefix_narrow_phase_vv, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_prefix_narrow_phase_ve, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_prefix_narrow_phase_vf, num_verts); 
    lcsv::Initializater::resize_buffer(device, collision_data->per_vert_prefix_narrow_phase_ee, num_verts); 
    collision_data->collision_indirect_cmd_buffer_broad_phase = device.create_indirect_dispatch_buffer(2); 
    collision_data->collision_indirect_cmd_buffer_narrow_phase = device.create_indirect_dispatch_buffer(4); 

    // BufferType<uint> broad_phase_collision_count; 
    // BufferType<uint> narrow_phase_collision_count; 
    // BufferType<uint> broad_phase_list_vf;
    // BufferType<uint> broad_phase_list_ee;
    // BufferType<uint2> narrow_phase_indices_vv; // 0
    // BufferType<uint3> narrow_phase_indices_ve; // 1
    // BufferType<uint4> narrow_phase_indices_vf; // 2
    // BufferType<uint4> narrow_phase_indices_ee; // 3
    // BufferType<uint> per_vert_num_broad_phase_vf; 
    // BufferType<uint> per_vert_num_broad_phase_ee; 
    // BufferType<uint> per_vert_num_narrow_phase_vv; 
    // BufferType<uint> per_vert_num_narrow_phase_ve; 
    // BufferType<uint> per_vert_num_narrow_phase_vf; 
    // BufferType<uint> per_vert_num_narrow_phase_ee; 
    // BufferType<uint> per_vert_prefix_narrow_phase_vv; 
    // BufferType<uint> per_vert_prefix_narrow_phase_ve; 
    // BufferType<uint> per_vert_prefix_narrow_phase_vf; 
    // BufferType<uint> per_vert_prefix_narrow_phase_ee; 

    const uint bytes = 
        sizeof(uint) * collision_data->broad_phase_list_vf.size() * 4 +
        sizeof(uint) * collision_data->broad_phase_list_ee.size() * 4 +
        sizeof(uint2) * collision_data->narrow_phase_indices_vv.size() * 4 +
        sizeof(uint3) * collision_data->narrow_phase_indices_ve.size() * 4 +
        sizeof(uint4) * collision_data->narrow_phase_indices_vf.size() * 4 +
        sizeof(uint4) * collision_data->narrow_phase_indices_ee.size() * 4 +
        sizeof(uint) * collision_data->per_vert_num_broad_phase_vf.size() * 4 +
        sizeof(uint) * collision_data->per_vert_num_broad_phase_ee.size() * 4 +
        sizeof(uint) * collision_data->per_vert_num_narrow_phase_vv.size() * 4 +
        sizeof(uint) * collision_data->per_vert_num_narrow_phase_ve.size() * 4 +
        sizeof(uint) * collision_data->per_vert_num_narrow_phase_vf.size() * 4 +
        sizeof(uint) * collision_data->per_vert_num_narrow_phase_ee.size() * 4 +
        sizeof(uint) * collision_data->per_vert_prefix_narrow_phase_vv.size() * 4 +
        sizeof(uint) * collision_data->per_vert_prefix_narrow_phase_ve.size() * 4 +
        sizeof(uint) * collision_data->per_vert_prefix_narrow_phase_vf.size() * 4 +
        sizeof(uint) * collision_data->per_vert_prefix_narrow_phase_ee.size() * 4;
    luisa::log_info("Allocated collision buffer size {} MB", bytes / (1024 * 1024));
    if (float(bytes) / (1024 * 1024) < 1.0f) luisa::log_info("Allocated collision buffer size {} GB", bytes / (1024 * 1024 * 1024));
}


}