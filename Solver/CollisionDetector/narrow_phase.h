#pragma once

#include "CollisionDetector/lbvh.h"
#include "Core/scalar.h"
#include "SimulationCore/simulation_data.h"
#include "SimulationCore/simulation_type.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>


namespace lcsv 
{

class NarrowPhasesDetector
{
    template<typename T>
    using Buffer = luisa::compute::Buffer<T>;
    using Stream = luisa::compute::Stream;
    using Device = luisa::compute::Device;
    
public:
    void unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream);
    void compile(luisa::compute::Device& device);
    void set_collision_data(
        CollisionDataCCD<std::vector>* host_ccd_ptr,
        CollisionDataCCD<luisa::compute::Buffer>* ccd_ptr
    ) 
    { 
        host_ccd_data = host_ccd_ptr; 
        ccd_data = ccd_ptr; 
    }

public:
    void narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
        const Buffer<float3>& sa_x_begin_left, 
        const Buffer<float3>& sa_x_begin_right, 
        const Buffer<float3>& sa_x_end_left,
        const Buffer<float3>& sa_x_end_right,
        const Buffer<uint3>& sa_faces_right,
        const float thickness);

    void narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
        const Buffer<float3>& sa_x_begin_left, 
        const Buffer<float3>& sa_x_begin_right, 
        const Buffer<float3>& sa_x_end_left,
        const Buffer<float3>& sa_x_end_right,
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
        const float thickness);

    void host_narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
        const std::vector<float3>& sa_x_begin_left, 
        const std::vector<float3>& sa_x_begin_right, 
        const std::vector<float3>& sa_x_end_left,
        const std::vector<float3>& sa_x_end_right,
        const std::vector<uint3>& sa_faces_right,
        const float thickness);

    void host_narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
        const std::vector<float3>& sa_x_begin_left, 
        const std::vector<float3>& sa_x_begin_right, 
        const std::vector<float3>& sa_x_end_left,
        const std::vector<float3>& sa_x_end_right,
        const std::vector<uint2>& sa_edges_left,
        const std::vector<uint2>& sa_edges_right,
        const float thickness);
    
    void reset_toi(Stream& stream);
    void host_reset_toi(Stream& stream);
    void download_collision_count(Stream& stream);
    float get_global_toi(Stream& stream);

public:
    CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
    CollisionDataCCD<std::vector>* host_ccd_data;

private:
    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<uint3>, float> fn_narrow_phase_vf_ccd_query;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<float3>,
        luisa::compute::BufferView<uint2>,
        luisa::compute::BufferView<uint2>, float> fn_narrow_phase_ee_ccd_query ;

    luisa::compute::Shader<1, 
        luisa::compute::BufferView<float>> fn_reset_toi;
};


// class AccdDetector
// {
// private:
//     CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
// };


}