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
    void narrow_phase_query_from_vf_pair(Stream& stream, 
        const Buffer<float>& sa_toi,
        const Buffer<float3>& sa_x_begin_left, 
        const Buffer<float3>& sa_x_begin_right, 
        const Buffer<float3>& sa_x_end_left,
        const Buffer<float3>& sa_x_end_right,
        const Buffer<uint3>& sa_faces_right,
        const float thickness);

    void narrow_phase_query_from_ee_pair(Stream& stream, 
        const Buffer<float>& sa_toi,
        const Buffer<float3>& sa_x_begin_left, 
        const Buffer<float3>& sa_x_begin_right, 
        const Buffer<float3>& sa_x_end_left,
        const Buffer<float3>& sa_x_end_right,
        const Buffer<uint2>& sa_edges_left,
        const Buffer<uint2>& sa_edges_right,
        const float thickness);

public:
    CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
    CollisionDataCCD<std::vector>* host_ccd_data;

private:

};


// class AccdDetector
// {
// private:
//     CollisionDataCCD<luisa::compute::Buffer>* ccd_data;
// };


}