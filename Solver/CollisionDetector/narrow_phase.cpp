#include "CollisionDetector/narrow_phase.h"
#include "CollisionDetector/accd.h"
#include "Utils/cpu_parallel.h"
#include <Eigen/Dense>

namespace lcsv 
{



void NarrowPhasesDetector::unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream)
{

}
void NarrowPhasesDetector::compile(luisa::compute::Device& device)
{

}

// inline Eigen::Vector<float, 3> float3_to_eigen3(const float3* vece)
// {
//     return 
// }
using EigenFloat3 = Eigen::Vector<float, 3>;
static inline auto float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
static inline auto eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };

void NarrowPhasesDetector::narrow_phase_query_from_vf_pair(Stream& stream, 
    const Buffer<float>& sa_toi,
    const Buffer<float3>& sa_x_begin_left, 
    const Buffer<float3>& sa_x_begin_right, 
    const Buffer<float3>& sa_x_end_left,
    const Buffer<float3>& sa_x_end_right,
    const Buffer<uint3>& sa_faces_right,
    const float thickness)
{
    std::vector<float3> host_x_begin(sa_x_begin_left.size());
    std::vector<float3> host_x_end(sa_x_begin_left.size());
    std::vector<uint3> host_face(sa_faces_right.size());
    std::vector<float> host_toi(sa_toi.size(), 1.0f);

    auto& host_count = host_ccd_data->broad_phase_collision_count;
    auto& host_list = host_ccd_data->broad_phase_list_vf;
    stream 
        << ccd_data->broad_phase_collision_count.copy_to(host_count.data()) 
        << sa_x_begin_left.copy_to(host_x_begin.data()) 
        << sa_x_end_left.copy_to(host_x_end.data()) 
        << sa_faces_right.copy_to(host_face.data()) 
        << luisa::compute::synchronize();

    const uint num_vf_broadphase = host_count[0];
    stream 
        << ccd_data->broad_phase_list_vf.view(0, num_vf_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    CpuParallel::parallel_sort(pair_view, pair_view + num_vf_broadphase, [](const uint2& left, const uint2& right)
    {
        if (left[0] == right[0]) { return left[1] < right[1]; }
        return left[0] < right[0];
    });

    std::atomic<float>* host_toi_floatview = (std::atomic<float>*)host_toi.data();

    float min_toi = CpuParallel::parallel_for_and_reduce(0, num_vf_broadphase, [&](const uint pair_idx)
    {
        const auto pair = pair_view[pair_idx];
        const uint left = pair[0];
        const uint right = pair[1];
        const uint3 right_face = host_face[right];

        if (left == right_face[0] || left == right_face[1] || left == right_face[2]) return 1.0f;
        
        EigenFloat3 p0 = float3_to_eigen3(host_x_begin[left]);
        EigenFloat3 p1 = float3_to_eigen3(host_x_end[left]);
        EigenFloat3 t00 = float3_to_eigen3(host_x_begin[right_face[0]]);
        EigenFloat3 t01 = float3_to_eigen3(host_x_begin[right_face[1]]);
        EigenFloat3 t02 = float3_to_eigen3(host_x_begin[right_face[2]]);
        EigenFloat3 t10 = float3_to_eigen3(host_x_end[right_face[0]]);
        EigenFloat3 t11 = float3_to_eigen3(host_x_end[right_face[1]]);
        EigenFloat3 t12 = float3_to_eigen3(host_x_end[right_face[2]]);

        float toi = host_accd::point_triangle_ccd(p0,  p1,
                                      t00, t01,
                                      t02, t10,
                                      t11, t12,
                                      0.0f);
        return toi;
    }, [](const float left, const float right) { return min_scalar(left, right); }, 1.25f);
    luisa::log_info("toi = {}", min_toi);

    host_ccd_data->toi_per_vert[0] = min_toi;
    stream 
        << ccd_data->toi_per_vert.view(0, 1).copy_from(&min_toi);
}

void NarrowPhasesDetector::narrow_phase_query_from_ee_pair(Stream& stream, 
    const Buffer<float>& sa_toi,
    const Buffer<float3>& sa_x_begin_left, 
    const Buffer<float3>& sa_x_begin_right, 
    const Buffer<float3>& sa_x_end_left,
    const Buffer<float3>& sa_x_end_right,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float thickness)
{
    
}


}
