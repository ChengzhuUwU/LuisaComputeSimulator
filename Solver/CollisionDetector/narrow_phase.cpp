#include "CollisionDetector/narrow_phase.h"
#include "CollisionDetector/accd.hpp"
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

void NarrowPhasesDetector::narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
    Buffer<float>& sa_toi,
    const Buffer<float3>& sa_x_begin_left, 
    const Buffer<float3>& sa_x_begin_right, 
    const Buffer<float3>& sa_x_end_left,
    const Buffer<float3>& sa_x_end_right,
    const Buffer<uint3>& sa_faces_right,
    const float thickness)
{
    
}
void NarrowPhasesDetector::host_narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
    std::vector<float>& sa_toi,
    const std::vector<float3>& sa_x_begin_left, 
    const std::vector<float3>& sa_x_begin_right, 
    const std::vector<float3>& sa_x_end_left,
    const std::vector<float3>& sa_x_end_right,
    const std::vector<uint3>& sa_faces_right,
    const float thickness)
{
    auto& host_count = host_ccd_data->broad_phase_collision_count;
    auto& host_list = host_ccd_data->broad_phase_list_vf;
    stream 
        << ccd_data->broad_phase_collision_count.copy_to(host_count.data()) 
        << luisa::compute::synchronize();

    const uint num_vf_broadphase = host_count[0];
    const uint num_ee_broadphase = host_count[1];
    stream 
        << ccd_data->broad_phase_list_vf.view(0, num_vf_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase);
    luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_vf_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

    std::atomic<float>* host_toi_floatview = (std::atomic<float>*)sa_toi.data();

    float min_toi = host_accd::line_search_max_t;
    min_toi = CpuParallel::parallel_for_and_reduce(0, num_vf_broadphase, [&](const uint pair_idx)
    {
        const auto pair = pair_view[pair_idx];
        const uint left = pair[0];
        const uint right = pair[1];
        const uint3 right_face = sa_faces_right[right];

        if (left == right_face[0] || left == right_face[1] || left == right_face[2]) return host_accd::line_search_max_t;
        
        EigenFloat3 t0_p =  float3_to_eigen3(sa_x_begin_left[left]);
        EigenFloat3 t1_p =  float3_to_eigen3(sa_x_end_left[left]);
        EigenFloat3 t0_f0 = float3_to_eigen3(sa_x_begin_right[right_face[0]]);
        EigenFloat3 t0_f1 = float3_to_eigen3(sa_x_begin_right[right_face[1]]);
        EigenFloat3 t0_f2 = float3_to_eigen3(sa_x_begin_right[right_face[2]]);
        EigenFloat3 t1_f0 = float3_to_eigen3(sa_x_end_right[right_face[0]]);
        EigenFloat3 t1_f1 = float3_to_eigen3(sa_x_end_right[right_face[1]]);
        EigenFloat3 t1_f2 = float3_to_eigen3(sa_x_end_right[right_face[2]]);

        float toi = host_accd::point_triangle_ccd(t0_p,  t1_p,
                                      t0_f0, t0_f1,
                                      t0_f2, t1_f0,
                                      t1_f1, t1_f2,
                                      thickness);

        if (toi != host_accd::line_search_max_t) luisa::log_info("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", pair_idx, toi, left, right, right_face);
        return toi;
    }, [](const float left, const float right) { return min_scalar(left, right); }, host_accd::line_search_max_t);

    min_toi /= host_accd::line_search_max_t;
    if (min_toi < 1e-5)
    {
        luisa::log_error("toi is too small : {}", min_toi);
    }
    luisa::log_info("toi = {}", min_toi);
    sa_toi[0] = min_toi;

}

void NarrowPhasesDetector::narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
    Buffer<float>& sa_toi,
    const Buffer<float3>& sa_x_begin_a, 
    const Buffer<float3>& sa_x_begin_b, 
    const Buffer<float3>& sa_x_end_a,
    const Buffer<float3>& sa_x_end_b,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float thickness)
{
    
}
void NarrowPhasesDetector::host_narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
    std::vector<float>& sa_toi,
    const std::vector<float3>& sa_x_begin_a, 
    const std::vector<float3>& sa_x_begin_b, 
    const std::vector<float3>& sa_x_end_a,
    const std::vector<float3>& sa_x_end_b,
    const std::vector<uint2>& sa_edges_left,
    const std::vector<uint2>& sa_edges_right,
    const float thickness)
{
    auto& host_count = host_ccd_data->broad_phase_collision_count;
    auto& host_list = host_ccd_data->broad_phase_list_ee;
    // stream 
    //     << ccd_data->broad_phase_collision_count.copy_to(host_count.data()) 
    //     << luisa::compute::synchronize();

    const uint num_ee_broadphase = host_count[1];
    stream 
        << ccd_data->broad_phase_list_ee.view(0, num_ee_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_ee_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

    std::atomic<float>* host_toi_floatview = (std::atomic<float>*)sa_toi.data();

    float min_toi = 1.25f;
    min_toi = CpuParallel::parallel_for_and_reduce(0, num_ee_broadphase, [&](const uint pair_idx)
    {
        const auto pair = pair_view[pair_idx];
        const uint left = pair[0];
        const uint right = pair[1];
        const uint2 left_edge  = sa_edges_left[left];
        const uint2 right_edge = sa_edges_right[right];

        if (
            left_edge[0] == right_edge[0] || 
            left_edge[0] == right_edge[1] || 
            left_edge[1] == right_edge[0] || 
            left_edge[1] == right_edge[1]) return host_accd::line_search_max_t;
        
        EigenFloat3 ea_t0_p0 = float3_to_eigen3(sa_x_begin_a[left_edge[0]]);
        EigenFloat3 ea_t0_p1 = float3_to_eigen3(sa_x_begin_a[left_edge[1]]);
        EigenFloat3 eb_t0_p0 = float3_to_eigen3(sa_x_begin_b[right_edge[0]]);
        EigenFloat3 eb_t0_p1 = float3_to_eigen3(sa_x_begin_b[right_edge[1]]);
        EigenFloat3 ea_t1_p0 = float3_to_eigen3(sa_x_end_a[left_edge[0]]);
        EigenFloat3 ea_t1_p1 = float3_to_eigen3(sa_x_end_a[left_edge[1]]);
        EigenFloat3 eb_t1_p0 = float3_to_eigen3(sa_x_end_b[right_edge[0]]);
        EigenFloat3 eb_t1_p1 = float3_to_eigen3(sa_x_end_b[right_edge[1]]);

        float toi = host_accd::edge_edge_ccd(
            ea_t0_p0, 
            ea_t0_p1, 
            eb_t0_p0, 
            eb_t0_p1, 
            ea_t1_p0, 
            ea_t1_p1, 
            eb_t1_p0, 
            eb_t1_p1, 
            thickness);

        if (toi != host_accd::line_search_max_t) luisa::log_info("BroadPhase Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
        return toi;
    }, [](const float left, const float right) { return min_scalar(left, right); }, host_accd::line_search_max_t);

    min_toi /= host_accd::line_search_max_t;
    if (min_toi < 1e-5)
    {
        luisa::log_error("toi is too small : {}", min_toi);
    }
    luisa::log_info("toi = {}", min_toi);
    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);
}


}
