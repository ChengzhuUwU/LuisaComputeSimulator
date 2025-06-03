#include "CollisionDetector/narrow_phase.h"
#include "CollisionDetector/accd.hpp"
#include "Utils/cpu_parallel.h"
#include <Eigen/Dense>
#include <iostream>
#include "Utils/reduce_helper.h"

namespace lcsv 
{

using EigenFloat3 = Eigen::Vector<float, 3>;
using EigenFloat4x3 = Eigen::Matrix<float, 3, 4, Eigen::ColMajor>;
static inline auto float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
static inline auto eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };
static inline auto make_eigen4x3(const EigenFloat3& c0, const EigenFloat3& c1, const EigenFloat3& c2, const EigenFloat3& c3) 
{ 
    EigenFloat4x3 mat; mat << c0, c1, c2, c3; return mat;
}

void NarrowPhasesDetector::unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    using namespace luisa::compute;

    // VF CCD Test
    {
        const float desire_toi = 0.6930697;
        luisa::log_info("VF Test, desire for toi {}", desire_toi);

        const uint vid = 1;
        const uint fid = 2;
        const uint3 face = uint3(4, 7, 5);
        float3 case_t0_p  = makeFloat3(0.48159984, -0.26639974, -0.48159984);
        float3 case_t1_p  = makeFloat3(0.47421163, -0.3129394, -0.47421163);
        float3 case_t0_f0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_t0_f1 = makeFloat3(0.6, -0.3, 0.5);
        float3 case_t0_f2 = makeFloat3(0.6, -0.3, -0.5);
        float3 case_t1_f0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_t1_f1 = makeFloat3(0.6, -0.3, 0.5);
        float3 case_t1_f2 = makeFloat3(0.6, -0.3, -0.5);

        {
            const auto t0_p  = float3_to_eigen3(case_t0_p ); 
            const auto t1_p  = float3_to_eigen3(case_t1_p );
            const auto t0_f0 = float3_to_eigen3(case_t0_f0); 
            const auto t0_f1 = float3_to_eigen3(case_t0_f1);
            const auto t0_f2 = float3_to_eigen3(case_t0_f2); 
            const auto t1_f0 = float3_to_eigen3(case_t1_f0);
            const auto t1_f1 = float3_to_eigen3(case_t1_f1); 
            const auto t1_f2 = float3_to_eigen3(case_t1_f2);

            float toi = host_accd::point_triangle_ccd(t0_p,  t1_p,
                                      t0_f0, t0_f1,
                                      t0_f2, t1_f0,
                                      t1_f1, t1_f2,
                                      1e-3);
            luisa::log_info("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", 0, toi, vid, fid, face);
        }
        {
            auto fn_test_ccd_vf = device.compile<1>([&](Float thickness)
            {
                Uint pair_idx = 0;
                Float toi = accd::line_search_max_t;
                
                {
                    Float3 t0_p  = case_t0_p ;
                    Float3 t1_p  = case_t1_p ;
                    Float3 t0_f0 = case_t0_f0;
                    Float3 t0_f1 = case_t0_f1;
                    Float3 t0_f2 = case_t0_f2;
                    Float3 t1_f0 = case_t1_f0;
                    Float3 t1_f1 = case_t1_f1;
                    Float3 t1_f2 = case_t1_f2;
        
                   Float toi = accd::point_triangle_ccd(t0_p,  t1_p,
                                      t0_f0, t0_f1,
                                      t0_f2, t1_f0,
                                      t1_f1, t1_f2,
                                      thickness);  
                    device_log("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", pair_idx, toi, vid, fid, face);
                };

                // toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);
            });
        
            stream << fn_test_ccd_vf(1e-3).dispatch(1) << synchronize();
        }
    }

    // EE CCD Test
    {
        float desire_toi = 0.91535777;
        luisa::log_info("EE Test, desire for toi {}", desire_toi);

        const uint left = 4;
        const uint right = 6;
        const uint2 left_edge = uint2(2, 3);
        const uint2 right_edge = uint2(4, 6);
        
        float3 case_ea_t0_p0 = makeFloat3(-0.499492, -0.279657, 0.460444);
        float3 case_ea_t0_p1 = makeFloat3(0.499997, -0.248673, 0.468853);
        float3 case_eb_t0_p0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_eb_t0_p1 = makeFloat3(-0.4, -0.3, 0.5);
        float3 case_ea_t1_p0 = makeFloat3(-0.49939114, -0.30410385, 0.4529846);
        float3 case_ea_t1_p1 = makeFloat3(0.4999971, -0.27044764, 0.4630015);
        float3 case_eb_t1_p0 = makeFloat3(-0.4, -0.3, -0.5);
        float3 case_eb_t1_p1 = makeFloat3(-0.4, -0.3, 0.5);
    
        {
            const auto ea00 = float3_to_eigen3(case_ea_t0_p0); 
            const auto ea01 = float3_to_eigen3(case_ea_t0_p1);
            const auto eb00 = float3_to_eigen3(case_eb_t0_p0); 
            const auto eb01 = float3_to_eigen3(case_eb_t0_p1);
            const auto ea10 = float3_to_eigen3(case_ea_t1_p0); 
            const auto ea11 = float3_to_eigen3(case_ea_t1_p1);
            const auto eb10 = float3_to_eigen3(case_eb_t1_p0); 
            const auto eb11 = float3_to_eigen3(case_eb_t1_p1);
    
            float toi = host_accd::edge_edge_ccd(
                ea00, 
                ea01, 
                eb00, 
                eb01, 
                ea10, 
                ea11, 
                eb10, 
                eb11, 1e-3);
            luisa::log_info("BroadPhase Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", 0, toi, left, left_edge, right, right_edge);
        }
    
        auto fn_test_ccd_ee = device.compile<1>([&](Float thickness)
        {
            Uint pair_idx = 0;
            Float toi = accd::line_search_max_t;
            
            {
                Float3 ea_t0_p0 = case_ea_t0_p0;
                Float3 ea_t0_p1 = case_ea_t0_p1;
                Float3 eb_t0_p0 = case_eb_t0_p0;
                Float3 eb_t0_p1 = case_eb_t0_p1;
                Float3 ea_t1_p0 = case_ea_t1_p0;
                Float3 ea_t1_p1 = case_ea_t1_p1;
                Float3 eb_t1_p0 = case_eb_t1_p0;
                Float3 eb_t1_p1 = case_eb_t1_p1;
    
                toi = accd::edge_edge_ccd(
                    ea_t0_p0, 
                    ea_t0_p1, 
                    eb_t0_p0, 
                    eb_t0_p1, 
                    ea_t1_p0, 
                    ea_t1_p1, 
                    eb_t1_p0, 
                    eb_t1_p1, 
                    thickness);  
            };
            
            // $if (toi != host_accd::line_search_max_t) 
            {
                device_log("BroadPhase Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
            };
    
            // toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);
        });
    
        stream << fn_test_ccd_ee(1e-3).dispatch(1) << synchronize();
    }
}

void NarrowPhasesDetector::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    fn_reset_toi = device.compile<1>([](Var<BufferView<float>> sa_toi)
    {
        sa_toi->write(dispatch_x(), accd::line_search_max_t);
    });
    
    fn_narrow_phase_vf_ccd_query = device.compile<1>(
    [
        broadphase_count = ccd_data->broad_phase_collision_count.view(1, 1),
        broadphase_list = ccd_data->broad_phase_list_vf.view()
    ](
        Var<BufferView<float>> sa_toi,
        Var<BufferView<float3>> sa_x_begin_left, 
        Var<BufferView<float3>> sa_x_begin_right, 
        Var<BufferView<float3>> sa_x_end_left,
        Var<BufferView<float3>> sa_x_end_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);

        const Uint3 face = sa_faces_right.read(fid);

        Float toi = accd::line_search_max_t;
        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {
            toi = accd::line_search_max_t;
        }
        $else
        {
            Float3 t0_p =  sa_x_begin_left->read(vid);
            Float3 t1_p =  sa_x_end_left->read(vid);
            Float3 t0_f0 = sa_x_begin_right->read(face[0]);
            Float3 t0_f1 = sa_x_begin_right->read(face[1]);
            Float3 t0_f2 = sa_x_begin_right->read(face[2]);
            Float3 t1_f0 = sa_x_end_right->read(face[0]);
            Float3 t1_f1 = sa_x_end_right->read(face[1]);
            Float3 t1_f2 = sa_x_end_right->read(face[2]);
    
            toi = accd::point_triangle_ccd(
                t0_p,  
                t1_p,              
                t0_f0, 
                t0_f1,                    
                t0_f2, 
                t1_f0,                 
                t1_f1, 
                t1_f2,              
                thickness);
            
            // BroadPhase Pair 12 : toi = 0.693334, vid 1 & fid 2 (face uint3(4, 7, 5)), dist = 0.0011289602 -> 0.00016696547
            $if (toi != accd::line_search_max_t)
            {
                device_log("VF Pair {} : toi = {}, vid {} & fid {} (face {})", 
                    pair_idx, toi, vid, fid, face
                );
            };
        };

        toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

        $if (pair_idx % 256 == 0)
        {
            sa_toi.atomic(0).fetch_min(toi);
        };
    });

    fn_narrow_phase_ee_ccd_query = device.compile<1>(
    [
        broadphase_count = ccd_data->broad_phase_collision_count.view(1, 1),
        broadphase_list = ccd_data->broad_phase_list_ee.view()
    ](
        Var<BufferView<float>> sa_toi,
        Var<BufferView<float3>> sa_x_begin_a, 
        Var<BufferView<float3>> sa_x_begin_b, 
        Var<BufferView<float3>> sa_x_end_a,
        Var<BufferView<float3>> sa_x_end_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float thickness
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);

        Float toi = accd::line_search_max_t;
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
            toi = accd::line_search_max_t;
        }
        $else
        {
            Float3 ea_t0_p0 = (sa_x_begin_a->read(left_edge[0]));
            Float3 ea_t0_p1 = (sa_x_begin_a->read(left_edge[1]));
            Float3 eb_t0_p0 = (sa_x_begin_b->read(right_edge[0]));
            Float3 eb_t0_p1 = (sa_x_begin_b->read(right_edge[1]));
            Float3 ea_t1_p0 = (sa_x_end_a->read(left_edge[0]));
            Float3 ea_t1_p1 = (sa_x_end_a->read(left_edge[1]));
            Float3 eb_t1_p0 = (sa_x_end_b->read(right_edge[0]));
            Float3 eb_t1_p1 = (sa_x_end_b->read(right_edge[1]));
    
            toi = accd::edge_edge_ccd(
                ea_t0_p0, 
                ea_t0_p1, 
                eb_t0_p0, 
                eb_t0_p1, 
                ea_t1_p0, 
                ea_t1_p1, 
                eb_t1_p0, 
                eb_t1_p1, 
                thickness);
        };
        
        $if (toi != host_accd::line_search_max_t) 
        {
            device_log("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
        };

        toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

        $if (pair_idx % 256 == 0)
        {
            sa_toi.atomic(0).fetch_min(toi);
        };
    });
   

}


void NarrowPhasesDetector::narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
    const Buffer<float3>& sa_x_begin_left, 
    const Buffer<float3>& sa_x_begin_right, 
    const Buffer<float3>& sa_x_end_left,
    const Buffer<float3>& sa_x_end_right,
    const Buffer<uint3>& sa_faces_right,
    const float thickness)
{
    auto& sa_toi = ccd_data->toi_per_vert;
    auto broadphase_count = ccd_data->broad_phase_collision_count.view(0, 2);
    auto& host_toi = host_ccd_data->toi_per_vert;
    auto& host_count = host_ccd_data->broad_phase_collision_count;

    stream 
        << fn_reset_toi(sa_toi).dispatch(sa_toi.size())
        << broadphase_count.copy_to(host_count.data()) 
        // << sa_toi.copy_to(host_toi.data()) 
        << luisa::compute::synchronize();

    const uint num_vf_broadphase = host_count[0];
    const uint num_ee_broadphase = host_count[1];

    luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase); // TODO: Indirect Dispatch
    luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase); // TODO: Indirect Dispatch
    // luisa::log_info("curr toi = {} before VF", host_toi[0]);

    // std::vector<float3> host_x_begin(sa_x_begin_left.size());
    // std::vector<float3> host_x_end(sa_x_end_left.size());
    // std::vector<uint3> host_faces(sa_faces_right.size());
    // stream 
    //         << sa_x_begin_left.copy_to(host_x_begin.data())
    //         << sa_x_end_left.copy_to(host_x_end.data())
    //         << sa_faces_right.copy_to(host_faces.data())
    //         << luisa::compute::synchronize();

    // host_narrow_phase_ccd_query_from_vf_pair(stream, 
    //         host_x_begin, 
    //         host_x_begin, 
    //         host_x_end, 
    //         host_x_end, 
    //         host_faces, 
    //         1e-3);

    stream << fn_narrow_phase_vf_ccd_query(
        sa_toi,
        sa_x_begin_left,
        sa_x_begin_right, // sa_x_begin_right
        sa_x_end_left,
        sa_x_end_right, // sa_x_end_right
        sa_faces_right, thickness
    ).dispatch(num_vf_broadphase);

    
}

void NarrowPhasesDetector::narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
    const Buffer<float3>& sa_x_begin_a, 
    const Buffer<float3>& sa_x_begin_b, 
    const Buffer<float3>& sa_x_end_a,
    const Buffer<float3>& sa_x_end_b,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float thickness)
{
    auto broadphase_count = ccd_data->broad_phase_collision_count.view(0, 2);
    auto& sa_toi = ccd_data->toi_per_vert;
    auto& host_count = host_ccd_data->broad_phase_collision_count;
    auto& host_toi = host_ccd_data->toi_per_vert;
    
    stream 
        // << sa_toi.copy_to(host_toi.data()) 
        << broadphase_count.copy_to(host_count.data()) 
        << luisa::compute::synchronize();

    const uint num_vf_broadphase = host_count[0]; 
    const uint num_ee_broadphase = host_count[1];

    // luisa::log_info("curr toi = {} from VF", host_toi[0]);

    // std::vector<float3> host_x_begin(sa_x_begin_a.size());
    // std::vector<float3> host_x_end(sa_x_end_a.size());
    // std::vector<uint2> host_edges(sa_edges_left.size());
    // stream 
    //         << sa_x_begin_a.copy_to(host_x_begin.data())
    //         << sa_x_end_a.copy_to(host_x_end.data())
    //         << sa_edges_left.copy_to(host_edges.data())
    //         << luisa::compute::synchronize();

    // host_narrow_phase_ccd_query_from_ee_pair(stream, 
    //         host_x_begin, 
    //         host_x_begin, 
    //         host_x_end, 
    //         host_x_end, 
    //         host_edges, 
    //         host_edges, 
    //         1e-3);

    stream << fn_narrow_phase_ee_ccd_query(sa_toi,
        sa_x_begin_a,
        sa_x_begin_b,
        sa_x_end_a,
        sa_x_end_b,
        sa_edges_left,
        sa_edges_left, thickness
    ).dispatch(num_ee_broadphase) << sa_toi.view(0, 1).copy_to(host_toi.data()) << luisa::compute::synchronize();

    host_toi[0] /= host_accd::line_search_max_t;
    if (host_toi[0] < 1e-5)
    {
        luisa::log_error("  small toi : {}", host_toi[0]);
    }

    luisa::log_info("toi = {}", host_toi[0]);
}


void NarrowPhasesDetector::host_narrow_phase_ccd_query_from_vf_pair(Stream& stream, 
    const std::vector<float3>& sa_x_begin_left, 
    const std::vector<float3>& sa_x_begin_right, 
    const std::vector<float3>& sa_x_end_left,
    const std::vector<float3>& sa_x_end_right,
    const std::vector<uint3>& sa_faces_right,
    const float thickness)
{
    auto& sa_toi = host_ccd_data->toi_per_vert;
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

        if (toi != host_accd::line_search_max_t) 
        {
            // luisa::log_info("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", 
            //     pair_idx, toi, left, right, right_face,
            // );
            luisa::log_info("VF Pair {} : toi = {}, vid {} & fid {} (face {}), dist = {} -> {}", 
                pair_idx, toi, left, right, right_face, 
                host_distance::point_triangle_distance_squared_unclassified(t0_p, t0_f0, t0_f1, t0_f2),
                host_distance::point_triangle_distance_squared_unclassified(t1_p, t1_f0, t1_f1, t1_f2)
            );
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_left[left]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_left[left]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[2]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[2]]);
        }
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

void NarrowPhasesDetector::host_narrow_phase_ccd_query_from_ee_pair(Stream& stream, 
    const std::vector<float3>& sa_x_begin_a, 
    const std::vector<float3>& sa_x_begin_b, 
    const std::vector<float3>& sa_x_end_a,
    const std::vector<float3>& sa_x_end_b,
    const std::vector<uint2>& sa_edges_left,
    const std::vector<uint2>& sa_edges_right,
    const float thickness)
{
    auto& sa_toi = host_ccd_data->toi_per_vert;
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

        if (toi != host_accd::line_search_max_t) 
        {
            luisa::log_info("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_a[left_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_a[left_edge[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_b[right_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_begin_b[right_edge[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_a[left_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_a[left_edge[1]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_b[right_edge[0]]);
            // luisa::log_info("             {} : positions : {}", pair_idx, sa_x_end_b[right_edge[1]]);
        }
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
