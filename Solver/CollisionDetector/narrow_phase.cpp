#include "CollisionDetector/narrow_phase.h"
#include "CollisionDetector/accd.hpp"
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/libuipc/codim_ipc_simplex_normal_contact_function.h"
#include "CollisionDetector/libuipc/distance/distance_flagged.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include <Eigen/Dense>
#include <iostream>
#include "Utils/reduce_helper.h"

namespace lcsv // 
{

using EigenFloat3x3 = Eigen::Matrix<float, 3, 3>;
using EigenFloat6x6 = Eigen::Matrix<float, 6, 6>;
using EigenFloat9x9 = Eigen::Matrix<float, 9, 9>;
using EigenFloat12x12 = Eigen::Matrix<float, 12, 12>;
using EigenFloat3   = Eigen::Matrix<float, 3, 1>;
using EigenFloat4   = Eigen::Matrix<float, 4, 1>;


static inline auto float3_to_eigen3(const float3& input) { EigenFloat3 vec; vec << input[0], input[1], input[2]; return vec; };
static inline auto eigen3_to_float3(const EigenFloat3& input) { return luisa::make_float3(input(0, 0), input(1, 0), input(2, 0)); };
static inline auto eigen4_to_float4(const EigenFloat4& input) { return luisa::make_float4(input(0, 0), input(1, 0), input(2, 0), input(3, 0)); };

static inline EigenFloat3x3 float3x3_to_eigen3x3(const float3x3& input)
{
    EigenFloat3x3 mat; mat << 
        input[0][0], input[1][0], input[2][0], 
        input[0][1], input[1][1], input[2][1], 
        input[0][2], input[1][2], input[2][2]; 
    return mat;
};
static inline float3x3 eigen3x3_to_float3x3(const EigenFloat3x3& input)
{
    return luisa::make_float3x3(
        input(0, 0), input(1, 0), input(2, 0), 
        input(0, 1), input(1, 1), input(2, 1), 
        input(0, 2), input(1, 2), input(2, 2));
};
static inline EigenFloat6x6 float6x6_to_eigen6x6(const float6x6& input)
{
    EigenFloat6x6 output;
    for (uint i = 0; i < 2; ++i) 
    {
        for (uint j = 0; j < 2; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
static inline float6x6 eigen6x6_to_float6x6(const EigenFloat6x6& input)
{
    float6x6 output;
    for (uint i = 0; i < 2; ++i) 
    {
        for (uint j = 0; j < 2; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
static inline EigenFloat9x9 float9x9_to_eigen9x9(const float9x9& input)
{
    EigenFloat9x9 output;
    for (uint i = 0; i < 3; ++i) 
    {
        for (uint j = 0; j < 3; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
static inline float9x9 eigen9x9_to_float9x9(const EigenFloat9x9& input)
{
    float9x9 output;
    for (uint i = 0; i < 3; ++i) 
    {
        for (uint j = 0; j < 3; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};
static inline EigenFloat12x12 float12x12_to_eigen12x12(const float12x12 input)
{
    EigenFloat12x12 output;
    for (uint i = 0; i < 4; ++i) 
    {
        for (uint j = 0; j < 4; ++j) 
        {
            output.block<3, 3>(i * 3, j * 3) = float3x3_to_eigen3x3(input.mat[i][j]);
        }
    }
    return output;
};
static inline float12x12 eigen12x12_to_float12x12(const EigenFloat12x12& input)
{
    float12x12 output;
    for (uint i = 0; i < 4; ++i) 
    {
        for (uint j = 0; j < 4; ++j) 
        {
            output.mat[i][j] = eigen3x3_to_float3x3(input.block<3, 3>(i * 3, j * 3));
        }
    }
    return output;
};

void NarrowPhasesDetector::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();
    
    compile_ccd(device);
    compile_dcd(device);
    compile_energy(device);
    compile_assemble(device);
}

void NarrowPhasesDetector::reset_toi(Stream& stream)
{
    auto& sa_toi = collision_data->toi_per_vert;
    stream << fn_reset_toi(sa_toi).dispatch(sa_toi.size());
}
void NarrowPhasesDetector::reset_broadphase_count(Stream& stream)
{
    stream << fn_reset_uint(collision_data->broad_phase_collision_count).dispatch(collision_data->broad_phase_collision_count.size());
}
void NarrowPhasesDetector::reset_narrowphase_count(Stream& stream)
{
    stream << fn_reset_uint(collision_data->narrow_phase_collision_count).dispatch(collision_data->narrow_phase_collision_count.size());
}
void NarrowPhasesDetector::reset_energy(Stream& stream)
{
    auto& contact_energy = collision_data->contact_energy;
    stream << fn_reset_energy(contact_energy).dispatch(contact_energy.size());
}
float NarrowPhasesDetector::download_energy(Stream& stream, const float kappa)
{
    auto& contact_energy = collision_data->contact_energy;
    auto& host_contact_energy = host_collision_data->contact_energy;
    stream 
        << contact_energy.copy_to(host_contact_energy.data())
        << luisa::compute::synchronize();
    return std::accumulate(host_contact_energy.begin(), host_contact_energy.end(), 0.0f);
    // return kappa * (host_contact_energy[2] + host_contact_energy[3]);
}
void NarrowPhasesDetector::host_reset_toi(Stream& stream)
{
    auto& sa_toi = host_collision_data->toi_per_vert;
    CpuParallel::parallel_set(sa_toi, 0.0f);
}
void NarrowPhasesDetector::download_broadphase_collision_count(Stream& stream)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    stream 
        << broadphase_count.copy_to(host_count.data()) 
        << luisa::compute::synchronize();

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];
    if (num_vf_broadphase > collision_data->broad_phase_list_vf.size() / 2) { luisa::log_error("BroadPhase VF outof range"); }
    if (num_ee_broadphase > collision_data->broad_phase_list_ee.size() / 2) { luisa::log_error("BroadPhase EE outof range"); }

    // luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase); // TODO: Indirect Dispatch
    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase); // TODO: Indirect Dispatch
}
void NarrowPhasesDetector::download_narrowphase_collision_count(Stream& stream)
{
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    stream 
        << narrowphase_count.copy_to(host_count.data()) 
        << luisa::compute::synchronize();
}
void NarrowPhasesDetector::download_narrowphase_list(Stream& stream)
{
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    // luisa::log_info("       num_vv = {}, num_ve = {}, num_vf = {}, num_ee = {}", num_vv, num_ve, num_vf, num_ee); 

    
    stream << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_to(host_collision_data->narrow_phase_list_vv.data()) << luisa::compute::synchronize(); 
    stream << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_to(host_collision_data->narrow_phase_list_ve.data()) << luisa::compute::synchronize(); 
    stream << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_to(host_collision_data->narrow_phase_list_vf.data()) << luisa::compute::synchronize(); 
    stream << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_to(host_collision_data->narrow_phase_list_ee.data()) << luisa::compute::synchronize(); 

    // Why this can not run ???
    // stream 
    //         << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_to(host_collision_data->narrow_phase_list_vv.data()) 
    //         << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_to(host_collision_data->narrow_phase_list_ve.data()) 
    //         << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_to(host_collision_data->narrow_phase_list_vf.data()) 
    //         << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_to(host_collision_data->narrow_phase_list_ee.data()) 
    //         << luisa::compute::synchronize();

    // luisa::log_info("Complete Download");
}
void NarrowPhasesDetector::upload_spd_narrowphase_list(Stream& stream)
{
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    // luisa::log_info("       num_vv = {}, num_ve = {}, num_vf = {}, num_ee = {}", num_vv, num_ve, num_vf, num_ee); 

    stream << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_from(host_collision_data->narrow_phase_list_vv.data()) << luisa::compute::synchronize(); 
    stream << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_from(host_collision_data->narrow_phase_list_ve.data()) << luisa::compute::synchronize(); 
    stream << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_from(host_collision_data->narrow_phase_list_vf.data()) << luisa::compute::synchronize(); 
    stream << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_from(host_collision_data->narrow_phase_list_ee.data()) << luisa::compute::synchronize(); 

    // stream 
    //     << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_from(host_collision_data->narrow_phase_list_vv.data()) 
    //     << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_from(host_collision_data->narrow_phase_list_ve.data()) 
    //     << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_from(host_collision_data->narrow_phase_list_vf.data()) 
    //     << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_from(host_collision_data->narrow_phase_list_ee.data()) 
    // ;

    // luisa::log_info("Complete Download");
}
float NarrowPhasesDetector::get_global_toi(Stream& stream)
{
    stream << luisa::compute::synchronize();

    auto& host_toi = host_collision_data->toi_per_vert[0];
    // if (host_toi != host_accd::line_search_max_t) luisa::log_info("             CCD linesearch toi = {}", host_toi);
    host_toi /= host_accd::line_search_max_t;
    if (host_toi < 1e-5)
    {
        luisa::log_error("  small toi : {}", host_toi);
    }
    return host_toi;
    // 
}

} // namespace lcsv 

namespace lcsv // CCD
{

void NarrowPhasesDetector::compile_ccd(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    fn_reset_toi = device.compile<1>([](Var<BufferView<float>> sa_toi)
    {
        sa_toi->write(dispatch_x(), accd::line_search_max_t);
    });
    fn_reset_uint = device.compile<1>([](Var<BufferView<uint>> sa_toi)
    {
        sa_toi->write(dispatch_x(), 0u);
    });
    fn_reset_float = device.compile<1>([](Var<BufferView<float>> sa_toi)
    {
        sa_toi->write(dispatch_x(), 0.0f);
    });
    fn_reset_energy = device.compile<1>([](Var<BufferView<float>> sa_energy)
    {
        sa_energy->write(dispatch_x(), 0.0f);
    });

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();
    
    fn_narrow_phase_vf_ccd_query = device.compile<1>(
    [
        sa_toi = collision_data->toi_per_vert.view(),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view()
    ](
        Var<BufferView<float3>> sa_x_begin_left, 
        Var<BufferView<float3>> sa_x_begin_right, 
        Var<BufferView<float3>> sa_x_end_left,
        Var<BufferView<float3>> sa_x_end_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float d_hat, // Not relavent to d_hat
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
            
            // $if (toi != accd::line_search_max_t)
            // {
            //     device_log("VF Pair {} : toi = {}, vid {} & fid {} (face {})", 
            //         pair_idx, toi, vid, fid, face
            //     );
            // };
        };

        toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

        $if (pair_idx % 256 == 0)
        {
            sa_toi->atomic(0).fetch_min(toi);
        };
    });

    fn_narrow_phase_ee_ccd_query = device.compile<1>(
    [
        sa_toi = collision_data->toi_per_vert.view(),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_begin_a, 
        Var<BufferView<float3>> sa_x_begin_b, 
        Var<BufferView<float3>> sa_x_end_a,
        Var<BufferView<float3>> sa_x_end_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat, // Not relavent to d_hat
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
        
        // $if (toi != host_accd::line_search_max_t) 
        // {
        //     device_log("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
        // };

        toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

        $if (pair_idx % 256 == 0)
        {
            sa_toi->atomic(0).fetch_min(toi);
        };
    });
}

// Device CCD
void NarrowPhasesDetector::vf_ccd_query(Stream& stream, 
    const Buffer<float3>& sa_x_begin_left, 
    const Buffer<float3>& sa_x_begin_right, 
    const Buffer<float3>& sa_x_end_left,
    const Buffer<float3>& sa_x_end_right,
    const Buffer<uint3>& sa_faces_right,
    const float d_hat,
    const float thickness)
{
    auto& sa_toi = collision_data->toi_per_vert;
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_toi = host_collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

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

    if (num_vf_broadphase != 0) 
    {
        stream << fn_narrow_phase_vf_ccd_query(
            sa_x_begin_left,
            sa_x_begin_right, // sa_x_begin_right
            sa_x_end_left,
            sa_x_end_right, // sa_x_end_right
            sa_faces_right, d_hat, thickness
        ).dispatch(num_vf_broadphase) ;
    
    }
    stream << sa_toi.view(0, 1).copy_to(host_toi.data())
    ;

}

void NarrowPhasesDetector::ee_ccd_query(Stream& stream, 
    const Buffer<float3>& sa_x_begin_a, 
    const Buffer<float3>& sa_x_begin_b, 
    const Buffer<float3>& sa_x_end_a,
    const Buffer<float3>& sa_x_end_b,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float d_hat,
    const float thickness)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& sa_toi = collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;
    auto& host_toi = host_collision_data->toi_per_vert;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()]; 
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

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

    if (num_ee_broadphase != 0)
    {
        stream << fn_narrow_phase_ee_ccd_query(
            sa_x_begin_a,
            sa_x_begin_b,
            sa_x_end_a,
            sa_x_end_b,
            sa_edges_left,
            sa_edges_left, d_hat, thickness
        ).dispatch(num_ee_broadphase);
    }
    stream << sa_toi.view(0, 1).copy_to(host_toi.data())
    ;
}

} // namespace lcsv 

namespace lcsv // DCD
{

constexpr float stiffness_repulsion = 1e2;

void NarrowPhasesDetector::compile_dcd(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    // Barrier Query
    fn_narrow_phase_vf_dcd_query_barrier = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view(),
        narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
        narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
        narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    ](
        Var<BufferView<float3>> sa_x_left, 
        Var<BufferView<float3>> sa_x_right,
        Var<BufferView<float3>> sa_rest_x_a, 
        Var<BufferView<float3>> sa_rest_x_b,
        Var<BufferView<uint3>> sa_faces_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);
        const Uint3 face = sa_faces_right.read(fid);

        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {

        }
        $else
        {
            Float3 p =  sa_x_left->read(vid);
            Float3 face_positions[3] = {
                sa_x_right->read(face[0]),
                sa_x_right->read(face[1]),
                sa_x_right->read(face[2]),
            };
            Float3& t0 = face_positions[0];
            Float3& t1 = face_positions[1];
            Float3& t2 = face_positions[2];

            Float3 bary = distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
            uint3 valid_indices = makeUint3(0, 1, 2);
            uint valid_count = distance::point_triangle_type(bary, valid_indices);
            
            Float3 x = bary[0] * (p - t0) +
                       bary[1] * (p - t1) +
                       bary[2] * (p - t2);
            Float d2 = length_squared_vec(x);
            $if (d2 < square_scalar(thickness + d_hat))
            {
                Float3 rest_p = sa_rest_x_a->read(vid);
                Float3 rest_t0 = sa_rest_x_b->read(face[0]);
                Float3 rest_t1 = sa_rest_x_b->read(face[1]);
                Float3 rest_t2 = sa_rest_x_b->read(face[2]);
                Float rest_d2 = distance::point_triangle_distance_squared_unclassified(
                    rest_p,
                    rest_t0,
                    rest_t1,
                    rest_t2
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float3 normal = x / d;
                    Float C = thickness + d_hat - d;
                    Float stiff_repulsion = C * stiffness_repulsion;
                    Float dBdD; Float ddBddD;
                    cipc::dKappaBarrierdD(dBdD, kappa, d2, d_hat, thickness);
                    cipc::ddKappaBarrierddD(ddBddD, kappa, d2, d_hat, thickness);
    
                    $if (valid_count == 3) // VF
                    {
                        Uint idx = narrowphase_count_vf->atomic(0).fetch_add(1u);
                        Var<CollisionPairVF> vf_pair;
                        vf_pair.indices = makeUint4(vid, face[0], face[1], face[2]);
                        vf_pair.vec1 = makeFloat4(normal.x, normal.y, normal.z, stiff_repulsion);
                        vf_pair.bary = bary;
                        {
                            Float12 G;
                            Float12 GradD;
                            DistanceGradient::point_triangle_distance2_gradient(p, t0, t1, t2, GradD); // GradiantD
                            mult_largevec_scalar(G, GradD, dBdD);                        
                            
                            $if (
                                luisa::compute::isnan(G.vec[0][0]) |
                                luisa::compute::isnan(G.vec[0][1]) |
                                luisa::compute::isnan(G.vec[0][2]) | 
                                luisa::compute::isnan(G.vec[1][0]) |
                                luisa::compute::isnan(G.vec[1][1]) |
                                luisa::compute::isnan(G.vec[1][2]) |
                                luisa::compute::isnan(G.vec[2][0]) |
                                luisa::compute::isnan(G.vec[2][1]) |
                                luisa::compute::isnan(G.vec[2][2]) |
                                luisa::compute::isnan(G.vec[3][0]) |
                                luisa::compute::isnan(G.vec[3][1]) |
                                luisa::compute::isnan(G.vec[3][2]) 
                            )
                            {
                                luisa::compute::device_log("VF gradient is NAN with : {} -> {}/{}/{} (d = {})", p, t0, t1, t2, d);
                            };

                            Float12x12 HessD;
                            DistanceGradient::point_triangle_distance2_hessian(p, t0, t1, t2, HessD); // HessianD
    
                            Float12x12 H = add_largemat(
                                outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                                mult_largemat_scalar(HessD, dBdD)
                            );
    
                            vf_pair.gradient[0] = G.vec[0];
                            vf_pair.gradient[1] = G.vec[1];
                            vf_pair.gradient[2] = G.vec[2];
                            vf_pair.gradient[3] = G.vec[3];
                            CollisionPair::write_upper_hessian(vf_pair.hessian, H);
                        }
                        narrowphase_list_vf->write(idx, vf_pair);
                    }
                    $elif (valid_count == 2) // VE
                    {
                        Uint idx = narrowphase_count_ve->atomic(0).fetch_add(1u);
                        Var<CollisionPairVE> ve_pair;
                        ve_pair.vid = vid;
                        ve_pair.edge = makeUint2(
                            face[valid_indices[0]], 
                            face[valid_indices[1]]
                        );
                        ve_pair.vec1 = makeFloat4(normal.x, normal.y, normal.z, stiff_repulsion);
                        ve_pair.bary = bary[valid_indices[0]];
                        
                        {
                            Float3& e0 = face_positions[valid_indices[0]];
                            Float3& e1 = face_positions[valid_indices[1]];
                            Float9 G;
                            Float9 GradD;
                            DistanceGradient::point_edge_distance2_gradient(p, e0, e1, GradD); // GradiantD
                            mult_largevec_scalar(G, GradD, dBdD);                        

                            $if (
                                luisa::compute::isnan(G.vec[0][0]) |
                                luisa::compute::isnan(G.vec[0][1]) |
                                luisa::compute::isnan(G.vec[0][2]) | 
                                luisa::compute::isnan(G.vec[1][0]) |
                                luisa::compute::isnan(G.vec[1][1]) |
                                luisa::compute::isnan(G.vec[1][2]) |
                                luisa::compute::isnan(G.vec[2][0]) |
                                luisa::compute::isnan(G.vec[2][1]) |
                                luisa::compute::isnan(G.vec[2][2]) 
                            )
                            {
                                luisa::compute::device_log("VE gradient is NAN with : {} -> {}/{} (d = {})", p, e0, e1, d);
                            };
    
                            Float9x9 HessD;
                            DistanceGradient::point_edge_distance2_hessian(p, e0, e1, HessD); // HessianD
    
                            Float9x9 H = add_largemat(
                                outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                                mult_largemat_scalar(HessD, dBdD)
                            );
    
                            ve_pair.gradient[0] = G.vec[0];
                            ve_pair.gradient[1] = G.vec[1];
                            ve_pair.gradient[2] = G.vec[2];
                            //  0  1  2  
                            //     3  4  
                            //        5  
                            //           
                            CollisionPair::write_upper_hessian(ve_pair.hessian, H);
                        }
                        narrowphase_list_ve->write(idx, ve_pair);
                    }
                    $else // VV // valid_count == 1
                    {
                        Uint idx = narrowphase_count_vv->atomic(0).fetch_add(1u);
                        Var<CollisionPairVV> vv_pair;
                        vv_pair.indices[0] = vid;
                        vv_pair.indices[1] = valid_indices[0];
                        vv_pair.vec1 = makeFloat4(normal.x, normal.y, normal.z, stiff_repulsion);
                        {
                            Float3& p0 = p;
                            Float3& p1 = face_positions[valid_indices[0]];
    
                            Float6 G;
                            Float6 GradD;
                            DistanceGradient::point_point_distance2_gradient(p0, p1, GradD); // GradiantD
                            mult_largevec_scalar(G, GradD, dBdD);            
                            
                            $if (
                                luisa::compute::isnan(GradD.vec[0][0]) |
                                luisa::compute::isnan(GradD.vec[0][1]) |
                                luisa::compute::isnan(GradD.vec[0][2]) | 
                                luisa::compute::isnan(GradD.vec[1][0]) |
                                luisa::compute::isnan(GradD.vec[1][1]) |
                                luisa::compute::isnan(GradD.vec[1][2]) 
                            )
                            {
                                luisa::compute::device_log("VV gradient is NAN with : {} -> {} (d = {})", p0, p1, d);
                            };
    
                            Float6x6 HessD;
                            DistanceGradient::point_point_distance2_hessian(p0, p1, HessD); // HessianD
    
                            Float6x6 H = add_largemat(
                                outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                                mult_largemat_scalar(HessD, dBdD)
                            );
    
                            //  0  1  
                            //     2  
                            vv_pair.gradient[0] = G.vec[0];
                            vv_pair.gradient[1] = G.vec[1];
                            CollisionPair::write_upper_hessian(vv_pair.hessian, H);
                        }
                        narrowphase_list_vv->write(idx, vv_pair);
                    };
                };
            };
        };
    });

    fn_narrow_phase_ee_dcd_query_barrier = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view(),
        narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_a, 
        Var<BufferView<float3>> sa_x_b,
        Var<BufferView<float3>> sa_rest_x_a, 
        Var<BufferView<float3>> sa_rest_x_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
        }
        $else
        {
            Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
            Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
            Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
            Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
    
            Float4 bary = distance::edge_edge_distance_coeff_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
            // Bool is_ee = all_vec(bary != 0.0f);
            luisa::uint2 valid_indices1; luisa::uint2 valid_indices2;
            uint2 valid_count = distance::edge_edge_type(bary, valid_indices1, valid_indices2);
            Bool is_ee = valid_count[0] == 2 && valid_count[1] == 2;

            Float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
            Float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
            Float3 x = x0 - x1;
            Float d2 = length_squared_vec(x);

            $if (d2 < square_scalar(d_hat + thickness) & is_ee)
            {
                Float3 rest_ea_p0 = (sa_rest_x_a->read(left_edge[0]));
                Float3 rest_ea_p1 = (sa_rest_x_a->read(left_edge[1]));
                Float3 rest_eb_p0 = (sa_rest_x_b->read(right_edge[0]));
                Float3 rest_eb_p1 = (sa_rest_x_b->read(right_edge[1]));
                Float rest_d2 = distance::edge_edge_distance_squared_unclassified(
                    rest_ea_p0,
                    rest_ea_p1,
                    rest_eb_p0,
                    rest_eb_p1
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float stiff = thickness + d_hat - d;
                    Float3 normal = x / d;
    
                    Float dBdD; Float ddBddD;
                    cipc::dKappaBarrierdD(dBdD, kappa, d2, d_hat, thickness);
                    cipc::ddKappaBarrierddD(ddBddD, kappa, d2, d_hat, thickness);
                    
                    Uint idx = narrowphase_count_ee->atomic(0).fetch_add(1u);
                    Var<CollisionPairEE> ee_pair;
                    ee_pair.indices = makeUint4(left_edge[0], left_edge[1], right_edge[0], right_edge[1]);
                    ee_pair.vec1 = makeFloat4(normal.x, normal.y, normal.z, stiff);
                    ee_pair.bary = bary;
                    {
                        Float12 GradD;
                        Float12 G; 
                        DistanceGradient::edge_edge_distance2_gradient(ea_p0, ea_p1, eb_p0, eb_p1, GradD); // GradiantD
                        mult_largevec_scalar(G, GradD, dBdD);                        
    
                        $if (
                                luisa::compute::isnan(G.vec[0][0]) |
                                luisa::compute::isnan(G.vec[0][1]) |
                                luisa::compute::isnan(G.vec[0][2]) | 
                                luisa::compute::isnan(G.vec[1][0]) |
                                luisa::compute::isnan(G.vec[1][1]) |
                                luisa::compute::isnan(G.vec[1][2]) |
                                luisa::compute::isnan(G.vec[2][0]) |
                                luisa::compute::isnan(G.vec[2][1]) |
                                luisa::compute::isnan(G.vec[2][2]) |
                                luisa::compute::isnan(G.vec[3][0]) |
                                luisa::compute::isnan(G.vec[3][1]) |
                                luisa::compute::isnan(G.vec[3][2]) 
                            )
                            {
                                luisa::compute::device_log("EE (bary = {}) gradient is NAN with : {}{} -> {}/{} (d = {})", bary, ea_p0, ea_p1, eb_p0, eb_p1, d);
                            };

                        Float12x12 H;
                        DistanceGradient::edge_edge_distance2_hessian(ea_p0, ea_p1, eb_p0, eb_p1, H); // HessianD
                        H = add_largemat(
                            outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                            mult_largemat_scalar(H, dBdD)
                        );
    
                        // luisa::compute::device_log("Detect EE pair on GPU {}", d);
                        // print_largevec(G);
                        // print_largemat(H);
    
                        ee_pair.gradient[0] = G.vec[0];
                        ee_pair.gradient[1] = G.vec[1];
                        ee_pair.gradient[2] = G.vec[2];
                        ee_pair.gradient[3] = G.vec[3];
                        CollisionPair::write_upper_hessian(ee_pair.hessian, H);
                    }
                    narrowphase_list_ee->write(idx, ee_pair);
                };
            };
            // Corner case (VV, VE) will only be considered in VF detection
        };
    });

    // Proximity Query
    fn_narrow_phase_vf_dcd_query_repulsion = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view(),
        narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
        narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
        narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    ](
        Var<BufferView<float3>> sa_x_left, 
        Var<BufferView<float3>> sa_x_right,
        Var<BufferView<float3>> sa_rest_x_a, 
        Var<BufferView<float3>> sa_rest_x_b,
        Var<BufferView<uint3>> sa_faces_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);
        const Uint3 face = sa_faces_right.read(fid);

        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {

        }
        $else
        {
            Float3 p =  sa_x_left->read(vid);
            Float3 face_positions[3] = {
                sa_x_right->read(face[0]),
                sa_x_right->read(face[1]),
                sa_x_right->read(face[2]),
            };
            Float3& t0 = face_positions[0];
            Float3& t1 = face_positions[1];
            Float3& t2 = face_positions[2];

            Float3 bary = distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
            
            Float3 x = bary[0] * (p - t0) +
                       bary[1] * (p - t1) +
                       bary[2] * (p - t2);
            Float d2 = length_squared_vec(x);
            // luisa::compute::device_log("VF pair {}-{} : d = {}", vid, face, sqrt_scalar(d2));
            $if (
                d2 < square_scalar(thickness + d_hat) 
                // & d2 > 1e-8f
            )
            {
                Float3 rest_p = sa_rest_x_a->read(vid);
                Float3 rest_t0 = sa_rest_x_b->read(face[0]);
                Float3 rest_t1 = sa_rest_x_b->read(face[1]);
                Float3 rest_t2 = sa_rest_x_b->read(face[2]);
                Float rest_d2 = distance::point_triangle_distance_squared_unclassified(
                    rest_p,
                    rest_t0,
                    rest_t1,
                    rest_t2
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float C = thickness + d_hat - d;
                    Float stiff = stiffness_repulsion * C;
                    $if (d < 5e-3f) { stiff = 1e3f * C; };
                    $if (d < 2e-3f) { stiff = 1e5f * C; };
                    $if (d < 1e-3f) { stiff = 1e7f * C; };
                    Float3 normal = x / d;
                    {
                        Uint idx = narrowphase_count_vf->atomic(0).fetch_add(1u);
                        Var<CollisionPairVF> vf_pair;
                        vf_pair.indices = makeUint4(vid, face[0], face[1], face[2]);
                        vf_pair.vec1 = makeFloat4(normal.x, normal.y, normal.z, stiff);
                        vf_pair.bary = bary;
                        {
                            Float4 weight = makeFloat4(1.0f, -bary[0], -bary[1], -bary[2]);
                            Float12 G;
                            
                            for (uint j = 0; j < 4; j++)
                            {
                                G.vec[j] = stiff * weight[j] * normal; // Gradient is negative of force
                            }

                            // G.vec[0] = stiff * weight[0] * normal;
                            // G.vec[1] = stiff * weight[1] * normal;
                            // G.vec[2] = stiff * weight[2] * normal;
                            // G.vec[3] = stiff * weight[3] * normal;
    
                            Float12x12 H;
                            Float3x3 xxT = stiff * outer_product(normal, normal);
                            for (uint j = 0; j < 4; j++)
                            {
                                for (uint jj = 0; jj < 4; jj++)
                                {
                                    H.mat[j][jj] = weight[j] * weight[jj] * xxT;
                                }
                            }
                            vf_pair.gradient[0] = G.vec[0];
                            vf_pair.gradient[1] = G.vec[1];
                            vf_pair.gradient[2] = G.vec[2];
                            vf_pair.gradient[3] = G.vec[3];
                            CollisionPair::write_upper_hessian(vf_pair.hessian, H);
                            // luisa::compute::device_log("VF pair {} ({}) with C = {}, G = {} - {} - {} - {}", idx, vf_pair.indices, C, G.vec[0], G.vec[1], G.vec[2], G.vec[3]);
                        }
                        narrowphase_list_vf->write(idx, vf_pair);
                    }
                };
            };
        };
    });

    fn_narrow_phase_ee_dcd_query_repulsion = device.compile<1>(
    [
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view(),
        narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_a, 
        Var<BufferView<float3>> sa_x_b,
        Var<BufferView<float3>> sa_rest_x_a, 
        Var<BufferView<float3>> sa_rest_x_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
        }
        $else
        {
            Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
            Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
            Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
            Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
    
            Float4 bary = distance::edge_edge_distance_coeff_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
            Bool is_ee = all_vec(bary != 0.0f);

            Float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
            Float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
            Float3 x = x0 - x1;
            Float d2 = length_squared_vec(x);
            // luisa::compute::device_log("EE pair {}-{} : d = {}", left_edge, right_edge, sqrt_scalar(d2));

            $if (
                d2 < square_scalar(thickness + d_hat)
                //  & d2 > 1e-8f
            )
            {
                Float3 rest_ea_p0 = (sa_rest_x_a->read(left_edge[0]));
                Float3 rest_ea_p1 = (sa_rest_x_a->read(left_edge[1]));
                Float3 rest_eb_p0 = (sa_rest_x_b->read(right_edge[0]));
                Float3 rest_eb_p1 = (sa_rest_x_b->read(right_edge[1]));
                Float rest_d2 = distance::edge_edge_distance_squared_unclassified(
                    rest_ea_p0,
                    rest_ea_p1,
                    rest_eb_p0,
                    rest_eb_p1
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float C = thickness + d_hat - d;
                    Float3 normal = normalize_vec(x);
                    Float stiff = stiffness_repulsion * C;
                    // $if (d < 5e-3f) { stiff = 1e3f * C; };
                    // $if (d < 2e-3f) { stiff = 1e5f * C; };
                    // $if (d < 1e-3f) { stiff = 1e7f * C; };
                    {
                        Uint idx = narrowphase_count_ee->atomic(0).fetch_add(1u);
                        Var<CollisionPairEE> ee_pair;
                        ee_pair.indices = makeUint4(left_edge[0], left_edge[1], right_edge[0], right_edge[1]);
                        ee_pair.vec1 = makeFloat4(normal.x, normal.y, normal.z, stiff);
                        ee_pair.bary = bary;
                        {
                            Float4 weight = makeFloat4(bary[0], bary[1], -bary[2], -bary[3]);
                            Float12 G;
                            
                            for (uint j = 0; j < 4; j++)
                            {
                                G.vec[j] = stiff * weight[j] * normal;
                            }
    
                            Float12x12 H;
                            Float3x3 xxT = stiff * outer_product(normal, normal);
                            for (uint j = 0; j < 4; j++)
                            {
                                for (uint jj = 0; jj < 4; jj++)
                                {
                                    H.mat[j][jj] = weight[j] * weight[jj] * xxT;
                                }
                            }
                            ee_pair.gradient[0] = G.vec[0];
                            ee_pair.gradient[1] = G.vec[1];
                            ee_pair.gradient[2] = G.vec[2];
                            ee_pair.gradient[3] = G.vec[3];
                            CollisionPair::write_upper_hessian(ee_pair.hessian, H);
                            // luisa::compute::device_log("EE pair {} ({}) with C = {}, G = {} - {} - {} - {}", idx, ee_pair.indices, C, G.vec[0], G.vec[1], G.vec[2], G.vec[3]);
                        }
                        narrowphase_list_ee->write(idx, ee_pair);
                    }
                };
            };
            // Corner case (VV, VE) will only be considered in VF detection
        };
    });
}

// Device DCD
void NarrowPhasesDetector::vf_dcd_query_barrier(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<float3>& sa_rest_x_left, 
    const Buffer<float3>& sa_rest_x_right, 
    const Buffer<uint3>& sa_faces_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_vf_broadphase != 0)
    {
        stream << 
            fn_narrow_phase_vf_dcd_query_barrier(sa_x_left, sa_x_right, sa_rest_x_left, sa_rest_x_right, sa_faces_right, d_hat, thickness, kappa).dispatch(num_vf_broadphase);
    }

}

void NarrowPhasesDetector::ee_dcd_query_barrier(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<float3>& sa_rest_x_left, 
    const Buffer<float3>& sa_rest_x_right, 
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_ee_broadphase != 0)
    {
        stream << 
            fn_narrow_phase_ee_dcd_query_barrier(sa_x_left, sa_x_right, sa_rest_x_left, sa_rest_x_right, sa_edges_left, sa_edges_right, d_hat, thickness, kappa).dispatch(num_ee_broadphase);
    }
}
void NarrowPhasesDetector::vf_dcd_query_repulsion(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<float3>& sa_rest_x_left, 
    const Buffer<float3>& sa_rest_x_right, 
    const Buffer<uint3>& sa_faces_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_vf_broadphase != 0)
    {
        stream << 
            fn_narrow_phase_vf_dcd_query_repulsion(sa_x_left, sa_x_right, sa_rest_x_left, sa_rest_x_right, sa_faces_right, d_hat, thickness, kappa).dispatch(num_vf_broadphase);
    }

}

void NarrowPhasesDetector::ee_dcd_query_repulsion(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<float3>& sa_rest_x_left, 
    const Buffer<float3>& sa_rest_x_right, 
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_ee_broadphase != 0)
    {
        stream << 
            fn_narrow_phase_ee_dcd_query_repulsion(sa_x_left, sa_x_right, sa_rest_x_left, sa_rest_x_right, sa_edges_left, sa_edges_right, d_hat, thickness, kappa).dispatch(num_ee_broadphase);
    }
}

template<int N>
Eigen::Matrix<float, N, N> spd_projection(const Eigen::Matrix<float, N, N>& orig_matrix)
{
    // Ensure the matrix is symmetric
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, N, N>> eigensolver(orig_matrix);
    Eigen::Matrix<float, N, 1> eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix<float, N, N> eigenvectors = eigensolver.eigenvectors();

    // Set negative eigenvalues to zero (or abs, as in your python code)
    for (int i = 0; i < N; ++i) 
    {
        eigenvalues[i] = std::max(0.0f, eigenvalues[i]);
        // eigenvalues(i) = std::abs(eigenvalues(i));
    }

    // Reconstruct the matrix: V * diag(lam) * V^T
    Eigen::Matrix<float, N, N> D = eigenvalues.asDiagonal();
    return eigenvectors * D * eigenvectors.transpose();
}

void NarrowPhasesDetector::host_ON2_dcd_query_libuipc(
        Eigen::SparseMatrix<float>& eigen_cgA,
        Eigen::VectorXf& eigen_cgB,
        const std::vector<float3>& sa_x_left, 
        const std::vector<float3>& sa_x_right, 
        const std::vector<float3>& sa_rest_x_left, 
        const std::vector<float3>& sa_rest_x_right, 
        const std::vector<uint3>& sa_faces_left,
        const std::vector<uint3>& sa_faces_right,
        const std::vector<uint2>& sa_edges_left,
        const std::vector<uint2>& sa_edges_right,
        const float d_hat, 
        const float thickness,
        const float kappa)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    // Single Thread
    {
        std::vector<Eigen::Triplet<float>> triplets_vv(num_vv * 36);
        std::vector<Eigen::Triplet<float>> triplets_ve(num_ve * 81);
        std::vector<Eigen::Triplet<float>> triplets_vf(num_vf * 144);
        std::vector<Eigen::Triplet<float>> triplets_ee(num_ee * 144);

        Eigen::SparseMatrix<float> eigen_cgA_vv;
        Eigen::SparseMatrix<float> eigen_cgA_ve;
        Eigen::SparseMatrix<float> eigen_cgA_vf;
        Eigen::SparseMatrix<float> eigen_cgA_ee;
        eigen_cgA_vv.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_vv.reserve(triplets_vv.size());
        eigen_cgA_ve.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_ve.reserve(triplets_ve.size());
        eigen_cgA_vf.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_vf.reserve(triplets_vf.size());
        eigen_cgA_ee.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_ee.reserve(triplets_ee.size());

        std::atomic<uint> num_vv(0);
        std::atomic<uint> num_ve(0);
        std::atomic<uint> num_vf(0);
        std::atomic<uint> num_ee(0);

        // VF
        CpuParallel::single_thread_for(0, sa_x_left.size(), [&](const uint left)
        {
            const auto p = float3_to_eigen3(sa_x_left[left]);
            for (uint right = 0; right < sa_faces_right.size(); right++)
            {
                const uint3 right_face = sa_faces_right[right];
                if (left == right_face[0] || left == right_face[1] || left == right_face[2]) continue; // Skip self-contact
                const auto t0 = float3_to_eigen3(sa_x_right[right_face[0]]);
                const auto t1 = float3_to_eigen3(sa_x_right[right_face[1]]);
                const auto t2 = float3_to_eigen3(sa_x_right[right_face[2]]);

                // Bool is_ee = all_vec(bary != 0.0f);
                auto bary = host_distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
                // uint3 valid_indices = makeUint3(0, 1, 2);
                // uint valid_count = host_distance::point_triangle_type(bary, valid_indices);
                
                auto x = bary[0] * (t0 - p) +
                            bary[1] * (t1 - p) +
                            bary[2] * (t2 - p);
                float d2 = (x.squaredNorm());
                
                if (d2 < square_scalar(thickness + d_hat))
                {
                    float d = sqrt_scalar(d2);
                    CollisionPairVF vf_pair;
                    vf_pair.indices = makeUint4(left, right_face[0], right_face[1], right_face[2]);
                    vf_pair.vec1 = makeFloat4(x[0], x[1], x[1], d);
                    vf_pair.bary = eigen3_to_float3(bary);
                    Eigen::Vector<float, 12>          G;
                    Eigen::Matrix<float, 12, 12>      H;
                    {
                        Eigen::Vector4i flag = uipc::backend::cuda::distance::point_triangle_distance_flag(p, t0, t1, t2);
                        uipc::backend::cuda::sym::codim_ipc_simplex_contact::PT_barrier_gradient_hessian(
                           G, H, flag, kappa, d_hat, thickness, 
                           p, 
                           t0, 
                           t1, 
                           t2);
                        // luisa::log_info("Get VF Pair : indices = {}, bary = {}, d = {}", 
                        //     vf_pair.indices, 
                        //     vf_pair.bary, d);
                        H = spd_projection(H);
                    }
                    uint idx = num_vf.fetch_add(1);
                    host_collision_data->narrow_phase_list_vf[idx] = (vf_pair);

                    Eigen::Vector<uint, 12> insert_indice;
                    insert_indice << 
                        3 * left + 0,
                        3 * left + 1,
                        3 * left + 2,
                        3 * right_face[0] + 0,
                        3 * right_face[0] + 1,
                        3 * right_face[0] + 2,
                        3 * right_face[1] + 0,
                        3 * right_face[1] + 1,
                        3 * right_face[1] + 2,
                        3 * right_face[2] + 0,
                        3 * right_face[2] + 1,
                        3 * right_face[2] + 2;
                    for (uint i = 0; i < 12; ++i)
                    {
                        for (uint j = 0; j < 12; ++j) 
                        {
                            triplets_vf.push_back(Eigen::Triplet<float>(
                                insert_indice[i], 
                                insert_indice[j], 
                                H(i, j)
                            ));
                        }
                        eigen_cgB(insert_indice[i]) -= G(i);
                    }

                    // luisa::log_info("VF Pair : indices = {}, p = {}, f = {}/{}/{}",
                    //     vf_pair.indices, 
                    //     eigen3_to_float3(p), 
                    //     eigen3_to_float3(t0), eigen3_to_float3(t1), eigen3_to_float3(t2));
                    // std::cout << "VF Pair: indices = " << insert_indice.transpose() << " , d = " << d << ", G = " << G.transpose() << " , H = \n" << H << std::endl;
                }
            }
        });
        // EE
        CpuParallel::single_thread_for(0, sa_edges_left.size(), [&](const uint left)
        {
            const uint2 left_edge = sa_edges_left[left];
            const auto ea_p0 = float3_to_eigen3(sa_x_left[left_edge[0]]);
            const auto ea_p1 = float3_to_eigen3(sa_x_left[left_edge[1]]);
            for (uint right = left + 1; right < sa_edges_right.size(); right++)
            {
                const uint2 right_edge = sa_edges_right[right];
                if (left_edge[0] == right_edge[0] || left_edge[0] == right_edge[1] ||
                    left_edge[1] == right_edge[0] || left_edge[1] == right_edge[1]) continue; // Skip self-contact
                const auto eb_p0 = float3_to_eigen3(sa_x_right[right_edge[0]]);
                const auto eb_p1 = float3_to_eigen3(sa_x_right[right_edge[1]]);

                auto bary = host_distance::edge_edge_distance_coeff_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
                // Bool is_ee = all_vec(bary != 0.0f);
                bool is_ee = bary.isZero(0.0f);

                auto x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
                auto x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
                auto x = x1 - x0;
                float d2 = (x.squaredNorm());

                if (d2 < square_scalar(thickness + d_hat))
                {
                    float d = sqrt_scalar(d2);
                    CollisionPairEE ee_pair;
                    ee_pair.indices = makeUint4(left_edge[0], left_edge[1], right_edge[0], right_edge[1]);
                    ee_pair.vec1 = makeFloat4(x[0], x[1], x[1], d);
                    ee_pair.bary = eigen4_to_float4(bary);
                    Eigen::Vector<float, 12>          G;
                    Eigen::Matrix<float, 12, 12>      H;
                    const auto t0_Ea0 = float3_to_eigen3(sa_rest_x_left[left_edge[0]]);
                    const auto t0_Ea1 = float3_to_eigen3(sa_rest_x_left[left_edge[1]]);
                    const auto t0_Eb0 = float3_to_eigen3(sa_rest_x_right[right_edge[0]]);
                    const auto t0_Eb1 = float3_to_eigen3(sa_rest_x_right[right_edge[1]]);
                    {
                        Eigen::Vector4i flag = uipc::backend::cuda::distance::edge_edge_distance_flag(ea_p0, ea_p1, eb_p0, eb_p1);
                        uipc::backend::cuda::sym::codim_ipc_simplex_contact::mollified_EE_barrier_gradient_hessian(
                           G, H, flag, kappa, d_hat, thickness, 
                           t0_Ea0, 
                           t0_Ea1, 
                           t0_Eb0, 
                           t0_Eb1, 
                           ea_p0, 
                           ea_p1, 
                           eb_p0, 
                           eb_p1);
                        H = spd_projection(H);
                        // luisa::log_info("Get EE Pair : indices = {}, bary = {}, d = {}", 
                        //     ee_pair.indices, 
                        //     ee_pair.bary, d);
                    }
                    uint idx = num_ee.fetch_add(1);
                    host_collision_data->narrow_phase_list_ee[idx] = (ee_pair);

                    Eigen::Vector<uint, 12> insert_indice;
                    insert_indice << 
                        3 * left_edge[0] + 0,
                        3 * left_edge[0] + 1,
                        3 * left_edge[0] + 2,
                        3 * left_edge[1] + 0,
                        3 * left_edge[1] + 1,
                        3 * left_edge[1] + 2,
                        3 * right_edge[0] + 0,
                        3 * right_edge[0] + 1,
                        3 * right_edge[0] + 2,
                        3 * right_edge[1] + 0,
                        3 * right_edge[1] + 1,
                        3 * right_edge[1] + 2;
                    for (uint i = 0; i < 12; ++i)
                    {
                        for (uint j = 0; j < 12; ++j) 
                        {
                            triplets_ee.push_back(Eigen::Triplet<float>(
                                insert_indice[i], 
                                insert_indice[j], 
                                H(i, j)
                            ));
                        }
                        eigen_cgB(insert_indice[i]) -= G(i);
                    }
                    // luisa::log_info("EE Pair : indices = {}, e1 = {}/{}, e2 = {}/{}, t0e1 = {}/{}, t0e1 = {}/{}", 
                    //     ee_pair.indices, 
                    //     eigen3_to_float3(ea_p0), eigen3_to_float3(ea_p1), eigen3_to_float3(eb_p0), eigen3_to_float3(eb_p1),
                    //     eigen3_to_float3(t0_Ea0), eigen3_to_float3(t0_Ea1), eigen3_to_float3(t0_Eb0), eigen3_to_float3(t0_Eb1)
                    // );
                    // std::cout << "EE Pair: indices = " << insert_indice.transpose() << " , d = " << d << ", G = " << G.transpose() << std::endl;
                    // std::cout << "EE Pair: indices = " << insert_indice.transpose() << " , d = " << d << ", G = " << G.transpose() << " , H = \n" << H << std::endl;
                }
            }
        });

        eigen_cgA_vv.setFromTriplets(triplets_vv.begin(), triplets_vv.end());
        eigen_cgA_ve.setFromTriplets(triplets_ve.begin(), triplets_ve.end());
        eigen_cgA_vf.setFromTriplets(triplets_vf.begin(), triplets_vf.end());
        eigen_cgA_ee.setFromTriplets(triplets_ee.begin(), triplets_ee.end());
        eigen_cgA += eigen_cgA_vv + eigen_cgA_ve + eigen_cgA_vf + eigen_cgA_ee;
    }
    
    
}

} // namespace lcsv 

namespace lcsv // Compute Barrier Gradient & Hessian & Assemble
{

void NarrowPhasesDetector::compile_assemble(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    // Assemble
    auto atomic_add_float3 = [](
        Var<BufferView<float3>>& sa_cgB, const Uint& idx, const Float3& vec
    )
    {
        sa_cgB.atomic(idx)[0].fetch_add(vec[0]);
        sa_cgB.atomic(idx)[1].fetch_add(vec[1]);
        sa_cgB.atomic(idx)[2].fetch_add(vec[2]);
    };
    auto atomic_sub_float3 = [](
        Var<BufferView<float3>>& sa_cgB, const Uint& idx, const Float3& vec
    )
    {
        sa_cgB.atomic(idx)[0].fetch_sub(vec[0]);
        sa_cgB.atomic(idx)[1].fetch_sub(vec[1]);
        sa_cgB.atomic(idx)[2].fetch_sub(vec[2]);
    };
    auto atomic_add_float3x3 = [](
        Var<BufferView<float3x3>>& sa_cgA_diag, const Uint& idx, const Float3x3& mat
    )
    {
        sa_cgA_diag.atomic(idx)[0][0].fetch_add(mat[0][0]);
        sa_cgA_diag.atomic(idx)[0][1].fetch_add(mat[0][1]);
        sa_cgA_diag.atomic(idx)[0][2].fetch_add(mat[0][2]);
        sa_cgA_diag.atomic(idx)[1][0].fetch_add(mat[1][0]);
        sa_cgA_diag.atomic(idx)[1][1].fetch_add(mat[1][1]);
        sa_cgA_diag.atomic(idx)[1][2].fetch_add(mat[1][2]);
        sa_cgA_diag.atomic(idx)[2][0].fetch_add(mat[2][0]);
        sa_cgA_diag.atomic(idx)[2][1].fetch_add(mat[2][1]);
        sa_cgA_diag.atomic(idx)[2][2].fetch_add(mat[2][2]);
    };

    fn_assemble_barrier_hessian_gradient_vv = device.compile<1>(
    [
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view()
    , &atomic_sub_float3, &atomic_add_float3x3](
        Var<BufferView<float3>> sa_cgB, 
        Var<BufferView<float3x3>> sa_cgA_diag
    )
    {
        const auto& pair = narrowphase_list_vv->read(dispatch_x());
        const auto indices = CollisionPair::get_indices(pair);
        for (uint j = 0; j < 2; j++)
        {
            $if (
                luisa::compute::isnan(pair.gradient[j].x) |
                luisa::compute::isnan(pair.gradient[j].y) |
                luisa::compute::isnan(pair.gradient[j].z) 
            )
            {
                luisa::compute::device_log("VV Gradient is nan");
            };
            atomic_add_float3x3(sa_cgA_diag, indices[j], pair.hessian[j]);
            atomic_sub_float3(sa_cgB, indices[j], pair.gradient[j]);
        }
    });

    fn_assemble_barrier_hessian_gradient_ve = device.compile<1>(
    [
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view()
    , &atomic_sub_float3, &atomic_add_float3x3](
        Var<BufferView<float3>> sa_cgB, 
        Var<BufferView<float3x3>> sa_cgA_diag
    )
    {
        const auto& pair = narrowphase_list_ve->read(dispatch_x());
        const auto indices = CollisionPair::get_indices(pair);
        for (uint j = 0; j < 3; j++)
        {
            $if (
                luisa::compute::isnan(pair.gradient[j].x) |
                luisa::compute::isnan(pair.gradient[j].y) |
                luisa::compute::isnan(pair.gradient[j].z) 
            )
            {
                luisa::compute::device_log("VE Gradient is nan");
            };
            atomic_add_float3x3(sa_cgA_diag, indices[j], pair.hessian[j]);
            atomic_sub_float3(sa_cgB, indices[j], pair.gradient[j]);
        }
    });

    fn_assemble_barrier_hessian_gradient_vf = device.compile<1>(
    [
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    , &atomic_sub_float3, &atomic_add_float3x3](
        Var<BufferView<float3>> sa_cgB, 
        Var<BufferView<float3x3>> sa_cgA_diag
    )
    {
        const auto& pair = narrowphase_list_vf->read(dispatch_x());
        const auto indices = CollisionPair::get_indices(pair);
        for (uint j = 0; j < 4; j++)
        {
            $if (
                luisa::compute::isnan(pair.gradient[j].x) |
                luisa::compute::isnan(pair.gradient[j].y) |
                luisa::compute::isnan(pair.gradient[j].z) 
            )
            {
                luisa::compute::device_log("VF Gradient is nan");
            };
            atomic_add_float3x3(sa_cgA_diag, indices[j], pair.hessian[j]);
            atomic_sub_float3(sa_cgB, indices[j], pair.gradient[j]);
        }
    });
    
    fn_assemble_barrier_hessian_gradient_ee = device.compile<1>(
    [
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    , &atomic_sub_float3, &atomic_add_float3x3](
        Var<BufferView<float3>> sa_cgB, 
        Var<BufferView<float3x3>> sa_cgA_diag
    )
    {
        const auto& pair = narrowphase_list_ee->read(dispatch_x());
        const auto indices = CollisionPair::get_indices(pair);
        for (uint j = 0; j < 4; j++)
        {
            $if (
                luisa::compute::isnan(pair.gradient[j].x) |
                luisa::compute::isnan(pair.gradient[j].y) |
                luisa::compute::isnan(pair.gradient[j].z) 
            )
            {
                luisa::compute::device_log("EE Gradient is nan");
            };
            atomic_add_float3x3(sa_cgA_diag, indices[j], pair.hessian[j]);
            atomic_sub_float3(sa_cgB, indices[j], pair.gradient[j]);
        }
    });

    // Spring-form contact energy
    fn_assemble_repulsion_hessian_gradient_vf = device.compile<1>(
    [
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    , &atomic_add_float3, &atomic_add_float3x3](
        Var<BufferView<float3>> sa_cgB, 
        Var<BufferView<float3x3>> sa_cgA_diag
    )
    {
        const Uint pair_idx = dispatch_x();
        const auto& pair = narrowphase_list_vf->read(pair_idx);
        const auto indices = CollisionPair::get_indices(pair);

        const Float stiff = CollisionPair::get_stiff(pair);
        const Float3 normal = CollisionPair::get_direction(pair);
        const Float3 face_bary = CollisionPair::get_vf_face_bary(pair);
        const Float4 weight = makeFloat4(1.0f, -face_bary.x, -face_bary.y, -face_bary.z);
        const Float3x3 xxT = stiff * outer_product(normal, normal);

        for (uint j = 0; j < 4; j++)
        {
            Float3 force = stiff * weight[j] * normal;
            Float3x3 hessian = stiff * weight[j] * weight[j] * xxT;
            
            // device_log("VF pair {} on vert {} : force = {}, stiff = {}, weight = {}", pair_idx, indices[j], force, stiff, weight);
            atomic_add_float3x3(sa_cgA_diag, indices[j], hessian);
            atomic_add_float3(sa_cgB, indices[j], force);
        }
    });

    fn_assemble_repulsion_hessian_gradient_ee = device.compile<1>(
    [
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    , &atomic_add_float3, &atomic_add_float3x3](
        Var<BufferView<float3>> sa_cgB, 
        Var<BufferView<float3x3>> sa_cgA_diag
    )
    {
        const Uint pair_idx = dispatch_x();
        const auto& pair = narrowphase_list_ee->read(pair_idx);
        const auto indices = CollisionPair::get_indices(pair);

        const Float stiff = CollisionPair::get_stiff(pair);
        const Float3 normal = CollisionPair::get_direction(pair);
        const Float2 edge1_bary = CollisionPair::get_ee_edge1_bary(pair);
        const Float2 edge2_bary = CollisionPair::get_ee_edge2_bary(pair);
        const Float4 weight = makeFloat4(edge1_bary.x, edge1_bary.y, -edge2_bary.x, -edge2_bary.y);
        const Float3x3 xxT = stiff * outer_product(normal, normal);

        for (uint j = 0; j < 4; j++)
        {
            Float3 force = stiff * weight[j] * normal;
            Float3x3 hessian = stiff * weight[j] * weight[j] * xxT;
            // device_log("EE pair {} on vert {} : force = {}, stiff = {}, weight = {}", pair_idx, indices[j], force, stiff, weight);
            atomic_add_float3x3(sa_cgA_diag, indices[j], hessian);
            atomic_add_float3(sa_cgB, indices[j], force);
        }
    });
}

void NarrowPhasesDetector::host_barrier_gradient_hessian_assemble(
    luisa::compute::Stream& stream, 
    Eigen::SparseMatrix<float>& eigen_cgA,
    Eigen::VectorXf& eigen_cgB)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    // if constexpr (use_eigen) 
    // Eigen Decomposition
    // if constexpr (false)
    {
        std::vector<Eigen::Triplet<float>> triplets_vv(num_vv * 36);
        std::vector<Eigen::Triplet<float>> triplets_ve(num_ve * 81);
        std::vector<Eigen::Triplet<float>> triplets_vf(num_vf * 144);
        std::vector<Eigen::Triplet<float>> triplets_ee(num_ee * 144);

        Eigen::SparseMatrix<float> eigen_cgA_vv;
        Eigen::SparseMatrix<float> eigen_cgA_ve;
        Eigen::SparseMatrix<float> eigen_cgA_vf;
        Eigen::SparseMatrix<float> eigen_cgA_ee;
        eigen_cgA_vv.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_vv.reserve(triplets_vv.size());
        eigen_cgA_ve.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_ve.reserve(triplets_ve.size());
        eigen_cgA_vf.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_vf.reserve(triplets_vf.size());
        eigen_cgA_ee.resize(eigen_cgA.rows(), eigen_cgA.cols()); eigen_cgA_ee.reserve(triplets_ee.size());

        CpuParallel::parallel_for(0, num_vv, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_vv[pair_idx];
            uint2& indices = pair.indices;
            float6x6 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat6x6 proj_H = (float6x6_to_eigen6x6(H));
            for (uint i = 0; i < 2; ++i) 
            {
                for (uint j = 0; j < 2; ++j) 
                {
                    uint prefix = pair_idx * 36 + i * 18 + j * 9;
                    for (int ii = 0; ii < 3; ++ii)
                    {
                        for (int jj = 0; jj < 3; ++jj) 
                        {
                            triplets_vv[prefix + ii * 3 + jj] = Eigen::Triplet<float>(
                                3 * pair.indices[i] + ii, 
                                3 * pair.indices[j] + jj, 
                                proj_H(ii + i * 3, jj + j * 3));
                        }
                    }
                }
            }
        });
        CpuParallel::parallel_for(0, num_ve, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_ve[pair_idx];
            uint3 indices = makeUint3(pair.vid, pair.edge[0], pair.edge[1]);
            float9x9 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat9x9 proj_H = (float9x9_to_eigen9x9(H));
            for (uint i = 0; i < 3; ++i) 
            {
                for (uint j = 0; j < 3; ++j) 
                {
                    uint prefix = pair_idx * 81 + i * 27 + j * 9;
                    for (int ii = 0; ii < 3; ++ii)
                    {
                        for (int jj = 0; jj < 3; ++jj) 
                        {
                            triplets_ve[prefix + ii * 3 + jj] = Eigen::Triplet<float>(
                                3 * indices[i] + ii, 
                                3 * indices[j] + jj, 
                                proj_H(ii + i * 3, jj + j * 3));
                        }
                    }
                }
            }
        });
        CpuParallel::parallel_for(0, num_vf, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_vf[pair_idx];
            uint4& indices = pair.indices;  luisa::log_info("Get VF Pair : indices = {}", indices);;
            float12x12 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat12x12 proj_H = (float12x12_to_eigen12x12(H));
            
            for (uint i = 0; i < 4; ++i) 
            {
                for (uint j = 0; j < 4; ++j) 
                {
                    uint prefix = pair_idx * 144 + i * 36 + j * 9;
                    for (int ii = 0; ii < 3; ++ii)
                    {
                        for (int jj = 0; jj < 3; ++jj) 
                        {
                            triplets_vf[prefix + ii * 3 + jj] = Eigen::Triplet<float>(
                                3 * indices[i] + ii, 
                                3 * indices[j] + jj, 
                                proj_H(ii + i * 3, jj + j * 3));
                        }
                    }
                }
            }
        });
        CpuParallel::parallel_for(0, num_ee, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_ee[pair_idx];
            uint4& indices = pair.indices; luisa::log_info("Get VF Pair : indices = {}", indices);;
            float12x12 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat12x12 proj_H = (float12x12_to_eigen12x12(H));
            for (uint i = 0; i < 4; ++i) 
            {
                for (uint j = 0; j < 4; ++j) 
                {
                    uint prefix = pair_idx * 144 + i * 36 + j * 9;
                    for (int ii = 0; ii < 3; ++ii)
                    {
                        for (int jj = 0; jj < 3; ++jj) 
                        {
                            triplets_ee[prefix + ii * 3 + jj] = Eigen::Triplet<float>(
                                3 * indices[i] + ii, 
                                3 * indices[j] + jj, 
                                proj_H(ii + i * 3, jj + j * 3));
                        }
                    }
                }
            }
        });
        CpuParallel::single_thread_for(0, num_vv, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_vv[pair_idx];
            eigen_cgB.segment<3>(3 * pair.indices[0]) += float3_to_eigen3(pair.gradient[0]);
            eigen_cgB.segment<3>(3 * pair.indices[1]) += float3_to_eigen3(pair.gradient[1]);
        });
        CpuParallel::single_thread_for(0, num_ve, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_ve[pair_idx];
            eigen_cgB.segment<3>(3 * pair.vid)     += float3_to_eigen3(pair.gradient[0]);
            eigen_cgB.segment<3>(3 * pair.edge[0]) += float3_to_eigen3(pair.gradient[1]);
            eigen_cgB.segment<3>(3 * pair.edge[1]) += float3_to_eigen3(pair.gradient[2]);
        });
        CpuParallel::single_thread_for(0, num_vf, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_vf[pair_idx];
            eigen_cgB.segment<3>(3 * pair.indices[0]) += float3_to_eigen3(pair.gradient[0]);
            eigen_cgB.segment<3>(3 * pair.indices[1]) += float3_to_eigen3(pair.gradient[1]);
            eigen_cgB.segment<3>(3 * pair.indices[2]) += float3_to_eigen3(pair.gradient[2]);
            eigen_cgB.segment<3>(3 * pair.indices[3]) += float3_to_eigen3(pair.gradient[3]);
        });
        CpuParallel::single_thread_for(0, num_ee, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_ee[pair_idx];
            eigen_cgB.segment<3>(3 * pair.indices[0]) += float3_to_eigen3(pair.gradient[0]);
            eigen_cgB.segment<3>(3 * pair.indices[1]) += float3_to_eigen3(pair.gradient[1]);
            eigen_cgB.segment<3>(3 * pair.indices[2]) += float3_to_eigen3(pair.gradient[2]);
            eigen_cgB.segment<3>(3 * pair.indices[3]) += float3_to_eigen3(pair.gradient[3]);
        });
        
        eigen_cgA_vv.setFromTriplets(triplets_vv.begin(), triplets_vv.end());
        eigen_cgA_ve.setFromTriplets(triplets_ve.begin(), triplets_ve.end());
        eigen_cgA_vf.setFromTriplets(triplets_vf.begin(), triplets_vf.end());
        eigen_cgA_ee.setFromTriplets(triplets_ee.begin(), triplets_ee.end());
        eigen_cgA += eigen_cgA_vv + eigen_cgA_ve + eigen_cgA_vf + eigen_cgA_ee;
    }
}
void NarrowPhasesDetector::host_barrier_hessian_spd_projection(
    luisa::compute::Stream& stream)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    {
        CpuParallel::parallel_for(0, num_vv, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_vv[pair_idx];
            uint2& indices = pair.indices;
            float6x6 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat6x6 proj_H = spd_projection(float6x6_to_eigen6x6(H));
            H = eigen6x6_to_float6x6(proj_H);
            CollisionPair::write_upper_hessian(pair.hessian, H);
        });
        CpuParallel::parallel_for(0, num_ve, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_ve[pair_idx];
            uint3 indices = makeUint3(pair.vid, pair.edge[0], pair.edge[1]);
            float9x9 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat9x9 proj_H = spd_projection(float9x9_to_eigen9x9(H));
            H = eigen9x9_to_float9x9(proj_H);
            CollisionPair::write_upper_hessian(pair.hessian, H);
        });
        CpuParallel::parallel_for(0, num_vf, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_vf[pair_idx];
            uint4& indices = pair.indices; //  luisa::log_info("Get VF Pair : indices = {}", indices);;
            float12x12 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat12x12 proj_H = spd_projection(float12x12_to_eigen12x12(H));
            H = eigen12x12_to_float12x12(proj_H);
            CollisionPair::write_upper_hessian(pair.hessian, H);
        });
        CpuParallel::parallel_for(0, num_ee, [&](const uint pair_idx)
        {
            auto& pair = host_collision_data->narrow_phase_list_ee[pair_idx];
            uint4& indices = pair.indices; // luisa::log_info("Get VF Pair : indices = {}", indices);;
            float12x12 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
            EigenFloat12x12 proj_H = spd_projection(float12x12_to_eigen12x12(H));
            H = eigen12x12_to_float12x12(proj_H);
            CollisionPair::write_upper_hessian(pair.hessian, H);
        });
    }
}

void NarrowPhasesDetector::barrier_hessian_assemble(luisa::compute::Stream& stream, Buffer<float3>& sa_cgB, Buffer<float3x3>& sa_cgA_diag)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    stream 
        << fn_assemble_barrier_hessian_gradient_vv(sa_cgB, sa_cgA_diag).dispatch(num_vv)
        << fn_assemble_barrier_hessian_gradient_ve(sa_cgB, sa_cgA_diag).dispatch(num_ve)
        << fn_assemble_barrier_hessian_gradient_vf(sa_cgB, sa_cgA_diag).dispatch(num_vf)
        << fn_assemble_barrier_hessian_gradient_ee(sa_cgB, sa_cgA_diag).dispatch(num_ee);
}
void NarrowPhasesDetector::repulsion_hessian_assemble(luisa::compute::Stream& stream, Buffer<float3>& sa_cgB, Buffer<float3x3>& sa_cgA_diag)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    stream 
        << fn_assemble_repulsion_hessian_gradient_vf(sa_cgB, sa_cgA_diag).dispatch(num_vf)
        << fn_assemble_repulsion_hessian_gradient_ee(sa_cgB, sa_cgA_diag).dispatch(num_ee);
}

void NarrowPhasesDetector::host_spmv_barrier(Stream& stream, const std::vector<float3>& input_array, std::vector<float3>& output_array)
{
    // Off-diag: Collision hessian
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    CpuParallel::single_thread_for(0, num_vv, [&](const uint pair_idx)
    {
        auto& pair = host_collision_data->narrow_phase_list_vv[pair_idx];
        auto indices = CollisionPair::get_indices(pair);
        float6x6 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
        
        H.mat[0][0] = makeFloat3x3(0.0f);
        H.mat[1][1] = makeFloat3x3(0.0f);

        float6 input_vec;
        float6 output_vec;
        float3 input[2] = {
            input_array[indices[0]],
            input_array[indices[1]],
        }; set_largevec(input_vec, input);
        mult_largemat_vec(output_vec, H, output_vec);
        output_array[indices[0]] += output_vec.vec[0];
        output_array[indices[1]] += output_vec.vec[1];
    });
    CpuParallel::single_thread_for(0, num_ve, [&](const uint pair_idx)
    {
        auto& pair = host_collision_data->narrow_phase_list_ve[pair_idx];
        auto indices = CollisionPair::get_indices(pair);
        float9x9 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
        
        H.mat[0][0] = makeFloat3x3(0.0f);
        H.mat[1][1] = makeFloat3x3(0.0f);
        H.mat[2][2] = makeFloat3x3(0.0f);

        float9 input_vec;
        float9 output_vec;
        float3 input[3] = {
            input_array[indices[0]],
            input_array[indices[1]],
            input_array[indices[2]],
        }; set_largevec(input_vec, input);
        mult_largemat_vec(output_vec, H, output_vec);
        output_array[indices[0]] += output_vec.vec[0];
        output_array[indices[1]] += output_vec.vec[1];
        output_array[indices[2]] += output_vec.vec[2];
    });
    CpuParallel::single_thread_for(0, num_vf, [&](const uint pair_idx)
    {
        auto& pair = host_collision_data->narrow_phase_list_vf[pair_idx];
        auto indices = CollisionPair::get_indices(pair);
        float12x12 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
       
        H.mat[0][0] = makeFloat3x3(0.0f);
        H.mat[1][1] = makeFloat3x3(0.0f);
        H.mat[2][2] = makeFloat3x3(0.0f);
        H.mat[3][3] = makeFloat3x3(0.0f);

        float12 input_vec;
        float12 output_vec;
        float3 input[4] = {
            input_array[indices[0]],
            input_array[indices[1]],
            input_array[indices[2]],
            input_array[indices[3]],
        }; set_largevec(input_vec, input);
        mult_largemat_vec(output_vec, H, output_vec);
        output_array[indices[0]] += output_vec.vec[0];
        output_array[indices[1]] += output_vec.vec[1];
        output_array[indices[2]] += output_vec.vec[2];
        output_array[indices[3]] += output_vec.vec[3];
    });
    CpuParallel::single_thread_for(0, num_ee, [&](const uint pair_idx)
    {
        auto& pair = host_collision_data->narrow_phase_list_ee[pair_idx];
        auto indices = CollisionPair::get_indices(pair);
        float12x12 H; CollisionPair::extract_upper_hessian(pair.hessian, H);
        
        H.mat[0][0] = makeFloat3x3(0.0f);
        H.mat[1][1] = makeFloat3x3(0.0f);
        H.mat[2][2] = makeFloat3x3(0.0f);
        H.mat[3][3] = makeFloat3x3(0.0f);
        
        float12 input_vec;
        float12 output_vec;
        float3 input[4] = {
            input_array[indices[0]],
            input_array[indices[1]],
            input_array[indices[2]],
            input_array[indices[3]],
        }; set_largevec(input_vec, input);
        mult_largemat_vec(output_vec, H, output_vec);
        output_array[indices[0]] += output_vec.vec[0];
        output_array[indices[1]] += output_vec.vec[1];
        output_array[indices[2]] += output_vec.vec[2];
        output_array[indices[3]] += output_vec.vec[3];
    });
}
void NarrowPhasesDetector::host_spmv_repulsion(Stream& stream, const std::vector<float3>& input_array, std::vector<float3>& output_array)
{
    // Off-diag: Collision hessian
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    CpuParallel::single_thread_for(0, num_vf, [&](const uint pair_idx)
    {
        auto& pair = host_collision_data->narrow_phase_list_vf[pair_idx];
        auto indices = CollisionPair::get_indices(pair);
        
        float3 input_vec[4] = {
            input_array[indices[0]],
            input_array[indices[1]],
            input_array[indices[2]],
            input_array[indices[3]],
        }; 
        float3 output_vec[4] = {
            Zero3,
            Zero3,
            Zero3,
            Zero3,
        }; 
        const float stiff = CollisionPair::get_stiff(pair);
        const float3 normal = CollisionPair::get_direction(pair);
        const float3 face_bary = CollisionPair::get_vf_face_bary(pair);
        const float4 weight = makeFloat4(1.0f, -face_bary.x, -face_bary.y, -face_bary.z);
        const float3x3 xxT = stiff * outer_product(normal, normal);

        for (uint j = 0; j < 4; j++)
        {
            for (uint jj = 0; jj < 4; jj++)
            {
                if (j != jj)
                {
                    float3x3 hessian = weight[j] * weight[jj] * xxT;
                    output_vec[j] += hessian * output_vec[jj];
                }
            }
        }
        output_array[indices[0]] += output_vec[0];
        output_array[indices[1]] += output_vec[1];
        output_array[indices[2]] += output_vec[2];
        output_array[indices[3]] += output_vec[3];
    });
    CpuParallel::single_thread_for(0, num_ee, [&](const uint pair_idx)
    {
        auto& pair = host_collision_data->narrow_phase_list_ee[pair_idx];
        auto indices = CollisionPair::get_indices(pair);
        
        float3 input_vec[4] = {
            input_array[indices[0]],
            input_array[indices[1]],
            input_array[indices[2]],
            input_array[indices[3]],
        }; 
        float3 output_vec[4] = {
            Zero3,
            Zero3,
            Zero3,
            Zero3,
        }; 
        const float stiff = CollisionPair::get_stiff(pair);
        const float3 normal = CollisionPair::get_direction(pair);
        const float2 edge1_bary = CollisionPair::get_ee_edge1_bary(pair);
        const float2 edge2_bary = CollisionPair::get_ee_edge2_bary(pair);
        const float4 weight = makeFloat4(edge1_bary.x, edge1_bary.y, -edge2_bary.x, -edge2_bary.y);
        const float3x3 xxT = stiff * outer_product(normal, normal);

        for (uint j = 0; j < 4; j++)
        {
            for (uint jj = 0; jj < 4; jj++)
            {
                if (j != jj)
                {
                    float3x3 hessian = weight[j] * weight[jj] * xxT;
                    output_vec[j] += hessian * output_vec[jj];
                }
            }
        }
        output_array[indices[0]] += output_vec[0];
        output_array[indices[1]] += output_vec[1];
        output_array[indices[2]] += output_vec[2];
        output_array[indices[3]] += output_vec[3];
    });
}


} // namespace lcsv



namespace lcsv // Compute barrier energy
{

void NarrowPhasesDetector::compile_energy(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    fn_compute_barrier_energy_from_vf = device.compile<1>(
    [
        contact_energy = collision_data->contact_energy.view(2, 1),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view(),
        narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
        narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
        narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    ]( 
        Var<BufferView<float3>> sa_x_left, 
        Var<BufferView<float3>> sa_x_right,
        Var<BufferView<float3>> sa_rest_x_left, 
        Var<BufferView<float3>> sa_rest_x_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);
        const Uint3 face = sa_faces_right.read(fid);

        Float energy = 0.0f;
        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {

        }
        $else
        {
            Float3 p =  sa_x_left->read(vid);
            Float3 t0 = sa_x_right->read(face[0]);
            Float3 t1 = sa_x_right->read(face[1]);
            Float3 t2 = sa_x_right->read(face[2]);

            Float3 bary = distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
            Float3 x = bary[0] * (t0 - p) +
                       bary[1] * (t1 - p) +
                       bary[2] * (t2 - p);
            Float d2 = length_squared_vec(x);
            $if (d2 < square_scalar(thickness + d_hat))
            {
                Float3 rest_p  = sa_rest_x_left->read(vid);
                Float3 rest_t0 = sa_rest_x_right->read(face[0]);
                Float3 rest_t1 = sa_rest_x_right->read(face[1]);
                Float3 rest_t2 = sa_rest_x_right->read(face[2]);
                
                Float rest_d2 = distance::point_triangle_distance_squared_unclassified(
                    rest_p,
                    rest_t0,
                    rest_t1,
                    rest_t2
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    cipc::KappaBarrier(energy, kappa, d2, d_hat, thickness);
                    // device_log("        VF pair {} 's energy = {}, d = {}, thickness = {}, d_hat = {}, kappa = {}", 
                    //     pair_idx, energy, sqrt_scalar(d2), thickness, d_hat, kappa);
                    // cipc::NoKappa_Barrier(energy, d2, d_hat, thickness);
                    // device_log("pair {} 's energy = {}, d = {}, d_hat = {}, vert = {}, face = {}", 
                    //     pair_idx, energy, sqrt_scalar(d2), thickness + d_hat, vid, face);
                };
            };
        };
        
        energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

        $if (pair_idx % 256 == 0)
        {
            $if (energy != 0.0f) 
            {
                contact_energy->atomic(0).fetch_add(energy);
            };
        };
    });

    fn_compute_barrier_energy_from_ee = device.compile<1>(
    [
        contact_energy = collision_data->contact_energy.view(3, 1),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view(),
        narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_a, 
        Var<BufferView<float3>> sa_x_b,
        Var<BufferView<float3>> sa_rest_x_left, 
        Var<BufferView<float3>> sa_rest_x_right,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);

        Float energy = 0.0f;
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
        }
        $else
        {
            Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
            Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
            Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
            Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
    
            Float4 bary = distance::edge_edge_distance_coeff_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
            Bool is_ee = all_vec(bary != 0.0f);
            Float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
            Float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
            Float d2 = length_squared_vec(x1 - x0);

            $if (d2 < square_scalar(thickness + d_hat))
            {
                Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
                Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
                Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
                Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
        
                Float d2 = distance::edge_edge_distance_squared_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
                $if (d2 < square_scalar(thickness + d_hat))
                {
                    cipc::KappaBarrier(energy, kappa, d2, d_hat, thickness);
                };

                // cipc::NoKappa_Barrier(energy, d2, d_hat, thickness);
                // device_log("        EE pair {} 's energy = {}, d = {}, thickness = {}, d_hat = {}, kappa = {}", 
                //     pair_idx, energy, sqrt_scalar(d2), thickness, d_hat, kappa);
            };
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        
        $if (pair_idx % 256 == 0)
        {
            $if (energy != 0.0f) 
            {
                contact_energy->atomic(0).fetch_add(energy);
            };
        };
    });

    fn_compute_repulsion_energy_from_vf = device.compile<1>(
    [
        contact_energy = collision_data->contact_energy.view(2, 1),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
        broadphase_list = collision_data->broad_phase_list_vf.view(),
        narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
        narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
        narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
        narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
        narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
        narrowphase_list_vf = collision_data->narrow_phase_list_vf.view()
    ]( 
        Var<BufferView<float3>> sa_x_left, 
        Var<BufferView<float3>> sa_x_right,
        Var<BufferView<float3>> sa_rest_x_left, 
        Var<BufferView<float3>> sa_rest_x_right,
        Var<BufferView<uint3>> sa_faces_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint vid =  broadphase_list->read(2 * pair_idx + 0);
        const Uint fid = broadphase_list->read(2 * pair_idx + 1);
        const Uint3 face = sa_faces_right.read(fid);

        Float energy = 0.0f;
        $if (
            vid == face[0] | 
            vid == face[1] | 
            vid == face[2]) 
        {

        }
        $else
        {
            Float3 p =  sa_x_left->read(vid);
            Float3 t0 = sa_x_right->read(face[0]);
            Float3 t1 = sa_x_right->read(face[1]);
            Float3 t2 = sa_x_right->read(face[2]);

            Float d2 = distance::point_triangle_distance_squared_unclassified(p, t0, t1, t2);
            $if (d2 < square_scalar(thickness + d_hat) & d2 > 1e-8f)
            {
                Float3 rest_p  = sa_rest_x_left->read(vid);
                Float3 rest_t0 = sa_rest_x_right->read(face[0]);
                Float3 rest_t1 = sa_rest_x_right->read(face[1]);
                Float3 rest_t2 = sa_rest_x_right->read(face[2]);
                
                Float rest_d2 = distance::point_triangle_distance_squared_unclassified(
                    rest_p,
                    rest_t0,
                    rest_t1,
                    rest_t2
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float C = thickness + d_hat - d;
                    energy = 0.5f * stiffness_repulsion * C * C;
                };
            };
        };
        
        energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

        $if (pair_idx % 256 == 0)
        {
            $if (energy != 0.0f) 
            {
                contact_energy->atomic(0).fetch_add(energy);
            };
        };
    });

    fn_compute_repulsion_energy_from_ee = device.compile<1>(
    [
        contact_energy = collision_data->contact_energy.view(3, 1),
        broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
        broadphase_list = collision_data->broad_phase_list_ee.view(),
        narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
        narrowphase_list_ee = collision_data->narrow_phase_list_ee.view()
    ](
        Var<BufferView<float3>> sa_x_a, 
        Var<BufferView<float3>> sa_x_b,
        Var<BufferView<float3>> sa_rest_x_a, 
        Var<BufferView<float3>> sa_rest_x_b,
        Var<BufferView<uint2>> sa_edges_left,
        Var<BufferView<uint2>> sa_edges_right,
        Float d_hat,
        Float thickness,
        Float kappa
    )
    {
        const Uint pair_idx = dispatch_x();
        const Uint left =  broadphase_list->read(2 * pair_idx + 0);
        const Uint right = broadphase_list->read(2 * pair_idx + 1);
        const Uint2 left_edge  = sa_edges_left.read(left);
        const Uint2 right_edge = sa_edges_right.read(right);

        Float energy = 0.0f;
        $if (
            left_edge[0] == right_edge[0] |
            left_edge[0] == right_edge[1] |
            left_edge[1] == right_edge[0] |
            left_edge[1] == right_edge[1]) 
        {
        }
        $else
        {
            Float3 ea_p0 = (sa_x_a->read(left_edge[0]));
            Float3 ea_p1 = (sa_x_a->read(left_edge[1]));
            Float3 eb_p0 = (sa_x_b->read(right_edge[0]));
            Float3 eb_p1 = (sa_x_b->read(right_edge[1]));
    
            Float d2 = distance::edge_edge_distance_squared_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
            $if (d2 < square_scalar(thickness + d_hat) & d2 > 1e-8f)
            {
                Float3 rest_ea_x0 = (sa_rest_x_a->read(left_edge[0]));
                Float3 rest_ea_x1 = (sa_rest_x_a->read(left_edge[1]));
                Float3 rest_eb_x0 = (sa_rest_x_b->read(right_edge[0]));
                Float3 rest_eb_x1 = (sa_rest_x_b->read(right_edge[1]));
    
                Float rest_d2 = distance::edge_edge_distance_squared_unclassified(
                    rest_ea_x0,
                    rest_ea_x1,
                    rest_eb_x0,
                    rest_eb_x1
                );
                $if (rest_d2 > square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float C = thickness + d_hat - d;
                    energy = 0.5f * stiffness_repulsion * C * C;
                };
            };
        };

        energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);
        
        $if (pair_idx % 256 == 0)
        {
            $if (energy != 0.0f) 
            {
                contact_energy->atomic(0).fetch_add(energy);
            };
        };
    });

}

void NarrowPhasesDetector::compute_barrier_energy_from_vf(Stream& stream, 
        const Buffer<float3>& sa_x_left, 
        const Buffer<float3>& sa_x_right, 
        const Buffer<float3>& sa_rest_x_left, 
        const Buffer<float3>& sa_rest_x_right, 
        const Buffer<uint3>& sa_faces_right,
        const float d_hat,
        const float thickness,
        const float kappa)
{
    auto& contact_energy = collision_data->contact_energy;
    auto& host_contact_energy = host_collision_data->contact_energy;
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_vf_broadphase != 0)
    {
        stream << fn_compute_repulsion_energy_from_vf(
        // stream << fn_compute_barrier_energy_from_vf(
            sa_x_left,
            sa_x_right,
            sa_rest_x_left,
            sa_rest_x_right,
            sa_faces_right, d_hat, thickness, kappa
        ).dispatch(num_vf_broadphase) 
            // << contact_energy.view(2, 1).copy_to(host_contact_energy.data() + 2)
        ;
    }
}

void NarrowPhasesDetector::compute_barrier_energy_from_ee(Stream& stream, 
    const Buffer<float3>& sa_x_left, 
    const Buffer<float3>& sa_x_right, 
    const Buffer<float3>& sa_rest_x_left, 
    const Buffer<float3>& sa_rest_x_right, 
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float d_hat,
    const float thickness,
        const float kappa)
{
    auto& contact_energy = collision_data->contact_energy;
    auto& host_contact_energy = host_collision_data->contact_energy;
    auto broadphase_count = collision_data->broad_phase_collision_count.view();
    auto& host_count = host_collision_data->broad_phase_collision_count;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_ee_broadphase != 0)
    {
        // stream << fn_compute_barrier_energy_from_ee(
        stream << fn_compute_repulsion_energy_from_ee(
            sa_x_left,
            sa_x_right, // sa_x_begin_right
            sa_rest_x_left,
            sa_rest_x_right,
            sa_edges_left, sa_edges_right,
            d_hat, thickness, kappa
        ).dispatch(num_ee_broadphase) 
        // stream << fn_compute_barrier_energy_from_ee(
        //     sa_x_left,
        //     sa_x_right, // sa_x_begin_right
        //     sa_edges_left, sa_edges_right,
        //     d_hat, thickness, kappa
        // ).dispatch(num_ee_broadphase) 
            // << contact_energy.view(3, 1).copy_to(host_contact_energy.data() + 3)
        ;
    }
    
}

double NarrowPhasesDetector::host_ON2_compute_barrier_energy_uipc(
    const std::vector<float3>& sa_x_left, 
    const std::vector<float3>& sa_x_right, 
    const std::vector<float3>& sa_rest_x_left,
    const std::vector<float3>& sa_rest_x_right,
    const std::vector<uint3>& sa_faces_left,
    const std::vector<uint3>& sa_faces_right,
    const std::vector<uint2>& sa_edge_left,
    const std::vector<uint2>& sa_edge_right,
    const float d_hat,
    const float thickness,
    const float kappa
)
{
    double total_energy = 0.0f;
    // VF
    CpuParallel::single_thread_for(0, sa_x_left.size(), [&](const uint left)
    {
        const auto p = float3_to_eigen3(sa_x_left[left]);
        for (uint right = 0; right < sa_faces_right.size(); right++)
        {
            const uint3 right_face = sa_faces_right[right];
            if (left == right_face[0] || left == right_face[1] || left == right_face[2]) continue; // Skip self-contact
            const auto t0 = float3_to_eigen3(sa_x_right[right_face[0]]);
            const auto t1 = float3_to_eigen3(sa_x_right[right_face[1]]);
            const auto t2 = float3_to_eigen3(sa_x_right[right_face[2]]);

            // Bool is_ee = all_vec(bary != 0.0f);
            auto bary = host_distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
            // uint3 valid_indices = makeUint3(0, 1, 2);
            // uint valid_count = host_distance::point_triangle_type(bary, valid_indices);
            
            auto x = bary[0] * (t0 - p) +
                        bary[1] * (t1 - p) +
                        bary[2] * (t2 - p);
            float d2 = (x.squaredNorm());
            
            if (d2 < square_scalar(thickness + d_hat))
            {
                float d = sqrt_scalar(d2);
                {
                    Eigen::Vector4i flag = uipc::backend::cuda::distance::point_triangle_distance_flag(p, t0, t1, t2);
                    auto e = uipc::backend::cuda::sym::codim_ipc_simplex_contact::PT_barrier_energy(
                        flag, kappa, d_hat, thickness, 
                        p, 
                        t0, 
                        t1, 
                        t2);
                    
                    luisa::log_info("        VF pair {}/{} 's energy = {}, d = {}, thickness = {}, d_hat = {}, kappa = {}", 
                        left, right, e, sqrt_scalar(d2), thickness, d_hat, kappa);
                    total_energy += e;
                }
            }
        }
    });
    // EE
    CpuParallel::single_thread_for(0, sa_edge_left.size(), [&](const uint left)
    {
        const uint2 left_edge = sa_edge_left[left];
        const auto ea_p0 = float3_to_eigen3(sa_x_left[left_edge[0]]);
        const auto ea_p1 = float3_to_eigen3(sa_x_left[left_edge[1]]);
        for (uint right = left + 1; right < sa_edge_right.size(); right++)
        {
            const uint2 right_edge = sa_edge_right[right];
            if (left_edge[0] == right_edge[0] || left_edge[0] == right_edge[1] ||
                left_edge[1] == right_edge[0] || left_edge[1] == right_edge[1]) continue; // Skip self-contact
            const auto eb_p0 = float3_to_eigen3(sa_x_right[right_edge[0]]);
            const auto eb_p1 = float3_to_eigen3(sa_x_right[right_edge[1]]);

            auto bary = host_distance::edge_edge_distance_coeff_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
            // Bool is_ee = all_vec(bary != 0.0f);
            bool is_ee = bary.isZero(0.0f);

            auto x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
            auto x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
            auto x = x1 - x0;
            float d2 = (x.squaredNorm());

            if (d2 < square_scalar(thickness + d_hat))
            {
                float d = sqrt_scalar(d2);
                {
                    const auto t0_Ea0 = float3_to_eigen3(sa_rest_x_left[left_edge[0]]);
                    const auto t0_Ea1 = float3_to_eigen3(sa_rest_x_left[left_edge[1]]);
                    const auto t0_Eb0 = float3_to_eigen3(sa_rest_x_right[right_edge[0]]);
                    const auto t0_Eb1 = float3_to_eigen3(sa_rest_x_right[right_edge[1]]);
                    Eigen::Vector4i flag = uipc::backend::cuda::distance::edge_edge_distance_flag(ea_p0, ea_p1, eb_p0, eb_p1);
                    auto e = uipc::backend::cuda::sym::codim_ipc_simplex_contact::mollified_EE_barrier_energy(
                        flag, kappa, d_hat, thickness, 
                        t0_Ea0, 
                        t0_Ea1, 
                        t0_Eb0, 
                        t0_Eb1, 
                        ea_p0, 
                        ea_p1, 
                        eb_p0, 
                        eb_p1);
                    luisa::log_info("        EE pair {}/{} 's energy = {}, d = {}, thickness = {}, d_hat = {}, kappa = {}", 
                        left, right, e, sqrt_scalar(d2), thickness, d_hat, kappa);
                    total_energy += e;
                }
            }
        }
    });
    return total_energy;
}

} // namespace lcsv 

namespace lcsv // Host CCD
{

void NarrowPhasesDetector::host_vf_ccd_query(Stream& stream, 
    const std::vector<float3>& sa_x_begin_left, 
    const std::vector<float3>& sa_x_begin_right, 
    const std::vector<float3>& sa_x_end_left,
    const std::vector<float3>& sa_x_end_right,
    const std::vector<uint3>& sa_faces_right,
    const float d_hat, 
    const float thickness)
{
    auto& sa_toi = host_collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;
    auto& host_list = host_collision_data->broad_phase_list_vf;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()]; 
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];
    stream 
        << collision_data->broad_phase_list_vf.view(0, num_vf_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    // luisa::log_info("num_vf_broadphase = {}", num_vf_broadphase);
    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_vf_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

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
                                      d_hat + thickness);

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

    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);

    // min_toi /= host_accd::line_search_max_t;
    // if (min_toi < 1e-5)
    // {
    //     luisa::log_error("toi is too small : {}", min_toi);
    // }
    // luisa::log_info("toi = {}", min_toi);
    // sa_toi[0] = min_toi;

}

void NarrowPhasesDetector::host_ee_ccd_query(Stream& stream, 
    const std::vector<float3>& sa_x_begin_a, 
    const std::vector<float3>& sa_x_begin_b, 
    const std::vector<float3>& sa_x_end_a,
    const std::vector<float3>& sa_x_end_b,
    const std::vector<uint2>& sa_edges_left,
    const std::vector<uint2>& sa_edges_right,
    const float d_hat, 
    const float thickness)
{
    auto& sa_toi = host_collision_data->toi_per_vert;
    auto& host_count = host_collision_data->broad_phase_collision_count;
    auto& host_list = host_collision_data->broad_phase_list_ee;

    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()]; 
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    stream 
        << collision_data->broad_phase_list_ee.view(0, num_ee_broadphase * 2).copy_to(host_list.data()) 
        << luisa::compute::synchronize();

    // luisa::log_info("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcsv::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_ee_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

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
            d_hat + thickness);

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

    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);

    // min_toi /= host_accd::line_search_max_t;
    // if (min_toi < 1e-5)
    // {
    //     luisa::log_error("toi is too small : {}", min_toi);
    // }
    // luisa::log_info("toi = {}", min_toi);
    // sa_toi[0] = min_scalar(min_toi, sa_toi[0]);
}

void NarrowPhasesDetector::unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    using namespace luisa::compute;

    // VF CCD Test
    if constexpr (false)
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
    if constexpr (false)
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

    // VF Barrier Gradient/Hessian Test
    if constexpr (false)
    {
        luisa::log_info("////////////////////////////////");
        luisa::log_info("VF Barrier Gradient/Hessian Test");
        luisa::log_info("////////////////////////////////");

        const float kappa = 1e5;
        const float d_hat = 1e-2;
        const float thickness = 0.0f;

        auto test_p = float3(0.49999505, -0.29309484, 0.45634925);
        auto test_t0 = float3(-0.4, -0.3, -0.5);
        auto test_t1 = float3(-0.4, -0.3, 0.5);
        auto test_t2 = float3(0.6, -0.3, 0.5);
        {
            Eigen::Vector<float, 12>          G;
            Eigen::Matrix<float, 12, 12>      H;
            {
                auto p = float3_to_eigen3(test_p);
                auto t0 = float3_to_eigen3(test_t0);
                auto t1 = float3_to_eigen3(test_t1);
                auto t2 = float3_to_eigen3(test_t2);
                Eigen::Vector4i flag = uipc::backend::cuda::distance::point_triangle_distance_flag(p, t0, t0, t2);
                
                if constexpr (false)
                {
                    float D;
                    uipc::backend::cuda::distance::point_triangle_distance2(flag, p, t0, t1, t2, D);
                    Eigen::Vector<float, 12> GradD;
                    uipc::backend::cuda::distance::point_triangle_distance2_gradient(flag, p, t0, t1, t2, GradD); // OK
                    float dBdD;
                    cipc::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness); // OK
                    G = dBdD * GradD; // OK
                    float ddBddD;
                    cipc::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness); // OK
                    Eigen::Matrix<float, 12, 12> HessD;
                    uipc::backend::cuda::distance::point_triangle_distance2_hessian(flag, p, t0, t1, t2, HessD);
                    // std::cout << "Test VF local value : ddBddD = " << ddBddD << ", HessD = \n" << HessD << std::endl;
                    // std::cout << "Test VF local value : ddBddD = " << ddBddD << " , GradD.transpose() = \n" << GradD.transpose() << std::endl ;
                    // std::cout << "Test VF local value : ddBddD * GradD.transpose() = \n" << ddBddD * GradD.transpose() << std::endl ;
                    // std::cout << "Test VF local value : ddBddD * GradD * GradD.transpose() = \n" << ddBddD * GradD * GradD.transpose() << std::endl ;
                    H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
                }

                uipc::backend::cuda::sym::codim_ipc_simplex_contact::PT_barrier_gradient_hessian(
                    G, H, flag, kappa, d_hat, thickness, 
                    p, 
                    t0, 
                    t1, 
                    t2);
                std::cout << "Test VF Barrier Flag =  " << flag << " Gradient/Hessian : "  << " G = \n" << G.transpose() << " \n H = \n" << H << std::endl;
                // H = spd_projection(H);
            }
        }
        {
            auto fn_test_dcd_vf = device.compile<1>([&](Float d_hat, Float thickness, Float kappa)
            {                
                
                Float3 p = test_p;
                Float3 t0 = test_t0;
                Float3 t1 = test_t1;
                Float3 t2 = test_t2;

                Float3 bary = distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
                uint3 valid_indices = makeUint3(0, 1, 2);
                uint valid_count = distance::point_triangle_type(bary, valid_indices);
                
                Float3 x = bary[0] * (t0 - p) +
                        bary[1] * (t1 - p) +
                        bary[2] * (t2 - p);
                Float d2 = length_squared_vec(x);
                $if (d2 < square_scalar(thickness + d_hat))
                {
                    Float d = sqrt_scalar(d2);
                    Float dBdD; Float ddBddD;
                    cipc::dKappaBarrierdD(dBdD, kappa, d2, d_hat, thickness);
                    cipc::ddKappaBarrierddD(ddBddD, kappa, d2, d_hat, thickness);

                    $if (valid_count == 3)
                    {
                        Var<CollisionPairVF> vf_pair;
                        vf_pair.vec1 = makeFloat4(x.x, x.y, x.z, d);
                        vf_pair.bary = bary;
                        {
                            Float12 G;
                            Float12 GradD;
                            DistanceGradient::point_triangle_distance2_gradient(p, t0, t1, t2, GradD); // GradiantD
                            mult_largevec_scalar(G, GradD, dBdD);                        

                            Float12x12 HessD;
                            DistanceGradient::point_triangle_distance2_hessian(p, t0, t1, t2, HessD); // HessianD

                            // device_log("Test VF local value : ddBddD = {} , H = ", ddBddD);
                            // print_largemat(H);
                            
                            // auto ggT = outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD);
                            // print_largevec(GradD);
                            // print_largevec(mult_largevec_scalar(GradD, ddBddD));
                            // device_log("ddBddD = {}", ddBddD);
                            // print_largemat(ggT);
                            
                            // H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
                            Float12x12 H = add_largemat(
                                outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                                mult_largemat_scalar(HessD, dBdD)
                            );
                            
                            vf_pair.gradient[0] = G.vec[0];
                            vf_pair.gradient[1] = G.vec[1];
                            vf_pair.gradient[2] = G.vec[2];
                            vf_pair.gradient[3] = G.vec[3];
                            //  0  1  2  3
                            //     4  5  6
                            //        7  8
                            //           9
                            CollisionPair::write_upper_hessian(vf_pair.hessian, H);
                            device_log("Test VF Barrier Valid count = {}", valid_count);
                            device_log("Test VF Barrier Gradient = ");
                            print_largevec(G);
                            device_log("Test VF Barrier Hessian = ");
                            print_largemat(H);
                        }
                    }
                    $else
                    {
                        device_log("Error Caulc VF Case");
                    };
                }
                $else
                {
                    device_log("VF Case Not Valid, d2 = {}, d_hat = {}, thickness = {}", d2, d_hat, thickness);
                };
            });
            
            stream << fn_test_dcd_vf(d_hat, thickness, kappa).dispatch(1) << synchronize(); 
        }
   
    }

    // VV Barrier Gradient/Hessian Test
    if constexpr (false)
    {
        luisa::log_info("////////////////////////////////");
        luisa::log_info("VV Barrier Gradient/Hessian Test");
        luisa::log_info("////////////////////////////////");

        const float kappa = 1e5;
        const float d_hat = 1e-2;
        const float thickness = 0.0f;

        auto test_p0 = float3(0.0, 0.0, 0.0);
        auto test_p1 = float3(5e-3, 5e-3, 0.0f);
        {
            Eigen::Vector<float, 6>         G;
            Eigen::Matrix<float, 6, 6>      H;
            {
                auto p0 = float3_to_eigen3(test_p0);
                auto p1 = float3_to_eigen3(test_p1);
                Eigen::Vector2i flag = uipc::backend::cuda::distance::point_point_distance_flag(p0, p1);

                if constexpr (false)
                {
                    float D;
                    uipc::backend::cuda::distance::point_point_distance2(flag, p0, p1, D);
                    Eigen::Vector<float, 6> GradD;
                    uipc::backend::cuda::distance::point_point_distance2_gradient(flag, p0, p1, GradD); // OK
                    float dBdD;
                    cipc::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness); // OK
                    G = dBdD * GradD; // OK
                    float ddBddD;
                    cipc::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness); // OK
                    Eigen::Matrix<float, 6, 6> HessD;
                    uipc::backend::cuda::distance::point_point_distance2_hessian(flag, p0, p1, HessD);
                    H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
                    std::cout << "Test VV local value : ddBddD = " << ddBddD << ", HessD = \n" << HessD << std::endl;
                    // std::cout << "Test VV local value : ddBddD = " << ddBddD << " , GradD.transpose() = \n" << GradD.transpose() << std::endl ;
                    // std::cout << "Test VV local value : ddBddD * GradD.transpose() = \n" << ddBddD * GradD.transpose() << std::endl ;
                    // std::cout << "Test VV local value : ddBddD * GradD * GradD.transpose() = \n" << ddBddD * GradD * GradD.transpose() << std::endl ;
                }

                uipc::backend::cuda::sym::codim_ipc_simplex_contact::PP_barrier_gradient_hessian(
                    G, H, flag, kappa, d_hat, thickness, 
                    p0, p1);
                std::cout << "Test VV Barrier Flag =  " << flag << " Gradient/Hessian : "  << " G = \n" << G.transpose() << " \n H = \n" << H << std::endl;
                // H = spd_projection(H);
            }
        }
        {
            auto fn_test_dcd_vv = device.compile<1>([&](Float d_hat, Float thickness, Float kappa)
            {                
                Float3 p0 = test_p0;
                Float3 p1 = test_p1;

                Float d2 = distance::point_point_distance_squared_unclassified(p0, p1);

                $if (d2 < square_scalar(thickness + d_hat))
                {
                    Float dBdD; Float ddBddD;
                    cipc::dKappaBarrierdD(dBdD, kappa, d2, d_hat, thickness);
                    cipc::ddKappaBarrierddD(ddBddD, kappa, d2, d_hat, thickness);

                    Var<CollisionPairVV> vv_pair;
                    {
                        Float6 G;
                        Float6 GradD;
                        DistanceGradient::point_point_distance2_gradient(p0, p1, GradD); // GradiantD
                        mult_largevec_scalar(G, GradD, dBdD);                        

                        Float6x6 HessD;
                        DistanceGradient::point_point_distance2_hessian(p0, p1, HessD); // HessianD

                        device_log("Test VV Barrier HessD = ");
                        print_largemat(HessD);

                        Float6x6 H = add_largemat(
                            outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                            mult_largemat_scalar(HessD, dBdD)
                        );
                        
                        vv_pair.gradient[0] = G.vec[0];
                        vv_pair.gradient[1] = G.vec[1];
                        CollisionPair::write_upper_hessian(vv_pair.hessian, H);
                        device_log("Test VV Barrier Gradient = ");
                        print_largevec(G);
                        device_log("Test VV Barrier Hessian = ");
                        print_largemat(H);
                    }
                }
                $else
                {
                    device_log("VV Case Not Valid, d2 = {}, d_hat = {}, thickness = {}", d2, d_hat, thickness);
                };
            });
            stream << fn_test_dcd_vv(d_hat, thickness, kappa).dispatch(1) << synchronize(); 
        }
    }

    // VE Barrier Gradient/Hessian Test
    if constexpr (false)
    {
        luisa::log_info("////////////////////////////////");
        luisa::log_info("VE Barrier Gradient/Hessian Test");
        luisa::log_info("////////////////////////////////");

        const float kappa = 1e5;
        const float d_hat = 1e-2;
        const float thickness = 0.0f;

        auto test_p = float3(0.0, 0.0, 0.0);
        auto test_e0 = float3(-5e-3, 5e-3, 0.0f);
        auto test_e1 = float3(5e-3, 5e-3, 0.0f);
        {
            Eigen::Vector<float, 9>          G;
            Eigen::Matrix<float, 9, 9>      H;
            {
                auto p =  float3_to_eigen3(test_p );
                auto e0 = float3_to_eigen3(test_e0);
                auto e1 = float3_to_eigen3(test_e1);
                Eigen::Vector3i flag = uipc::backend::cuda::distance::point_edge_distance_flag(p, e0, e1);

                uipc::backend::cuda::sym::codim_ipc_simplex_contact::PE_barrier_gradient_hessian(
                    G, H, flag, kappa, d_hat, thickness, 
                    p, e0, e1);
                std::cout << "Test VE Barrier Flag =  " << flag << " Gradient/Hessian : "  << " G = \n" << G.transpose() << " \n H = \n" << H << std::endl;
                // H = spd_projection(H);
            }
        }
        {
            auto fn_test_dcd_ve = device.compile<1>([&](Float d_hat, Float thickness, Float kappa)
            {                
                Float3 p  = test_p ;
                Float3 e0 = test_e0;
                Float3 e1 = test_e1;

                Float d2 = distance::point_edge_distance_squared_unclassified(p, e0, e1);

                $if (d2 < square_scalar(thickness + d_hat))
                {
                    Float dBdD; Float ddBddD;
                    cipc::dKappaBarrierdD(dBdD, kappa, d2, d_hat, thickness);
                    cipc::ddKappaBarrierddD(ddBddD, kappa, d2, d_hat, thickness);

                    Var<CollisionPairVE> vv_pair;
                    {
                        Float9 G;
                        Float9 GradD;
                        DistanceGradient::point_edge_distance2_gradient(p, e0, e1, GradD); // GradiantD
                        mult_largevec_scalar(G, GradD, dBdD);                        

                        Float9x9 HessD;
                        DistanceGradient::point_edge_distance2_hessian(p, e0, e1, HessD); // HessianD

                        Float9x9 H = add_largemat(
                            outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                            mult_largemat_scalar(HessD, dBdD)
                        );
                        
                        vv_pair.gradient[0] = G.vec[0];
                        vv_pair.gradient[1] = G.vec[1];
                        vv_pair.gradient[2] = G.vec[2];
                        CollisionPair::write_upper_hessian(vv_pair.hessian, H);
                        device_log("Test VE Barrier Gradient = ");
                        print_largevec(G);
                        device_log("Test VE Barrier Hessian = ");
                        print_largemat(H);
                    }
                }
                $else
                {
                    device_log("VE Case Not Valid, d2 = {}, d_hat = {}, thickness = {}", d2, d_hat, thickness);
                };
            });
            stream << fn_test_dcd_ve(d_hat, thickness, kappa).dispatch(1) << synchronize(); 
        }
    }

    // EE Barrier Gradient/Hessian Test
    if constexpr (false)
    {
        luisa::log_info("////////////////////////////////");
        luisa::log_info("EE Barrier Gradient/Hessian Test");
        luisa::log_info("////////////////////////////////");
        const float kappa = 1e5;
        const float d_hat = 1e-2;
        const float thickness = 0.0f;

        // auto test_ea_p0 = float3(-0.50001013, -0.2954696, 0.4555479);
        // auto test_ea_p1 = float3(0.49999505, -0.29309484, 0.45634925);
        // auto test_eb_p0 = float3(-0.4, -0.3, -0.5);
        // auto test_eb_p1 = float3(-0.4, -0.3, 0.5);
        // auto test_t0_Ea0 = float3(-0.5, 0, 0.5);
        // auto test_t0_Ea1 = float3(0.5, 0, 0.5);
        // auto test_t0_Eb0 = float3(-0.4, -0.3, -0.5);
        // auto test_t0_Eb1 = float3(-0.4, -0.3, 0.5);
        // auto test_ea_p0 = float3(-0.0261442, -0.00669399, 0.343152);
        // auto test_ea_p1 = float3(-0.0149436, -0.00669399, 0.343419);
        // auto test_eb_p0 = float3(-0.0228759, -0.00669399, 0.333348);
        // auto test_eb_p1 = float3(-0.0340765, -0.00669399, 0.333081);

        auto test_ea_p0 = float3(0.444727, -0.00669399, 0.303925);
        auto test_ea_p1 = float3(0.45453, -0.00669399, 0.303925);
        auto test_eb_p0 = float3(0.44745, -0.00669399, 0.294121);
        auto test_eb_p1 = float3(0.437646, -0.00669399, 0.294121);

        luisa::compute::Buffer<float3> edge_positions = device.create_buffer<float3>(4);
        std::vector<float3> tmp_positions = {
            test_ea_p0,
            test_ea_p1,
            test_eb_p0,
            test_eb_p1,
        }; stream << edge_positions.copy_from(tmp_positions.data()) << synchronize();

        if (false)
        {
            Eigen::Vector<float, 12>          G;
            Eigen::Matrix<float, 12, 12>      H;
            {
                auto ea_p0 = float3_to_eigen3(test_ea_p0);
                auto ea_p1 = float3_to_eigen3(test_ea_p1);
                auto eb_p0 = float3_to_eigen3(test_eb_p0);
                auto eb_p1 = float3_to_eigen3(test_eb_p1);
                // auto t0_Ea0 = float3_to_eigen3(test_t0_Ea0);
                // auto t0_Ea1 = float3_to_eigen3(test_t0_Ea1);
                // auto t0_Eb0 = float3_to_eigen3(test_t0_Eb0);
                // auto t0_Eb1 = float3_to_eigen3(test_t0_Eb1);

                Eigen::Vector4i flag = uipc::backend::cuda::distance::edge_edge_distance_flag(ea_p0, ea_p1, eb_p0, eb_p1);
                {
                    float D;
                    uipc::backend::cuda::distance::edge_edge_distance2(flag, ea_p0, ea_p1, eb_p0, eb_p1, D);
                    Eigen::Vector<float, 12> GradD;
                    uipc::backend::cuda::distance::edge_edge_distance2_gradient(flag, ea_p0, ea_p1, eb_p0, eb_p1, GradD); // OK
                    float dBdD;
                    cipc::dKappaBarrierdD(dBdD, kappa, D, d_hat, thickness);
                    G = dBdD * GradD;
                    float ddBddD;
                    cipc::ddKappaBarrierddD(ddBddD, kappa, D, d_hat, thickness);
                    Eigen::Matrix<float, 12, 12> HessD;
                    uipc::backend::cuda::distance::edge_edge_distance2_hessian(flag, ea_p0, ea_p1, eb_p0, eb_p1, HessD);
                    H = ddBddD * GradD * GradD.transpose() + dBdD * HessD;
                }
                // uipc::backend::cuda::sym::codim_ipc_simplex_contact::mollified_EE_barrier_gradient_hessian(
                //     G, H, flag, kappa, d_hat, thickness, 
                //     t0_Ea0, 
                //     t0_Ea1, 
                //     t0_Eb0, 
                //     t0_Eb1, 
                //     ea_p0, 
                //     ea_p1, 
                //     eb_p0, 
                //     eb_p1);
                
                // luisa::log_info("Get EE Pair : indices = {}, bary = {}, d = {}", 
                //     ee_pair.indices, 
                //     ee_pair.bary, d);
                
                std::cout << "Test EE Barrier Flag =  " << flag << " Gradient/Hessian : "  << " G = \n" << G.transpose() << " \n H = \n" << H << std::endl;
                // H = spd_projection(H);
            }
        }
        {
            auto fn_test_dcd_ee = device.compile<1>([&](Float d_hat, Float thickness, Float kappa)
            {                
                // Get from buffer
                // Float3 ea_p0 = edge_positions->read(0);
                // Float3 ea_p1 = edge_positions->read(1);
                // Float3 eb_p0 = edge_positions->read(2);
                // Float3 eb_p1 = edge_positions->read(3);

                // Get from host
                Float3 ea_p0 = test_ea_p0;
                Float3 ea_p1 = test_ea_p1;
                Float3 eb_p0 = test_eb_p0;
                Float3 eb_p1 = test_eb_p1;

                Float4 bary = distance::edge_edge_distance_coeff_unclassified(ea_p0, ea_p1, eb_p0, eb_p1);
                Bool is_ee = all_vec(bary != 0.0f);
                
                // Float3 t0_Ea0 = test_t0_Ea0;
                // Float3 t0_Ea1 = test_t0_Ea1;
                // Float3 t0_Eb0 = test_t0_Eb0;
                // Float3 t0_Eb1 = test_t0_Eb1;
                
                luisa::uint2 valid_indices1; luisa::uint2 valid_indices2;
                auto valid_count = distance::edge_edge_type(bary, valid_indices1, valid_indices2);

                Float3 x0 = bary[0] * ea_p0 + bary[1] * ea_p1;
                Float3 x1 = bary[2] * eb_p0 + bary[3] * eb_p1;
                Float3 x = x1 - x0;
                Float d2 = length_squared_vec(x);

                device_log("Test EE bary = {} d = {}", bary, sqrt_scalar(d2));
                
                $if (d2 < square_scalar(d_hat + thickness) & is_ee)
                {
                    Float d = sqrt_scalar(d2);
                    Float dBdD; Float ddBddD;
                    cipc::dKappaBarrierdD(dBdD, kappa, d2, d_hat, thickness);
                    cipc::ddKappaBarrierddD(ddBddD, kappa, d2, d_hat, thickness);

                    $if (is_ee)
                    {
                        Var<CollisionPairEE> ee_pair;
                        ee_pair.vec1 = makeFloat4(x.x, x.y, x.z, d);
                        ee_pair.bary = bary;
                        {
                            Float12 GradD;
                            Float12 G; 
                            DistanceGradient::edge_edge_distance2_gradient(ea_p0, ea_p1, eb_p0, eb_p1, GradD); // GradiantD
                            mult_largevec_scalar(G, GradD, dBdD);                        

                            Float12x12 H;
                            DistanceGradient::edge_edge_distance2_hessian(ea_p0, ea_p1, eb_p0, eb_p1, H); // HessianD
                            H = add_largemat(
                                outer_product_largevec(mult_largevec_scalar(GradD, ddBddD), GradD),
                                mult_largemat_scalar(H, dBdD)
                            );

                            ee_pair.gradient[0] = G.vec[0];
                            ee_pair.gradient[1] = G.vec[1];
                            ee_pair.gradient[2] = G.vec[2];
                            ee_pair.gradient[3] = G.vec[3];
                            //  0  1  2  3
                            //     4  5  6
                            //        7  8
                            //           9
                            CollisionPair::write_upper_hessian(ee_pair.hessian, H);
                            device_log("Test EE Barrier Valid count = {}", valid_count);
                            device_log("Test EE Barrier Gradient = ");
                            print_largevec(G);
                            device_log("Test EE Barrier Hessian = ");
                            print_largemat(H);
                        }
                    }
                    $else
                    {
                        device_log("Error Caulc EE Case");
                    };
                }
                $else
                {
                    device_log("EE Case Not Valid, d2 = {}, d_hat = {}, thickness = {}", d2, d_hat, thickness);
                };
            });
            
            stream << fn_test_dcd_ee(d_hat, thickness, kappa).dispatch(1) << synchronize(); 
        }
   
    }


}

}
