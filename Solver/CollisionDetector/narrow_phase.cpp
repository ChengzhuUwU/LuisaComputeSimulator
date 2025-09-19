#include "CollisionDetector/narrow_phase.h"
#include "CollisionDetector/accd.hpp"
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/libuipc/codim_ipc_simplex_normal_contact_function.h"
#include "CollisionDetector/libuipc/distance/distance_flagged.h"
#include "Core/lc_to_eigen.h"
#include "SimulationCore/scene_params.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "luisa/core/basic_types.h"
#include <Eigen/Dense>
#include <iostream>

namespace lcs // Data IO
{

void NarrowPhasesDetector::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    ContactEnergyType contact_energy_type = get_scene_params().contact_energy_type == 0 ? ContactEnergyType::Quadratic : ContactEnergyType::Barrier; // Quadratic or Barrier

    compile_ccd(device);
    compile_dcd(device, contact_energy_type);
    compile_energy(device, contact_energy_type);
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
float NarrowPhasesDetector::download_energy(Stream& stream)
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
    if (num_vf_broadphase > collision_data->broad_phase_list_vf.size() / 2) {
        LUISA_ERROR("BroadPhase VF outof range : {} ({})", num_vf_broadphase, collision_data->broad_phase_list_vf.size() / 2);
    }
    if (num_ee_broadphase > collision_data->broad_phase_list_ee.size() / 2) {
        LUISA_ERROR("BroadPhase EE outof range : {} ({})", num_ee_broadphase, collision_data->broad_phase_list_ee.size() / 2);
    }

    // LUISA_INFO("num_vf_broadphase = {}", num_vf_broadphase); // TODO: Indirect Dispatch
    // LUISA_INFO("num_ee_broadphase = {}", num_ee_broadphase); // TODO: Indirect Dispatch
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

    // LUISA_INFO("       num_vv = {}, num_ve = {}, num_vf = {}, num_ee = {}", num_vv, num_ve, num_vf, num_ee);

    // if (num_vv != 0) stream << collision_data->narrow_phase_list_vv.copy_to(host_collision_data->narrow_phase_list_vv.data());
    // if (num_ve != 0) stream << collision_data->narrow_phase_list_ve.copy_to(host_collision_data->narrow_phase_list_ve.data());
    // if (num_vf != 0) stream << collision_data->narrow_phase_list_vf.copy_to(host_collision_data->narrow_phase_list_vf.data());
    // if (num_ee != 0) stream << collision_data->narrow_phase_list_ee.copy_to(host_collision_data->narrow_phase_list_ee.data());
    if (num_vv != 0)
        stream << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_to(host_collision_data->narrow_phase_list_vv.data());
    if (num_ve != 0)
        stream << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_to(host_collision_data->narrow_phase_list_ve.data());
    if (num_vf != 0)
        stream << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_to(host_collision_data->narrow_phase_list_vf.data());
    if (num_ee != 0)
        stream << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_to(host_collision_data->narrow_phase_list_ee.data());
    stream << luisa::compute::synchronize();

    // Why this can not run ???
    // stream
    //         << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_to(host_collision_data->narrow_phase_list_vv.data())
    //         << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_to(host_collision_data->narrow_phase_list_ve.data())
    //         << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_to(host_collision_data->narrow_phase_list_vf.data())
    //         << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_to(host_collision_data->narrow_phase_list_ee.data())
    //         << luisa::compute::synchronize();

    // LUISA_INFO("Complete Download");
}
void NarrowPhasesDetector::upload_spd_narrowphase_list(Stream& stream)
{
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vv = host_count[collision_data->get_vv_count_offset()];
    const uint num_ve = host_count[collision_data->get_ve_count_offset()];
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    // LUISA_INFO("       num_vv = {}, num_ve = {}, num_vf = {}, num_ee = {}", num_vv, num_ve, num_vf, num_ee);

    if (num_vv != 0)
        stream << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_from(host_collision_data->narrow_phase_list_vv.data());
    if (num_ve != 0)
        stream << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_from(host_collision_data->narrow_phase_list_ve.data());
    if (num_vf != 0)
        stream << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_from(host_collision_data->narrow_phase_list_vf.data());
    if (num_ee != 0)
        stream << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_from(host_collision_data->narrow_phase_list_ee.data());

    // stream
    //     << collision_data->narrow_phase_list_vv.view(0, num_vv).copy_from(host_collision_data->narrow_phase_list_vv.data())
    //     << collision_data->narrow_phase_list_ve.view(0, num_ve).copy_from(host_collision_data->narrow_phase_list_ve.data())
    //     << collision_data->narrow_phase_list_vf.view(0, num_vf).copy_from(host_collision_data->narrow_phase_list_vf.data())
    //     << collision_data->narrow_phase_list_ee.view(0, num_ee).copy_from(host_collision_data->narrow_phase_list_ee.data())
    // ;

    // LUISA_INFO("Complete Download");
}
float NarrowPhasesDetector::get_global_toi(Stream& stream)
{
    stream << luisa::compute::synchronize();

    auto& host_toi = host_collision_data->toi_per_vert[0];
    // if (host_toi != host_accd::line_search_max_t) LUISA_INFO("             CCD linesearch toi = {}", host_toi);
    host_toi /= host_accd::line_search_max_t;
    if (host_toi < 1e-7) {
        LUISA_ERROR("  small toi : {}", host_toi);
    }
    return host_toi;
}

} // namespace lcs // Data IO

namespace lcs // CCD
{

void NarrowPhasesDetector::compile_ccd(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    fn_reset_toi = device.compile<1>([](Var<BufferView<float>> sa_toi) {
        sa_toi->write(dispatch_x(), accd::line_search_max_t);
    });
    fn_reset_uint = device.compile<1>([](Var<BufferView<uint>> sa_toi) {
        sa_toi->write(dispatch_x(), 0u);
    });
    fn_reset_float = device.compile<1>([](Var<BufferView<float>> sa_toi) {
        sa_toi->write(dispatch_x(), 0.0f);
    });
    fn_reset_energy = device.compile<1>([](Var<BufferView<float>> sa_energy) {
        sa_energy->write(dispatch_x(), 0.0f);
    });

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    fn_narrow_phase_vf_ccd_query = device.compile<1>(
        [sa_toi = collision_data->toi_per_vert.view(),
            broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
            broadphase_list = collision_data->broad_phase_list_vf.view()](
            Var<BufferView<float3>> sa_x_begin_left,
            Var<BufferView<float3>> sa_x_begin_right,
            Var<BufferView<float3>> sa_x_end_left,
            Var<BufferView<float3>> sa_x_end_right,
            Var<BufferView<uint3>> sa_faces_right,
            Float d_hat, // Not relavent to d_hat
            Float thickness) {
            const Uint pair_idx = dispatch_x();
            const Uint vid = broadphase_list->read(2 * pair_idx + 0);
            const Uint fid = broadphase_list->read(2 * pair_idx + 1);

            const Uint3 face = sa_faces_right.read(fid);

            Float toi = accd::line_search_max_t;
            $if(
                vid == face[0] | vid == face[1] | vid == face[2])
            {
                toi = accd::line_search_max_t;
            }
            $else
            {
                Float3 t0_p = sa_x_begin_left->read(vid);
                Float3 t1_p = sa_x_end_left->read(vid);
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

            $if(pair_idx % 256 == 0)
            {
                sa_toi->atomic(0).fetch_min(toi);
            };
        });

    fn_narrow_phase_ee_ccd_query = device.compile<1>(
        [sa_toi = collision_data->toi_per_vert.view(),
            broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
            broadphase_list = collision_data->broad_phase_list_ee.view()](
            Var<BufferView<float3>> sa_x_begin_a,
            Var<BufferView<float3>> sa_x_begin_b,
            Var<BufferView<float3>> sa_x_end_a,
            Var<BufferView<float3>> sa_x_end_b,
            Var<BufferView<uint2>> sa_edges_left,
            Var<BufferView<uint2>> sa_edges_right,
            Float d_hat, // Not relavent to d_hat
            Float thickness) {
            const Uint pair_idx = dispatch_x();
            const Uint left = broadphase_list->read(2 * pair_idx + 0);
            const Uint right = broadphase_list->read(2 * pair_idx + 1);
            const Uint2 left_edge = sa_edges_left.read(left);
            const Uint2 right_edge = sa_edges_right.read(right);

            Float toi = accd::line_search_max_t;
            $if(
                left_edge[0] == right_edge[0] | left_edge[0] == right_edge[1] | left_edge[1] == right_edge[0] | left_edge[1] == right_edge[1])
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
                // device_log("EE CCD : left = {}, edge1 = {}, right = {}, edge2 = {}, TOI = {}, ea_t0_p0 = {}, ea_t0_p1 = {}, eb_t0_p0 = {}, eb_t0_p1 = {}, ea_t1_p0 = {}, ea_t1_p1 = {}, eb_t1_p0 = {}, eb_t1_p1 = {}",
                //     left, left_edge, right, right_edge, toi,
                //     ea_t0_p0, ea_t0_p1, eb_t0_p0, eb_t0_p1, ea_t1_p0, ea_t1_p1 , eb_t1_p0, eb_t1_p1
                // );
            };

            // $if (toi != host_accd::line_search_max_t)
            // {
            //     device_log("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
            // };

            toi = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, toi, ParallelIntrinsic::warp_reduce_op_min<float>);

            $if(pair_idx % 256 == 0)
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

    if (num_vf_broadphase != 0) {
        stream << fn_narrow_phase_vf_ccd_query(
            sa_x_begin_left,
            sa_x_begin_right, // sa_x_begin_right
            sa_x_end_left,
            sa_x_end_right, // sa_x_end_right
            sa_faces_right, d_hat, thickness)
                      .dispatch(num_vf_broadphase);
    }
    stream << sa_toi.view(0, 1).copy_to(host_toi.data());
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

    // LUISA_INFO("curr toi = {} from VF", host_toi[0]);

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

    if (num_ee_broadphase != 0) {
        stream << fn_narrow_phase_ee_ccd_query(
            sa_x_begin_a,
            sa_x_begin_b,
            sa_x_end_a,
            sa_x_end_b,
            sa_edges_left,
            sa_edges_left, d_hat, thickness)
                      .dispatch(num_ee_broadphase);
    }
    stream << sa_toi.view(0, 1).copy_to(host_toi.data());
}

} // namespace lcs // CCD

namespace lcs // DCD
{

// constexpr float stiffness_repulsion = 1e9;
constexpr bool use_area_weighting = true;
constexpr float rest_distance_culling_rate = 1.0f;

void NarrowPhasesDetector::compile_dcd(luisa::compute::Device& device, const ContactEnergyType contact_energy_type)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    fn_narrow_phase_vf_dcd_query = device.compile<1>(
        [broadphase_count = collision_data->broad_phase_collision_count.view(offset_vf, 1),
            broadphase_list = collision_data->broad_phase_list_vf.view(),
            narrowphase_count_vv = collision_data->narrow_phase_collision_count.view(offset_vv, 1),
            narrowphase_count_ve = collision_data->narrow_phase_collision_count.view(offset_ve, 1),
            narrowphase_count_vf = collision_data->narrow_phase_collision_count.view(offset_vf, 1),
            narrowphase_list_vv = collision_data->narrow_phase_list_vv.view(),
            narrowphase_list_ve = collision_data->narrow_phase_list_ve.view(),
            narrowphase_list_vf = collision_data->narrow_phase_list_vf.view(), contact_energy_type](
            Var<BufferView<float3>> sa_x_left,
            Var<BufferView<float3>> sa_x_right,
            Var<BufferView<float3>> sa_rest_x_a,
            Var<BufferView<float3>> sa_rest_x_b,
            Var<BufferView<float>> sa_rest_area_a,
            Var<BufferView<float>> sa_rest_area_b,
            Var<BufferView<uint3>> sa_faces_right,
            Float d_hat,
            Float thickness,
            Float kappa) {
            const Uint pair_idx = dispatch_x();
            const Uint vid = broadphase_list->read(2 * pair_idx + 0);
            const Uint fid = broadphase_list->read(2 * pair_idx + 1);
            const Uint3 face = sa_faces_right.read(fid);

            $if(
                vid == face[0] | vid == face[1] | vid == face[2])
            {
            }
            $else
            {
                Float3 p = sa_x_left->read(vid);
                Float3 face_positions[3] = {
                    sa_x_right->read(face[0]),
                    sa_x_right->read(face[1]),
                    sa_x_right->read(face[2]),
                };
                Float3& t0 = face_positions[0];
                Float3& t1 = face_positions[1];
                Float3& t2 = face_positions[2];

                Float3 bary = distance::point_triangle_distance_coeff_unclassified(p, t0, t1, t2);

                Float3 x = bary[0] * (p - t0) + bary[1] * (p - t1) + bary[2] * (p - t2);
                Float d2 = length_squared_vec(x);
                // luisa::compute::device_log("VF pair {}-{} : d = {}", vid, face, sqrt_scalar(d2));
                $if(
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
                        rest_t2);
                    $if(rest_d2 > rest_distance_culling_rate * square_scalar(thickness + d_hat))
                    {
                        Float d = sqrt_scalar(d2);
                        Float3 normal = x / d;

                        Float k1;
                        Float k2;
                        Float avg_area = 1.0f;

                        if constexpr (use_area_weighting) {
                            Float area_a = sa_rest_area_a->read(vid);
                            Float area_b = sa_rest_area_b->read(fid);
                            avg_area = 0.5f * (area_a + area_b);
                            // luisa::compute::device_log("VF pair: with diff = {}, normal = {}, d = {}, proj = {}, C = {}, stiff = {} (area = {}) bary = {}", x, normal, d, dot_vec(normal, x), C, stiff, avg_area, bary);
                        }

                        if (contact_energy_type == ContactEnergyType::Quadratic) {
                            Float C = thickness + d_hat - d;
                            Float stiff = avg_area * kappa;
                            k1 = stiff * C;
                            k2 = stiff;
                        } else if (contact_energy_type == ContactEnergyType::Barrier) {
                            Float dBdD;
                            Float ddBddD;
                            // dBdD = kappa * ipc::barrier_first_derivative(d2 - square_scalar(thickness), square_scalar(d_hat));
                            // ddBddD = kappa * ipc::barrier_second_derivative(d2 - square_scalar(thickness), square_scalar(d_hat));
                            cipc::dKappaBarrierdD(dBdD, avg_area * kappa, d2, d_hat, thickness);
                            cipc::ddKappaBarrierddD(ddBddD, avg_area * kappa, d2, d_hat, thickness);
                            k1 = dBdD;
                            k2 = ddBddD;
                        }

                        {
                            Uint idx = narrowphase_count_vf->atomic(0).fetch_add(1u);
                            Var<CollisionPairVF> vf_pair;
                            vf_pair.indices = make_uint4(vid, face[0], face[1], face[2]);
                            vf_pair.vec1 = make_float4(normal.x, normal.y, normal.z, avg_area);
                            CollisionPair::write_vf_weight(vf_pair, bary);
                            CollisionPair::write_vf_stiff(vf_pair, k1, k2);
                            ;
                            narrowphase_list_vf->write(idx, vf_pair);
                        }
                    };
                };
            };
        });

    fn_narrow_phase_ee_dcd_query = device.compile<1>(
        [broadphase_count = collision_data->broad_phase_collision_count.view(offset_ee, 1),
            broadphase_list = collision_data->broad_phase_list_ee.view(),
            narrowphase_count_ee = collision_data->narrow_phase_collision_count.view(offset_ee, 1),
            narrowphase_list_ee = collision_data->narrow_phase_list_ee.view(), contact_energy_type](
            Var<BufferView<float3>> sa_x_a,
            Var<BufferView<float3>> sa_x_b,
            Var<BufferView<float3>> sa_rest_x_a,
            Var<BufferView<float3>> sa_rest_x_b,
            Var<BufferView<float>> sa_rest_area_a,
            Var<BufferView<float>> sa_rest_area_b,
            Var<BufferView<uint2>> sa_edges_left,
            Var<BufferView<uint2>> sa_edges_right,
            Float d_hat,
            Float thickness,
            Float kappa) {
            const Uint pair_idx = dispatch_x();
            const Uint left = broadphase_list->read(2 * pair_idx + 0);
            const Uint right = broadphase_list->read(2 * pair_idx + 1);
            const Uint2 left_edge = sa_edges_left.read(left);
            const Uint2 right_edge = sa_edges_right.read(right);
            $if(
                left_edge[0] == right_edge[0] | left_edge[0] == right_edge[1] | left_edge[1] == right_edge[0] | left_edge[1] == right_edge[1])
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

                $if(
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
                        rest_eb_p1);
                    $if(rest_d2 > rest_distance_culling_rate * square_scalar(thickness + d_hat))
                    {
                        Float d = sqrt_scalar(d2);
                        Float C = thickness + d_hat - d;
                        // Float3 normal = normalize_vec(x);
                        Float3 normal = x / d;

                        Float k1;
                        Float k2;
                        Float avg_area = 1.0f;
                        if constexpr (use_area_weighting) {
                            Float area_a = sa_rest_area_a->read(left);
                            Float area_b = sa_rest_area_b->read(right);
                            avg_area = 0.5f * (area_a + area_b);
                            // luisa::compute::device_log("EE pair: with diff = {}, normal = {}, d = {}, proj = {}, C = {}, stiff = {} (area = {}) bary = {}", x, normal, d, dot_vec(normal, x), C, stiff, avg_area, bary);
                        }

                        if (contact_energy_type == ContactEnergyType::Quadratic) {
                            Float stiff = kappa * avg_area;
                            k1 = stiff * C;
                            k2 = stiff;
                        } else if (contact_energy_type == ContactEnergyType::Barrier) {
                            Float dBdD;
                            Float ddBddD;
                            cipc::dKappaBarrierdD(dBdD, avg_area * kappa, d2, d_hat, thickness);
                            cipc::ddKappaBarrierddD(ddBddD, avg_area * kappa, d2, d_hat, thickness);
                            k1 = dBdD;
                            k2 = ddBddD;
                        }

                        {
                            Uint idx = narrowphase_count_ee->atomic(0).fetch_add(1u);
                            Var<CollisionPairEE> ee_pair;
                            ee_pair.indices = make_uint4(left_edge[0], left_edge[1], right_edge[0], right_edge[1]);
                            ee_pair.vec1 = make_float4(normal.x, normal.y, normal.z, avg_area);
                            CollisionPair::write_ee_weight(ee_pair, bary);
                            CollisionPair::write_ee_stiff(ee_pair, k1, k2);
                            narrowphase_list_ee->write(idx, ee_pair);
                        }
                    };
                };
                // Corner case (VV, VE) will only be considered in VF detection
            };
        });
}

// Device DCD
void NarrowPhasesDetector::vf_dcd_query_repulsion(Stream& stream,
    const Buffer<float3>& sa_x_left,
    const Buffer<float3>& sa_x_right,
    const Buffer<float3>& sa_rest_x_left,
    const Buffer<float3>& sa_rest_x_right,
    const Buffer<float>& sa_rest_area_left,
    const Buffer<float>& sa_rest_area_right,
    const Buffer<uint3>& sa_faces_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto& host_count = host_collision_data->broad_phase_collision_count;
    const uint num_vf_broadphase = host_count[collision_data->get_vf_count_offset()];
    if (num_vf_broadphase != 0) {
        stream << fn_narrow_phase_vf_dcd_query(sa_x_left, sa_x_right, sa_rest_x_left, sa_rest_x_right, sa_rest_area_left, sa_rest_area_right, sa_faces_right, d_hat, thickness, kappa).dispatch(num_vf_broadphase);
    }
}
void NarrowPhasesDetector::ee_dcd_query_repulsion(Stream& stream,
    const Buffer<float3>& sa_x_left,
    const Buffer<float3>& sa_x_right,
    const Buffer<float3>& sa_rest_x_left,
    const Buffer<float3>& sa_rest_x_right,
    const Buffer<float>& sa_rest_area_left,
    const Buffer<float>& sa_rest_area_right,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto& host_count = host_collision_data->broad_phase_collision_count;
    const uint num_ee_broadphase = host_count[collision_data->get_ee_count_offset()];

    if (num_ee_broadphase != 0) {
        stream << fn_narrow_phase_ee_dcd_query(sa_x_left, sa_x_right, sa_rest_x_left, sa_rest_x_right, sa_rest_area_left, sa_rest_area_right, sa_edges_left, sa_edges_right, d_hat, thickness, kappa).dispatch(num_ee_broadphase);
    }
}

} // namespace lcs

namespace lcs // Compute Barrier Gradient & Hessian & Assemble
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
                                 Var<BufferView<float3>>& sa_cgB, const Uint& idx, const Float3& vec) {
        sa_cgB.atomic(idx)[0].fetch_add(vec[0]);
        sa_cgB.atomic(idx)[1].fetch_add(vec[1]);
        sa_cgB.atomic(idx)[2].fetch_add(vec[2]);
    };
    auto atomic_sub_float3 = [](
                                 Var<BufferView<float3>>& sa_cgB, const Uint& idx, const Float3& vec) {
        sa_cgB.atomic(idx)[0].fetch_sub(vec[0]);
        sa_cgB.atomic(idx)[1].fetch_sub(vec[1]);
        sa_cgB.atomic(idx)[2].fetch_sub(vec[2]);
    };
    auto atomic_add_float3x3 = [](
                                   Var<BufferView<float3x3>>& sa_cgA_diag, const Uint& idx, const Float3x3& mat) {
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

    // Spring-form contact energy
    fn_assemble_repulsion_hessian_gradient_vf = device.compile<1>(
        [narrowphase_list_vf = collision_data->narrow_phase_list_vf.view(), &atomic_add_float3, &atomic_add_float3x3](
            Var<BufferView<float3>> sa_x_left,
            Var<BufferView<float3>> sa_x_right,
            Float d_hat,
            Float thickness,
            Var<BufferView<float3>> sa_cgB,
            Var<BufferView<float3x3>> sa_cgA_diag) {
            const Uint pair_idx = dispatch_x();
            const auto& pair = narrowphase_list_vf->read(pair_idx);
            const auto indices = CollisionPair::get_indices(pair);

            const Float3 normal = CollisionPair::get_direction(pair);
            const Float4 weight = CollisionPair::get_vf_weight(pair);

            // const Float3 diff =
            //     weight[0] * sa_x_left.read(indices[0]) +
            //     weight[1] * sa_x_right.read(indices[1]) +
            //     weight[2] * sa_x_right.read(indices[2]) +
            //     weight[3] * sa_x_right.read(indices[3]);
            // const Float d = length_vec(diff);

            // const Float stiff = CollisionPair::get_stiff(pair);
            const Float2 stiff = CollisionPair::get_vf_stiff(pair);
            const Float k1 = stiff[0];
            const Float k2 = stiff[1];

            // const Float d = dot_vec(diff, normal);
            // $if (d < d_hat + thickness)
            {
                // const Float C = thickness + d_hat - d;
                const Float3x3 nnT = outer_product(normal, normal);

                for (uint j = 0; j < 4; j++) {
                    Float3 force;
                    Float3x3 hessian;
                    force = k1 * weight[j] * normal;
                    hessian = k2 * weight[j] * weight[j] * nnT;

                    // device_log("VF pair {} on vert {} : force = {}, diff = {}, stiff = {}, weight = {}, d = {}/{}", pair_idx, indices[j], force, diff, stiff, weight, d, dot_vec(diff, normal));
                    atomic_add_float3x3(sa_cgA_diag, indices[j], hessian);
                    atomic_add_float3(sa_cgB, indices[j], force);
                }
            };
        });

    fn_assemble_repulsion_hessian_gradient_ee = device.compile<1>(
        [narrowphase_list_ee = collision_data->narrow_phase_list_ee.view(), &atomic_add_float3, &atomic_add_float3x3](
            Var<BufferView<float3>> sa_x_left,
            Var<BufferView<float3>> sa_x_right,
            Float d_hat,
            Float thickness,
            Var<BufferView<float3>> sa_cgB,
            Var<BufferView<float3x3>> sa_cgA_diag) {
            const Uint pair_idx = dispatch_x();
            const auto& pair = narrowphase_list_ee->read(pair_idx);
            const auto indices = CollisionPair::get_indices(pair);

            const Float3 normal = CollisionPair::get_direction(pair);
            const Float4 weight = CollisionPair::get_ee_weight(pair);

            // const Float stiff = CollisionPair::get_stiff(pair);
            const Float2 stiff = CollisionPair::get_ee_stiff(pair);
            const Float k1 = stiff[0];
            const Float k2 = stiff[1];

            {
                // const Float C = thickness + d_hat - d;
                const Float3x3 nnT = outer_product(normal, normal);

                for (uint j = 0; j < 4; j++) {
                    Float3 force;
                    Float3x3 hessian;
                    force = k1 * weight[j] * normal;
                    hessian = k2 * weight[j] * weight[j] * nnT;

                    // device_log("VF pair {} on vert {} : force = {}, diff = {}, stiff = {}, weight = {}, d = {}/{}", pair_idx, indices[j], force, diff, stiff, weight, d, dot_vec(diff, normal));
                    atomic_add_float3x3(sa_cgA_diag, indices[j], hessian);
                    atomic_add_float3(sa_cgB, indices[j], force);
                }
            };
        });

    // SpMV
    fn_atomic_add_spmv_vf = device.compile<1>(
        [narrowphase_list_vf = collision_data->narrow_phase_list_vf.view(), &atomic_add_float3](
            Var<BufferView<float3>> input_array,
            Var<BufferView<float3>> output_array) {
            const Uint pair_idx = dispatch_x();
            const auto& pair = narrowphase_list_vf->read(pair_idx);
            const auto indices = CollisionPair::get_indices(pair);

            Float3 input_vec[4] = {
                input_array.read(indices[0]),
                input_array.read(indices[1]),
                input_array.read(indices[2]),
                input_array.read(indices[3]),
            };
            Float3 output_vec[4] = {
                make_float3(0.0f),
                make_float3(0.0f),
                make_float3(0.0f),
                make_float3(0.0f),
            };

            const Float stiff = CollisionPair::get_vf_k2(pair);
            const Float3 normal = CollisionPair::get_direction(pair);
            const Float4 weight = CollisionPair::get_vf_weight(pair);
            const Float3x3 xxT = stiff * outer_product(normal, normal);

            for (uint j = 0; j < 4; j++) {
                for (uint jj = 0; jj < 4; jj++) {
                    if (j != jj) {
                        Float3x3 hessian = weight[j] * weight[jj] * xxT;
                        output_vec[j] += hessian * input_vec[jj];
                    }
                }
            }

            atomic_add_float3(output_array, indices[0], output_vec[0]);
            atomic_add_float3(output_array, indices[1], output_vec[1]);
            atomic_add_float3(output_array, indices[2], output_vec[2]);
            atomic_add_float3(output_array, indices[3], output_vec[3]);
        });

    fn_atomic_add_spmv_ee = device.compile<1>(
        [narrowphase_list_ee = collision_data->narrow_phase_list_ee.view(), &atomic_add_float3](
            Var<BufferView<float3>> input_array,
            Var<BufferView<float3>> output_array) {
            const Uint pair_idx = dispatch_x();
            const auto& pair = narrowphase_list_ee->read(pair_idx);
            const auto indices = CollisionPair::get_indices(pair);

            Float3 input_vec[4] = {
                input_array.read(indices[0]),
                input_array.read(indices[1]),
                input_array.read(indices[2]),
                input_array.read(indices[3]),
            };
            Float3 output_vec[4] = {
                make_float3(0.0f),
                make_float3(0.0f),
                make_float3(0.0f),
                make_float3(0.0f),
            };

            const Float stiff = CollisionPair::get_vf_k2(pair);
            const Float3 normal = CollisionPair::get_direction(pair);
            const Float4 weight = CollisionPair::get_ee_weight(pair);
            const Float3x3 xxT = stiff * outer_product(normal, normal);

            for (uint j = 0; j < 4; j++) {
                for (uint jj = 0; jj < 4; jj++) {
                    if (j != jj) {
                        Float3x3 hessian = weight[j] * weight[jj] * xxT;
                        output_vec[j] += hessian * input_vec[jj];
                    }
                }
            }

            atomic_add_float3(output_array, indices[0], output_vec[0]);
            atomic_add_float3(output_array, indices[1], output_vec[1]);
            atomic_add_float3(output_array, indices[2], output_vec[2]);
            atomic_add_float3(output_array, indices[3], output_vec[3]);
        });
}

void NarrowPhasesDetector::compute_repulsion_gradiant_hessian_and_assemble(luisa::compute::Stream& stream,
    const Buffer<float3>& sa_x_left,
    const Buffer<float3>& sa_x_right,
    const float d_hat,
    const float thickness,
    Buffer<float3>& sa_cgB, Buffer<float3x3>& sa_cgA_diag)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;
    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    if (num_vf != 0)
        stream << fn_assemble_repulsion_hessian_gradient_vf(sa_x_left, sa_x_right, d_hat, thickness, sa_cgB, sa_cgA_diag).dispatch(num_vf);
    if (num_ee != 0)
        stream << fn_assemble_repulsion_hessian_gradient_ee(sa_x_left, sa_x_right, d_hat, thickness, sa_cgB, sa_cgA_diag).dispatch(num_ee);
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

    CpuParallel::single_thread_for(0, num_vf, [&](const uint pair_idx) {
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
        // const float stiff = CollisionPair::get_stiff(pair);
        const float stiff = CollisionPair::get_vf_k2(pair);
        const float3 normal = CollisionPair::get_direction(pair);
        const float4 weight = CollisionPair::get_vf_weight(pair);
        const float3x3 xxT = stiff * outer_product(normal, normal);

        for (uint j = 0; j < 4; j++) {
            for (uint jj = 0; jj < 4; jj++) {
                if (j != jj) {
                    float3x3 hessian = weight[j] * weight[jj] * xxT;
                    output_vec[j] += hessian * input_vec[jj];
                }
            }
        }
        output_array[indices[0]] += output_vec[0];
        output_array[indices[1]] += output_vec[1];
        output_array[indices[2]] += output_vec[2];
        output_array[indices[3]] += output_vec[3];
    });

    CpuParallel::single_thread_for(0, num_ee, [&](const uint pair_idx) {
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
        // const float stiff = CollisionPair::get_stiff(pair);
        const float stiff = CollisionPair::get_ee_k2(pair);
        const float3 normal = CollisionPair::get_direction(pair);
        const float4 weight = CollisionPair::get_ee_weight(pair);
        const float3x3 xxT = stiff * outer_product(normal, normal);

        for (uint j = 0; j < 4; j++) {
            for (uint jj = 0; jj < 4; jj++) {
                if (j != jj) {
                    float3x3 hessian = weight[j] * weight[jj] * xxT;
                    output_vec[j] += hessian * input_vec[jj];
                }
            }
        }
        output_array[indices[0]] += output_vec[0];
        output_array[indices[1]] += output_vec[1];
        output_array[indices[2]] += output_vec[2];
        output_array[indices[3]] += output_vec[3];
    });
}
void NarrowPhasesDetector::device_spmv(Stream& stream, const Buffer<float3>& input_array, Buffer<float3>& output_array)
{
    // Off-diag: Collision hessian
    auto narrowphase_count = collision_data->narrow_phase_collision_count.view();
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vf = host_count[collision_data->get_vf_count_offset()];
    const uint num_ee = host_count[collision_data->get_ee_count_offset()];

    if (num_vf != 0)
        stream << fn_atomic_add_spmv_vf(input_array, output_array).dispatch(num_vf);
    if (num_ee != 0)
        stream << fn_atomic_add_spmv_ee(input_array, output_array).dispatch(num_ee);
}

} // namespace lcs

namespace lcs // Compute barrier energy
{

void NarrowPhasesDetector::compile_energy(luisa::compute::Device& device, const ContactEnergyType contact_energy_type)
{
    using namespace luisa::compute;

    const uint offset_vv = collision_data->get_vv_count_offset();
    const uint offset_ve = collision_data->get_ve_count_offset();
    const uint offset_vf = collision_data->get_vf_count_offset();
    const uint offset_ee = collision_data->get_ee_count_offset();

    fn_compute_repulsion_energy_from_vf = device.compile<1>(
        [contact_energy = collision_data->contact_energy.view(offset_vf, 1),
            narrowphase_list_vf = collision_data->narrow_phase_list_vf.view(), contact_energy_type](
            Var<BufferView<float3>> sa_x_left,
            Var<BufferView<float3>> sa_x_right,
            Float d_hat,
            Float thickness,
            Float kappa) {
            const Uint pair_idx = dispatch_x();
            const auto& pair = narrowphase_list_vf->read(pair_idx);
            const auto indices = CollisionPair::get_indices(pair);

            const Float3 normal = CollisionPair::get_direction(pair);
            const Float4 weight = CollisionPair::get_vf_weight(pair);
            const Float3 diff = weight[0] * sa_x_left.read(indices[0]) + weight[1] * sa_x_right.read(indices[1]) + weight[2] * sa_x_right.read(indices[2]) + weight[3] * sa_x_right.read(indices[3]);
            const Float d2 = length_squared_vec(diff);
            const Float d = sqrt_scalar(d2);
            // const Float d = dot_vec(diff, normal);

            Float energy = 0.0f;

            $if(d < thickness + d_hat)
            {
                const Float area = CollisionPair::get_area(pair);
                if (contact_energy_type == ContactEnergyType::Quadratic) {
                    Float C = thickness + d_hat - d;
                    const Float stiff = kappa * area; // k2 = stiffness * area
                    energy = 0.5f * stiff * C * C;
                    // device_log("VF energy : Pair {} , weight = {}, diff = {}, normal = {} d = {}, proj = {}, C = {}, E = {} ", pair_idx, weight, diff, normal, length_vec(diff), d, C, energy);
                } else if (contact_energy_type == ContactEnergyType::Barrier) {
                    cipc::KappaBarrier(energy, area * kappa, d2, d_hat, thickness);
                }
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if(pair_idx % 256 == 0)
            {
                $if(energy != 0.0f)
                {
                    contact_energy->atomic(0).fetch_add(energy);
                };
            };
        });

    fn_compute_repulsion_energy_from_ee = device.compile<1>(
        [contact_energy = collision_data->contact_energy.view(offset_ee, 1),
            narrowphase_list_ee = collision_data->narrow_phase_list_ee.view(), contact_energy_type](
            Var<BufferView<float3>> sa_x_left,
            Var<BufferView<float3>> sa_x_right,
            Float d_hat,
            Float thickness,
            Float kappa) {
            const Uint pair_idx = dispatch_x();
            const auto& pair = narrowphase_list_ee->read(pair_idx);
            const auto indices = CollisionPair::get_indices(pair);

            const Float3 normal = CollisionPair::get_direction(pair);
            const Float4 weight = CollisionPair::get_ee_weight(pair);
            const Float3 diff = weight[0] * sa_x_left.read(indices[0]) + weight[1] * sa_x_left.read(indices[1]) + weight[2] * sa_x_right.read(indices[2]) + weight[3] * sa_x_right.read(indices[3]);
            const Float d2 = length_squared_vec(diff);
            const Float d = sqrt_scalar(d2);
            // const Float d = dot_vec(diff, normal);

            Float energy = 0.0f;

            $if(d < thickness + d_hat)
            {
                const Float area = CollisionPair::get_area(pair);
                if (contact_energy_type == ContactEnergyType::Quadratic) {
                    Float C = thickness + d_hat - d;
                    const Float stiff = kappa * area; // k2 = stiffness * area
                    energy = 0.5f * stiff * C * C;
                } else if (contact_energy_type == ContactEnergyType::Barrier) {
                    cipc::KappaBarrier(energy, area * kappa, d2, d_hat, thickness);
                }
            };

            energy = ParallelIntrinsic::block_intrinsic_reduce(pair_idx, energy, ParallelIntrinsic::warp_reduce_op_sum<float>);

            $if(pair_idx % 256 == 0)
            {
                $if(energy != 0.0f)
                {
                    contact_energy->atomic(0).fetch_add(energy);
                };
            };
        });
}

void NarrowPhasesDetector::compute_penalty_energy_from_vf(Stream& stream,
    const Buffer<float3>& sa_x_left,
    const Buffer<float3>& sa_x_right,
    const Buffer<float3>& sa_rest_x_left,
    const Buffer<float3>& sa_rest_x_right,
    const Buffer<float>& sa_rest_area_left,
    const Buffer<float>& sa_rest_area_right,
    const Buffer<uint3>& sa_faces_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto& contact_energy = collision_data->contact_energy;
    auto& host_count = host_collision_data->narrow_phase_collision_count;

    const uint num_vf_narrowphase = host_count[collision_data->get_vf_count_offset()];

    if (num_vf_narrowphase != 0) {
        stream << fn_compute_repulsion_energy_from_vf(
            sa_x_left,
            sa_x_right,
            d_hat, thickness, kappa)
                      .dispatch(num_vf_narrowphase)
            // << contact_energy.view(2, 1).copy_to(host_contact_energy.data() + 2)
            ;
    }
}

void NarrowPhasesDetector::compute_penalty_energy_from_ee(Stream& stream,
    const Buffer<float3>& sa_x_left,
    const Buffer<float3>& sa_x_right,
    const Buffer<float3>& sa_rest_x_left,
    const Buffer<float3>& sa_rest_x_right,
    const Buffer<float>& sa_rest_area_left,
    const Buffer<float>& sa_rest_area_right,
    const Buffer<uint2>& sa_edges_left,
    const Buffer<uint2>& sa_edges_right,
    const float d_hat,
    const float thickness,
    const float kappa)
{
    auto& host_count = host_collision_data->narrow_phase_collision_count;
    const uint num_ee_narrowphase = host_count[collision_data->get_ee_count_offset()];
    if (num_ee_narrowphase != 0) {
        stream << fn_compute_repulsion_energy_from_ee(
            sa_x_left,
            sa_x_right,
            d_hat, thickness, kappa)
                      .dispatch(num_ee_narrowphase);
    }
}

} // namespace lcs

namespace lcs // Host Methods
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

    // LUISA_INFO("num_vf_broadphase = {}", num_vf_broadphase);
    // LUISA_INFO("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcs::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_vf_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

    float min_toi = host_accd::line_search_max_t;
    min_toi = CpuParallel::parallel_for_and_reduce(0, num_vf_broadphase, [&](const uint pair_idx) {
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
            // LUISA_INFO("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", 
            //     pair_idx, toi, left, right, right_face,
            // );
            LUISA_INFO("VF Pair {} : toi = {}, vid {} & fid {} (face {}), dist = {} -> {}", 
                pair_idx, toi, left, right, right_face, 
                host_distance::point_triangle_distance_squared_unclassified(t0_p, t0_f0, t0_f1, t0_f2),
                host_distance::point_triangle_distance_squared_unclassified(t1_p, t1_f0, t1_f1, t1_f2)
            );
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_left[left]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_left[left]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[0]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[1]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_right[right_face[2]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[0]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[1]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_right[right_face[2]]);
        }
        return toi; }, [](const float left, const float right) { return min_scalar(left, right); }, host_accd::line_search_max_t);

    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);

    // min_toi /= host_accd::line_search_max_t;
    // if (min_toi < 1e-5)
    // {
    //     LUISA_ERROR("toi is too small : {}", min_toi);
    // }
    // LUISA_INFO("toi = {}", min_toi);
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

    // LUISA_INFO("num_ee_broadphase = {}", num_ee_broadphase);

    uint2* pair_view = (lcs::uint2*)host_list.data();
    // CpuParallel::parallel_sort(pair_view, pair_view + num_ee_broadphase, [](const uint2& left, const uint2& right)
    // {
    //     if (left[0] == right[0]) { return left[1] < right[1]; }
    //     return left[0] < right[0];
    // });

    float min_toi = 1.25f;
    min_toi = CpuParallel::parallel_for_and_reduce(0, num_ee_broadphase, [&](const uint pair_idx) {
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
            LUISA_INFO("EE Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", pair_idx, toi, left, left_edge, right, right_edge);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_a[left_edge[0]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_a[left_edge[1]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_b[right_edge[0]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_begin_b[right_edge[1]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_a[left_edge[0]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_a[left_edge[1]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_b[right_edge[0]]);
            // LUISA_INFO("             {} : positions : {}", pair_idx, sa_x_end_b[right_edge[1]]);
        }
        return toi; }, [](const float left, const float right) { return min_scalar(left, right); }, host_accd::line_search_max_t);

    sa_toi[0] = min_scalar(min_toi, sa_toi[0]);

    // min_toi /= host_accd::line_search_max_t;
    // if (min_toi < 1e-5)
    // {
    //     LUISA_ERROR("toi is too small : {}", min_toi);
    // }
    // LUISA_INFO("toi = {}", min_toi);
    // sa_toi[0] = min_scalar(min_toi, sa_toi[0]);
}

void NarrowPhasesDetector::unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    using namespace luisa::compute;

    // VF CCD Test
    if constexpr (false) {
        const float desire_toi = 0.6930697;
        LUISA_INFO("VF Test, desire for toi {}", desire_toi);

        const uint vid = 1;
        const uint fid = 2;
        const uint3 face = uint3(4, 7, 5);
        float3 case_t0_p = luisa::make_float3(0.48159984, -0.26639974, -0.48159984);
        float3 case_t1_p = luisa::make_float3(0.47421163, -0.3129394, -0.47421163);
        float3 case_t0_f0 = luisa::make_float3(-0.4, -0.3, -0.5);
        float3 case_t0_f1 = luisa::make_float3(0.6, -0.3, 0.5);
        float3 case_t0_f2 = luisa::make_float3(0.6, -0.3, -0.5);
        float3 case_t1_f0 = luisa::make_float3(-0.4, -0.3, -0.5);
        float3 case_t1_f1 = luisa::make_float3(0.6, -0.3, 0.5);
        float3 case_t1_f2 = luisa::make_float3(0.6, -0.3, -0.5);

        {
            const auto t0_p = float3_to_eigen3(case_t0_p);
            const auto t1_p = float3_to_eigen3(case_t1_p);
            const auto t0_f0 = float3_to_eigen3(case_t0_f0);
            const auto t0_f1 = float3_to_eigen3(case_t0_f1);
            const auto t0_f2 = float3_to_eigen3(case_t0_f2);
            const auto t1_f0 = float3_to_eigen3(case_t1_f0);
            const auto t1_f1 = float3_to_eigen3(case_t1_f1);
            const auto t1_f2 = float3_to_eigen3(case_t1_f2);

            float toi = host_accd::point_triangle_ccd(t0_p, t1_p,
                t0_f0, t0_f1,
                t0_f2, t1_f0,
                t1_f1, t1_f2,
                1e-3);
            LUISA_INFO("BroadPhase Pair {} : toi = {}, vid {} & fid {} (face {})", 0, toi, vid, fid, face);
        }
        {
            auto fn_test_ccd_vf = device.compile<1>([&](Float thickness) {
                Uint pair_idx = 0;
                Float toi = accd::line_search_max_t;

                {
                    Float3 t0_p = case_t0_p;
                    Float3 t1_p = case_t1_p;
                    Float3 t0_f0 = case_t0_f0;
                    Float3 t0_f1 = case_t0_f1;
                    Float3 t0_f2 = case_t0_f2;
                    Float3 t1_f0 = case_t1_f0;
                    Float3 t1_f1 = case_t1_f1;
                    Float3 t1_f2 = case_t1_f2;

                    Float toi = accd::point_triangle_ccd(t0_p, t1_p,
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
    // if constexpr (false)
    {
        // float desire_toi = 0.91535777;
        // LUISA_INFO("EE Test, desire for toi {}", desire_toi);

        const uint left = 4;
        const uint right = 6;
        const uint2 left_edge = uint2(2, 3);
        const uint2 right_edge = uint2(4, 6);

        float3 case_ea_t0_p0 = luisa::make_float3(-0.402716, -0.290011, 0.452109);
        float3 case_ea_t0_p1 = luisa::make_float3(0.50008, 0.138455, 0.490343);
        float3 case_eb_t0_p0 = luisa::make_float3(-0.4, -0.300001, -0.5);
        float3 case_eb_t0_p1 = luisa::make_float3(-0.399998, -0.300016, 0.5);
        float3 case_ea_t1_p0 = luisa::make_float3(-0.40609047, -0.30418798, 0.4480959);
        float3 case_ea_t1_p1 = luisa::make_float3(0.500001, 0.11660194, 0.4934225);
        float3 case_eb_t1_p0 = luisa::make_float3(-0.4, -0.300001, -0.5);
        float3 case_eb_t1_p1 = luisa::make_float3(-0.399998, -0.300016, 0.5);
        const float thickenss = 0;

        // float3 case_ea_t0_p0 = luisa::make_float3(-0.499492, -0.279657, 0.460444);
        // float3 case_ea_t0_p1 = luisa::make_float3(0.499997, -0.248673, 0.468853);
        // float3 case_eb_t0_p0 = luisa::make_float3(-0.4, -0.3, -0.5);
        // float3 case_eb_t0_p1 = luisa::make_float3(-0.4, -0.3, 0.5);
        // float3 case_ea_t1_p0 = luisa::make_float3(-0.49939114, -0.30410385, 0.4529846);
        // float3 case_ea_t1_p1 = luisa::make_float3(0.4999971, -0.27044764, 0.4630015);
        // float3 case_eb_t1_p0 = luisa::make_float3(-0.4, -0.3, -0.5);
        // float3 case_eb_t1_p1 = luisa::make_float3(-0.4, -0.3, 0.5);
        // const float thickness = 1e-3;

        {
            const auto ea00 = float3_to_eigen3(case_ea_t0_p0);
            const auto ea01 = float3_to_eigen3(case_ea_t0_p1);
            const auto eb00 = float3_to_eigen3(case_eb_t0_p0);
            const auto eb01 = float3_to_eigen3(case_eb_t0_p1);
            const auto ea10 = float3_to_eigen3(case_ea_t1_p0);
            const auto ea11 = float3_to_eigen3(case_ea_t1_p1);
            const auto eb10 = float3_to_eigen3(case_eb_t1_p0);
            const auto eb11 = float3_to_eigen3(case_eb_t1_p1);

            auto d2 = host_distance::edge_edge_distance_squared_unclassified(ea00, ea01, eb00, eb01);
            LUISA_INFO("Start distance = {}", d2);

            float toi = host_accd::edge_edge_ccd(
                ea00,
                ea01,
                eb00,
                eb01,
                ea10,
                ea11,
                eb10,
                eb11, 0);
            LUISA_INFO("BroadPhase Pair {} : toi = {}, edge1 {} ({}) & edge2 {} ({})", 0, toi, left, left_edge, right, right_edge);
        }

        std::vector<float3> input_positions = {
            case_ea_t0_p0,
            case_ea_t0_p1,
            case_eb_t0_p0,
            case_eb_t0_p1,
            case_ea_t1_p0,
            case_ea_t1_p1,
            case_eb_t1_p0,
            case_eb_t1_p1,
        };
        luisa::compute::Buffer<float3> buffer = device.create_buffer<float3>(input_positions.size());
        stream << buffer.copy_from(input_positions.data());

        auto fn_test_ccd_ee = device.compile<1>([&](Float thickness) {
            Uint pair_idx = 0;
            Float toi = accd::line_search_max_t;

            {
                Float3 ea_t0_p0 = buffer->read(0);
                Float3 ea_t0_p1 = buffer->read(1);
                Float3 eb_t0_p0 = buffer->read(2);
                Float3 eb_t0_p1 = buffer->read(3);
                Float3 ea_t1_p0 = buffer->read(4);
                Float3 ea_t1_p1 = buffer->read(5);
                Float3 eb_t1_p0 = buffer->read(6);
                Float3 eb_t1_p1 = buffer->read(7);

                auto d2 = distance::edge_edge_distance_squared_unclassified(ea_t0_p0, ea_t0_p1, eb_t0_p0, eb_t0_p1);
                device_log("Start distance = {}", d2);

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

        stream << fn_test_ccd_ee(0).dispatch(1) << synchronize();
    }
}

}
