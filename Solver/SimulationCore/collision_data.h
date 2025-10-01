#pragma once

#include "Core/float_nxn.h"
#include "Core/lc_to_eigen.h"
#include "Core/matrix_triplet.h"
#include "SimulationCore/simulation_type.h"
#include "Utils/buffer_allocator.h"
#include <vector>
#include <string>
#include <luisa/luisa-compute.h>


namespace lcs
{

// struct CollisionPairVV
// {
//     uint2 indices; // vid1:1, vid2:1
//     float4 vec1; // normal:3, stiff:1
//     float3 gradient[2];
//     float3x3 hessian[3];
// };
// struct CollisionPairVE
// {
//     uint2 edge;
//     uint vid;
//     float bary;
//     float4 vec1; // normal:3, stiff 1
//     float3 gradient[3];
//     float3x3 hessian[6];
// };
// struct CollisionPairVF
// {
//     uint4 indices; // vid:1, face:3
//     float4 vec1; // normal:3, stiff:1
//     float2 bary; // bary
//     float2 vec2;
//     float3 gradient[4];
//     float3x3 hessian[10];
// };
// struct CollisionPairEE
// {
//     uint4 indices;
//     float4 vec1; // normal:3, stiff 1
//     float2 bary; //
//     float2 vec2;
//     float3 gradient[4];
//     float3x3 hessian[10];
// };
struct CollisionPairVV
{
    uint2  indices;  // vid1:1, vid2:1
    float4 vec1;     // normal:3, stiff:1
};
struct CollisionPairVE
{
    uint2  edge;
    uint   vid;
    float  bary;
    float4 vec1;  // normal:3, stiff 1
};
struct CollisionPairVF
{
    uint4  indices;  // vid:1, face:3
    float4 vec1;     // normal:3, stiff:1
    float2 bary;     // bary
    float2 vec2;
};
struct CollisionPairEE
{
    uint4  indices;
    float4 vec1;  // normal:3, stiff 1
    float2 bary;  //
    float2 vec2;
};
struct ReducedCollisionPairInfo
{
    std::array<float, 3> weighted_model_pos1;
    uint                 affine_body_idx1;
    std::array<float, 3> weighted_model_pos2;
    uint                 affine_body_idx2;
};
// enum CollisionListType
// {
//     CollisionListTypeVV,
//     CollisionListTypeVF,
//     CollisionListTypeEE,
//     CollisionListTypeEF,
// };

}  // namespace lcs

LUISA_STRUCT(lcs::CollisionPairVV, indices, vec1){};
LUISA_STRUCT(lcs::CollisionPairVE, edge, vid, bary, vec1){};
LUISA_STRUCT(lcs::CollisionPairVF, indices, vec1, bary, vec2){};
LUISA_STRUCT(lcs::CollisionPairEE, indices, vec1, bary, vec2){};

// LUISA_STRUCT(lcs::CollisionPairVV, indices, vec1, gradient, hessian) {};
// LUISA_STRUCT(lcs::CollisionPairVE, edge, vid, bary, vec1, gradient, hessian) {};
// LUISA_STRUCT(lcs::CollisionPairVF, indices, vec1, bary, vec2, gradient, hessian) {};
// LUISA_STRUCT(lcs::CollisionPairEE, indices, vec1, bary, vec2, gradient, hessian) {};


namespace lcs
{

namespace CollisionPair
{
    namespace CollisionCount
    {
        // clang-format off
        inline constexpr uint vv_offset() { return 0; }
        inline constexpr uint ve_offset() { return 1; }
        inline constexpr uint vf_offset() { return 2; }
        inline constexpr uint ee_offset() { return 3; }
        inline constexpr uint per_vert_vv_offset() { return 4; }
        inline constexpr uint per_vert_ve_offset() { return 5; }
        inline constexpr uint per_vert_vf_offset() { return 6; }
        inline constexpr uint per_vert_ee_offset() { return 7; }
        // clang-format on
    }  // namespace CollisionCount
}  // namespace CollisionPair


template <template <typename...> typename BufferType>
struct CollisionData : SimulationType
{
    BufferType<uint> broad_phase_collision_count;   // 0: VV, 1: VE, 2: VF, 3: EE
    BufferType<uint> narrow_phase_collision_count;  // 0: VV, 1: VE, 2: VF, 3: EE
                                                    // 4: PerVertVV, 5: PerVertVE,
                                                    // 6: PerVertVF, 6: PervertEE

    BufferType<uint>  broad_phase_list_vf;
    BufferType<uint>  broad_phase_list_ee;
    BufferType<float> toi_per_vert;
    BufferType<float> contact_energy;

    BufferType<CollisionPairVV>          narrow_phase_list_vv;  // 0
    BufferType<CollisionPairVE>          narrow_phase_list_ve;  // 1
    BufferType<CollisionPairVF>          narrow_phase_list_vf;  // 2
    BufferType<CollisionPairEE>          narrow_phase_list_ee;  // 3
    BufferType<ReducedCollisionPairInfo> reduced_narrow_phase_list_info_vf;
    BufferType<ReducedCollisionPairInfo> reduced_narrow_phase_list_info_ee;
    // BufferType<uint> narrow_phase_indices_ef;

    BufferType<uint> per_vert_num_broad_phase_vf;
    BufferType<uint> per_vert_num_broad_phase_ee;

    BufferType<uint>                       per_vert_num_narrow_phase_vv;
    BufferType<uint>                       per_vert_num_narrow_phase_ve;
    BufferType<uint>                       per_vert_num_narrow_phase_vf;
    BufferType<uint>                       per_vert_num_narrow_phase_ee;
    BufferType<uint>                       per_vert_prefix_narrow_phase_vv;
    BufferType<uint>                       per_vert_prefix_narrow_phase_ve;
    BufferType<uint>                       per_vert_prefix_narrow_phase_vf;
    BufferType<uint>                       per_vert_prefix_narrow_phase_ee;
    BufferType<ushort>                     narrow_phase_vf_pair_offset_in_vert;
    BufferType<ushort>                     narrow_phase_ee_pair_offset_in_vert;
    BufferType<uint>                       vert_adj_vf_pairs_csr;
    BufferType<uint>                       vert_adj_ee_pairs_csr;
    luisa::compute::IndirectDispatchBuffer collision_indirect_cmd_buffer_broad_phase;
    luisa::compute::IndirectDispatchBuffer collision_indirect_cmd_buffer_narrow_phase;


    const uint get_vv_count_offset() { return 0; }
    const uint get_ve_count_offset() { return 1; }
    const uint get_vf_count_offset() { return 2; }
    const uint get_ee_count_offset() { return 3; }

    // template<template<typename...> typename BufferType>
    inline void resize_collision_data(luisa::compute::Device& device, const uint num_verts, const uint num_faces, const uint num_edges)
    {
        // const uint num_verts = mesh_data->num_verts;
        // const uint num_edges = mesh_data->num_edges;

        const uint per_element_count_BP = 256;
        const uint per_element_count_NP = 96;

        constexpr bool use_vv_ve = false;
        lcs::Initializer::resize_buffer(device, this->broad_phase_collision_count, 4);
        lcs::Initializer::resize_buffer(device, this->narrow_phase_collision_count, 8);
        lcs::Initializer::resize_buffer(device, this->contact_energy, 4);
        lcs::Initializer::resize_buffer(device, this->toi_per_vert, num_verts);
        lcs::Initializer::resize_buffer(device, this->broad_phase_list_vf, per_element_count_BP * num_verts);
        lcs::Initializer::resize_buffer(device, this->broad_phase_list_ee, per_element_count_BP * num_edges);
        // lcs::Initializer::resize_buffer(device, this->narrow_phase_list_vv, per_element_count_NP * num_verts);
        // lcs::Initializer::resize_buffer(device, this->narrow_phase_list_ve, per_element_count_NP * num_verts);
        if (use_vv_ve)
            lcs::Initializer::resize_buffer(device, this->narrow_phase_list_vv, per_element_count_NP * num_verts);
        if (use_vv_ve)
            lcs::Initializer::resize_buffer(device, this->narrow_phase_list_ve, per_element_count_NP * num_verts);
        if (!use_vv_ve)
            lcs::Initializer::resize_buffer(device, this->narrow_phase_list_vv, 1);
        if (!use_vv_ve)
            lcs::Initializer::resize_buffer(device, this->narrow_phase_list_ve, 1);
        lcs::Initializer::resize_buffer(device, this->narrow_phase_list_vf, per_element_count_NP * num_verts);
        lcs::Initializer::resize_buffer(device, this->narrow_phase_list_ee, per_element_count_NP * num_edges);
        lcs::Initializer::resize_buffer(device, this->vert_adj_vf_pairs_csr, 4 * per_element_count_NP * num_verts);
        lcs::Initializer::resize_buffer(device, this->vert_adj_ee_pairs_csr, 4 * per_element_count_NP * num_edges);
        lcs::Initializer::resize_buffer(device, this->narrow_phase_vf_pair_offset_in_vert, 4 * per_element_count_NP * num_verts);
        lcs::Initializer::resize_buffer(device, this->narrow_phase_ee_pair_offset_in_vert, 4 * per_element_count_NP * num_edges);
        lcs::Initializer::resize_buffer(device, this->reduced_narrow_phase_list_info_vf, per_element_count_NP * num_verts);
        lcs::Initializer::resize_buffer(device, this->reduced_narrow_phase_list_info_ee, per_element_count_NP * num_edges);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_broad_phase_vf, num_verts);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_broad_phase_ee, num_verts);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_vv, num_verts);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_ve, num_verts);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_vf, num_verts);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_narrow_phase_ee, num_verts);
        lcs::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_vv, num_verts + 1);
        lcs::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_ve, num_verts + 1);
        lcs::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_vf, num_verts + 1);
        lcs::Initializer::resize_buffer(device, this->per_vert_prefix_narrow_phase_ee, num_verts + 1);
        this->collision_indirect_cmd_buffer_broad_phase  = device.create_indirect_dispatch_buffer(2);
        this->collision_indirect_cmd_buffer_narrow_phase = device.create_indirect_dispatch_buffer(4);

        const uint collision_pair_bytes =
            sizeof(uint) * this->broad_phase_list_vf.size() + sizeof(uint) * this->broad_phase_list_ee.size()
            + sizeof(CollisionPairVV) * this->narrow_phase_list_vv.size()
            + sizeof(CollisionPairVE) * this->narrow_phase_list_ve.size()
            + sizeof(CollisionPairVF) * this->narrow_phase_list_vf.size()
            + sizeof(CollisionPairEE) * this->narrow_phase_list_ee.size()
            + sizeof(uint) * this->vert_adj_vf_pairs_csr.size()
            + sizeof(uint) * this->vert_adj_ee_pairs_csr.size()
            + sizeof(ushort) * this->narrow_phase_vf_pair_offset_in_vert.size()
            + sizeof(ushort) * this->narrow_phase_ee_pair_offset_in_vert.size()
            + sizeof(CollisionPairEE) * this->reduced_narrow_phase_list_info_vf.size()
            + sizeof(CollisionPairEE) * this->reduced_narrow_phase_list_info_ee.size();

        LUISA_INFO("Allocated collision buffer size {} MB", collision_pair_bytes / (1024 * 1024));
        if (float(collision_pair_bytes) / (1024 * 1024 * 1024) > 1.0f)
            LUISA_INFO("Allocated buffer size for collision pair = {} GB",
                       collision_pair_bytes / (1024 * 1024 * 1024));
    }
};


}  // namespace lcs


// Matrix free hessian functions
namespace lcs
{
namespace CollisionPair
{

    // template<typename T> auto get_indices(const T& pair) { return pair.indices; }
    inline auto get_indices(const CollisionPairVV& pair)
    {
        return pair.indices;
    }
    inline auto get_indices(const CollisionPairVE& pair)
    {
        return luisa::make_uint3(pair.vid, pair.edge[0], pair.edge[1]);
    }
    inline auto get_indices(const CollisionPairVF& pair)
    {
        return pair.indices;
    }
    inline auto get_indices(const CollisionPairEE& pair)
    {
        return pair.indices;
    }
    inline auto get_indices(const Var<CollisionPairVV>& pair)
    {
        return pair.indices;
    }
    inline auto get_indices(const Var<CollisionPairVE>& pair)
    {
        return luisa::compute::make_uint3(pair.vid, pair.edge[0], pair.edge[1]);
    }
    inline auto get_indices(const Var<CollisionPairVF>& pair)
    {
        return pair.indices;
    }
    inline auto get_indices(const Var<CollisionPairEE>& pair)
    {
        return pair.indices;
    }

    template <typename T>
    auto get_vv_vid1(const T& pair)
    {
        return pair.indices[0];
    }
    template <typename T>
    auto get_vv_vid2(const T& pair)
    {
        return pair.indices[1];
    }
    template <typename T>
    auto get_ve_vid(const T& pair)
    {
        return pair.vid;
    }
    template <typename T>
    auto get_ve_edge(const T& pair)
    {
        return pair.edge;
    }
    template <typename T>
    auto get_vf_vid(const T& pair)
    {
        return pair.indices[0];
    }
    template <typename T>
    auto get_vf_face(const T& pair)
    {
        return pair.indices.yzw();
    }
    template <typename T>
    auto get_ee_edge1(const T& pair)
    {
        return pair.indices.xy();
    }
    template <typename T>
    auto get_ee_edge2(const T& pair)
    {
        return pair.indices.zw();
    }

    // template<typename T> auto get_stiff(const T& pair) { return pair.vec1[3]; }
    template <typename T>
    auto get_direction(const T& pair)
    {
        return pair.vec1.xyz();
    }
    template <typename T>
    auto get_area(const T& pair)
    {
        return pair.vec1[3];
    }

    // inline auto get_vv_bary(const CollisionPairVV& pair) { return makeFloat2(1.0f, 1.0f); }
    inline auto get_ve_edge_bary(const CollisionPairVE& pair)
    {
        return luisa::make_float2(pair.bary, 1.0f - pair.bary);
    }
    inline auto get_vf_face_bary(const CollisionPairVF& pair)
    {
        return luisa::make_float3(pair.bary[0], pair.bary[1], 1.0f - pair.bary[0] - pair.bary[1]);
    }
    inline auto get_ee_edge1_bary(const CollisionPairEE& pair)
    {
        return luisa::make_float2(pair.bary[0], 1.0f - pair.bary[0]);
    }
    inline auto get_ee_edge2_bary(const CollisionPairEE& pair)
    {
        return luisa::make_float2(pair.bary[1], 1.0f - pair.bary[1]);
    }

    inline auto get_ve_edge_bary(const Var<CollisionPairVE>& pair)
    {
        return luisa::compute::make_float2(pair.bary, 1.0f - pair.bary);
    }
    inline auto get_vf_face_bary(const Var<CollisionPairVF>& pair)
    {
        return luisa::compute::make_float3(pair.bary[0], pair.bary[1], 1.0f - pair.bary[0] - pair.bary[1]);
    }
    inline auto get_ee_edge1_bary(const Var<CollisionPairEE>& pair)
    {
        return luisa::compute::make_float2(pair.bary[0], 1.0f - pair.bary[0]);
    }
    inline auto get_ee_edge2_bary(const Var<CollisionPairEE>& pair)
    {
        return luisa::compute::make_float2(pair.bary[1], 1.0f - pair.bary[1]);
    }

    template <typename T>
    auto get_vf_weight(const T& pair)
    {
        return makeFloat4(1.0f, -pair.bary[0], -pair.bary[1], pair.bary[0] + pair.bary[1] - 1.0f);
    }
    template <typename T>
    auto get_ee_weight(const T& pair)
    {
        return makeFloat4(pair.bary[0], 1.0f - pair.bary[0], -pair.bary[1], pair.bary[1] - 1.0f);
    }
    template <typename T, typename Vec3>
    void write_vf_weight(T& pair, const Vec3& face_bary)
    {
        pair.bary = makeFloat2(face_bary[0], face_bary[1]);
    }
    template <typename T, typename Vec4>
    void write_ee_weight(T& pair, const Vec4& edge_bary)
    {
        pair.bary = makeFloat2(edge_bary[0], edge_bary[2]);
    }

    template <typename T>
    auto get_vf_stiff(const T& pair)
    {
        return pair.vec2;
    }
    template <typename T>
    auto get_ee_stiff(const T& pair)
    {
        return pair.vec2;
    }

    template <typename T>
    auto get_vf_k1(const T& pair)
    {
        return pair.vec2[0];
    }
    template <typename T>
    auto get_vf_k2(const T& pair)
    {
        return pair.vec2[1];
    }
    template <typename T>
    auto get_ee_k1(const T& pair)
    {
        return pair.vec2[0];
    }
    template <typename T>
    auto get_ee_k2(const T& pair)
    {
        return pair.vec2[1];
    }

    template <typename T, typename FloatType>
    void write_vf_stiff(T& pair, const FloatType& k1, const FloatType& k2)
    {
        pair.vec2 = makeFloat2(k1, k2);
    }
    template <typename T, typename FloatType>
    void write_ee_stiff(T& pair, const FloatType& k1, const FloatType& k2)
    {
        pair.vec2 = makeFloat2(k1, k2);
    }

}  // namespace CollisionPair
}  // namespace lcs


// Full matrix
namespace lcs
{
namespace CollisioinPair
{
    inline void write_upper_hessian(luisa::compute::ArrayFloat3x3<3>& hessian, Float6x6& H)
    {
        //   0   2
        //  t2   1
        hessian[0] = H.mat[0][0];
        hessian[1] = H.mat[1][1];
        hessian[2] = H.mat[1][0];
    }
    inline void write_upper_hessian(float3x3 hessian[3], float6x6& H)
    {
        //   0   2
        //  t2   1
        hessian[0] = H.mat[0][0];
        hessian[1] = H.mat[1][1];
        hessian[2] = H.mat[1][0];
    }
    inline void extract_upper_hessian(float3x3 hessian[3], float6x6& H)
    {
        //   0   2
        //  t2   1
        H.mat[0][0] = hessian[0];
        H.mat[0][1] = transpose_mat(hessian[2]);
        H.mat[1][0] = hessian[2];
        H.mat[1][1] = hessian[1];
    }

    inline void write_upper_hessian(luisa::compute::ArrayFloat3x3<6>& hessian, Float9x9& H)
    {
        //   0   3   5
        //  t3   1   4
        //  t5  t4   2
        hessian[0] = H.mat[0][0];
        hessian[1] = H.mat[1][1];
        hessian[2] = H.mat[2][2];
        hessian[3] = H.mat[1][0];
        hessian[4] = H.mat[2][1];
        hessian[5] = H.mat[2][0];
    }
    inline void write_upper_hessian(float3x3 hessian[6], float9x9& H)
    {
        //   0   3   5
        //  t3   1   4
        //  t5  t4   2
        hessian[0] = H.mat[0][0];
        hessian[1] = H.mat[1][1];
        hessian[2] = H.mat[2][2];
        hessian[3] = H.mat[1][0];
        hessian[4] = H.mat[2][1];
        hessian[5] = H.mat[2][0];
    }
    inline void extract_upper_hessian(float3x3 hessian[6], float9x9& H)
    {
        //   0   3   5
        //  t3   1   4
        //  t5  t4   2
        H.mat[0][0] = hessian[0];
        H.mat[0][1] = transpose_mat(hessian[3]);
        H.mat[0][2] = transpose_mat(hessian[5]);
        H.mat[1][0] = hessian[3];
        H.mat[1][1] = hessian[1];
        H.mat[1][2] = transpose_mat(hessian[4]);
        H.mat[2][0] = hessian[5];
        H.mat[2][1] = hessian[4];
        H.mat[2][2] = hessian[2];
    }

    inline void write_upper_hessian(luisa::compute::ArrayFloat3x3<10>& hessian, Float12x12& H)
    {
        //   0   4   7   9
        //  t4   1   5   8
        //  t7  t5   2   6
        //  t9  t8  t6   3
        hessian[0] = H.mat[0][0];
        hessian[1] = H.mat[1][1];
        hessian[2] = H.mat[2][2];
        hessian[3] = H.mat[3][3];
        hessian[4] = H.mat[1][0];
        hessian[5] = H.mat[2][1];
        hessian[6] = H.mat[3][2];
        hessian[7] = H.mat[2][0];
        hessian[8] = H.mat[3][1];
        hessian[9] = H.mat[3][0];
    }
    inline void write_upper_hessian(float3x3 hessian[10], float12x12& H)
    {

        //   0   4   7   9
        //  t4   1   5   8
        //  t7  t5   2   6
        //  t9  t8  t6   3
        hessian[0] = H.mat[0][0];
        hessian[1] = H.mat[1][1];
        hessian[2] = H.mat[2][2];
        hessian[3] = H.mat[3][3];
        hessian[4] = H.mat[1][0];
        hessian[5] = H.mat[2][1];
        hessian[6] = H.mat[3][2];
        hessian[7] = H.mat[2][0];
        hessian[8] = H.mat[3][1];
        hessian[9] = H.mat[3][0];
    }
    inline void extract_upper_hessian(float3x3 hessian[10], float12x12& H)
    {
        //   0   4   7   9
        //  t4   1   5   8
        //  t7  t5   2   6
        //  t9  t8  t6   3
        H.mat[0][0] = hessian[0];
        H.mat[0][1] = transpose_mat(hessian[4]);
        H.mat[0][2] = transpose_mat(hessian[7]);
        H.mat[0][3] = transpose_mat(hessian[9]);
        H.mat[1][0] = hessian[4];
        H.mat[1][1] = hessian[1];
        H.mat[1][2] = transpose_mat(hessian[5]);
        H.mat[1][3] = transpose_mat(hessian[8]);
        H.mat[2][0] = hessian[7];
        H.mat[2][1] = hessian[5];
        H.mat[2][2] = hessian[2];
        H.mat[2][3] = transpose_mat(hessian[3]);
        H.mat[3][0] = hessian[9];
        H.mat[3][1] = hessian[8];
        H.mat[3][2] = hessian[6];
        H.mat[3][3] = hessian[3];
    }
}  // namespace CollisioinPair
};  // namespace lcs