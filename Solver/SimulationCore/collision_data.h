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

namespace lcs
{

constexpr uint mask_collision_type = 1u << 31;
constexpr uint mask_get_index      = ~(mask_collision_type);

namespace CollisionPair
{
    // clang-format off
    constexpr uint type_vv() { return 1 << 0; }
    constexpr uint type_ve() { return 1 << 1; }
    constexpr uint type_vf() { return 1 << 2; }
    constexpr uint type_ee() { return 1 << 3; }
    // clang-format on

    struct CollisionPairTemplate
    {
        uint4  indices;
        float4 vec1;  // normal:3, area:1
        float4 vec2;  // stiff:2, bary:2

        uint  get_index(const uint i) const { return indices[i] & mask_get_index; }
        uint4 get_indices() const { return indices & luisa::make_uint4(mask_get_index); }

        float3 get_normal() const { return vec1.xyz(); }
        float  get_stiffness() const { return vec1[3]; }

        float         get_k1() const { return vec2[0]; }
        float         get_k2() const { return vec2[1]; }
        luisa::float2 get_stiff() const { return vec2.xy(); }

        // clang-format off
        uint get_collision_type() const
        {
            return ((indices[0] & lcs::mask_collision_type) >> 31)
                 + ((indices[1] & lcs::mask_collision_type) >> 30)
                 + ((indices[2] & lcs::mask_collision_type) >> 29)
                 + ((indices[3] & lcs::mask_collision_type) >> 28);
        }
        void set_collision_type(const uint type)
        {
            if      (type == type_vf()) { indices[2] |= mask_collision_type; }
            else if (type == type_ee()) { indices[3] |= mask_collision_type; }
            else if (type == type_vv()) [[unlikely]] { indices[0] |= mask_collision_type; } 
            else if (type == type_ve()) [[unlikely]] { indices[1] |= mask_collision_type; }
        }

        void make_vv_pair(const uint2& vids, const float3& normal, const float k1, const float k2, const float area)
        {
            indices = luisa::make_uint4(vids.x, -1u, vids.y, -1u);
            vec1 = luisa::make_float4(normal, area);
            vec2 = luisa::make_float4(k1, k2, 1.0f, 1.0f);
            set_collision_type(type_vv());
        }
        void make_ve_pair(const uint3& vids, const float3& normal, const float k1, const float k2, const float area, const float2& edge_bary)
        {
            indices = luisa::make_uint4(vids.x, -1u, vids.y, vids.z);
            vec1 = luisa::make_float4(normal, area);
            vec2 = luisa::make_float4(k1, k2, edge_bary[0], edge_bary[1]);
            set_collision_type(type_ve());
        }
        void make_vf_pair(const uint4& vids, const float3& normal, const float k1, const float k2, const float area, const float3& face_bary)
        {
            indices = vids;
            vec1 = luisa::make_float4(normal, area);
            vec2 = luisa::make_float4(k1, k2, face_bary[0], face_bary[1]);
            set_collision_type(type_vf());
        }
        void make_ee_pair(const uint4& vids, const float3& normal, const float k1, const float k2, const float area, const float2& edge1_bary, const float2& edge2_bary)
        {
            indices = vids;
            vec1 = luisa::make_float4(normal, area);
            vec2 = luisa::make_float4(k1, k2, edge1_bary[0], edge2_bary[0]);
            set_collision_type(type_ee());
        }
        
        [[nodiscard]] float4 get_vv_weight() const { return luisa::make_float4(1.0f, 0.0f, -1.0f, 0.0f); }
        [[nodiscard]] float4 get_ve_weight() const { return luisa::make_float4(1.0f, 0.0f, -vec2[2], -vec2[3]); }
        [[nodiscard]] float4 get_vf_weight() const { return luisa::make_float4(1.0f, -vec2[2], -vec2[3], vec2[2] + vec2[3] - 1.0f); }
        [[nodiscard]] float4 get_ee_weight() const { return luisa::make_float4(vec2[2], 1.0f - vec2[2], -vec2[3], vec2[3] - 1.0f); }
        
        [[nodiscard]] float4 get_weight() const
        {
            uint type = get_collision_type();
            if      (type == type_vf()) { return get_vf_weight(); }
            else if (type == type_ee()) { return get_ee_weight(); }
            else if (type == type_vv()) { return get_vv_weight(); } 
            else if (type == type_ve()) { return get_ve_weight(); }
            else [[unlikely]] { return luisa::make_float4(0.0f); }
        }
        // clang-format on
    };
}  // namespace CollisionPair
};  // namespace lcs

// clang-format off
LUISA_STRUCT(lcs::CollisionPair::CollisionPairTemplate, indices, vec1, vec2)
{
    luisa::compute::Var<uint> get_collision_type() const
    {
        return ((indices[0] & lcs::mask_collision_type) >> 31) + 
               ((indices[1] & lcs::mask_collision_type) >> 30) + 
               ((indices[2] & lcs::mask_collision_type) >> 29) + 
               ((indices[3] & lcs::mask_collision_type) >> 28);
    }
    void set_collision_type(const luisa::compute::Var<uint>& type)
    {
        indices[0] |= ((type >> 0) & 1u) << 31;
        indices[1] |= ((type >> 1) & 1u) << 31;
        indices[2] |= ((type >> 2) & 1u) << 31;
        indices[3] |= ((type >> 3) & 1u) << 31;
    }
    void make_vv_pair(const luisa::compute::UInt2&  vids,
                    const luisa::compute::Float3& normal,
                    const luisa::compute::Float   k1,
                    const luisa::compute::Float   k2,
                    const luisa::compute::Float   area)
    {
        indices = luisa::compute::make_uint4(vids.x, -1u, vids.y, -1u);
        vec1    = luisa::compute::make_float4(normal, area);
        vec2    = luisa::compute::make_float4(k1, k2, 1.0f, 1.0f);
        set_collision_type(lcs::CollisionPair::type_vv());
    }
    void make_ve_pair(const luisa::compute::UInt3&  vids,
                    const luisa::compute::Float3& normal,
                    const luisa::compute::Float   k1,
                    const luisa::compute::Float   k2,
                    const luisa::compute::Float   area,
                    const luisa::compute::Float2& edge_bary)
    {
        indices = luisa::compute::make_uint4(vids.x, -1u, vids.y, vids.z);
        vec1    = luisa::compute::make_float4(normal, area);
        vec2    = luisa::compute::make_float4(k1, k2, edge_bary[0], edge_bary[1]);
        set_collision_type(lcs::CollisionPair::type_ve());
    }
    void make_vf_pair(const luisa::compute::UInt4&  vids,
                    const luisa::compute::Float3& normal,
                    const luisa::compute::Float   k1,
                    const luisa::compute::Float   k2,
                    const luisa::compute::Float   area,
                    const luisa::compute::Float3& face_bary)
    {
        indices = vids;
        vec1    = luisa::compute::make_float4(normal, area);
        vec2    = luisa::compute::make_float4(k1, k2, face_bary[0], face_bary[1]);
        set_collision_type(lcs::CollisionPair::type_vf());
    }
    void make_ee_pair(const luisa::compute::UInt4&  vids,
                    const luisa::compute::Float3& normal,
                    const luisa::compute::Float   k1,
                    const luisa::compute::Float   k2,
                    const luisa::compute::Float   area,
                    const luisa::compute::Float2& edge1_bary,
                    const luisa::compute::Float2& edge2_bary)
    {
        indices = vids;
        vec1    = luisa::compute::make_float4(normal, area);
        vec2    = luisa::compute::make_float4(k1, k2, edge1_bary[0], edge2_bary[0]);
        set_collision_type(lcs::CollisionPair::type_ee());
    }

    luisa::compute::UInt get_index(const uint i) const { return indices[i] & lcs::mask_get_index; }
    luisa::compute::UInt get_index(const luisa::compute::Uint& i) const { return indices[i] & lcs::mask_get_index; }
    luisa::compute::UInt4 get_indices() const { return indices & luisa::compute::make_uint4(lcs::mask_get_index); }
    
    void get_active_indices(
        luisa::compute::ArrayVar<uint, 8>& active_indices,
        luisa::compute::UInt& left_active_count,
        luisa::compute::UInt& right_active_count,
        const luisa::compute::UInt2& body_13_indices, 
        const luisa::compute::UInt& abd_prefix) const 
    { 
        const auto type = get_collision_type();
        luisa::compute::UInt4 orig_indices = get_indices();

        left_active_count = 0;
        right_active_count = 0;
        $if(body_13_indices[0] == -1u) // Left is soft
        {
            $if(type == lcs::CollisionPair::type_vf() | 
                type == lcs::CollisionPair::type_vv() | 
                type == lcs::CollisionPair::type_ve())
            {
                active_indices[left_active_count] = orig_indices[0] | (0 << 30);
                left_active_count += 1;
            }
            $else
            {
                active_indices[left_active_count + 0] = orig_indices[0] | (0 << 30);
                active_indices[left_active_count + 1] = orig_indices[1] | (1 << 30);
                left_active_count += 2;
            };
        }
        $else
        {
            active_indices[left_active_count + 0] = (abd_prefix + 4 * body_13_indices[0] + 0) | (0 << 28);
            active_indices[left_active_count + 1] = (abd_prefix + 4 * body_13_indices[0] + 1) | (1 << 28);
            active_indices[left_active_count + 2] = (abd_prefix + 4 * body_13_indices[0] + 2) | (2 << 28);
            active_indices[left_active_count + 3] = (abd_prefix + 4 * body_13_indices[0] + 3) | (3 << 28);
            left_active_count += 4;
        };

        right_active_count = left_active_count;

        $if(body_13_indices[1] == -1u) // Right is soft
        {
            $if(type == lcs::CollisionPair::type_vv())
            {
                active_indices[right_active_count] = orig_indices[2] | (2 << 30);
                right_active_count += 1u;
            }
            $elif(type == lcs::CollisionPair::type_ve() | 
                  type == lcs::CollisionPair::type_ee())
            {
                active_indices[right_active_count + 0] = orig_indices[2] | (2 << 30);
                active_indices[right_active_count + 1] = orig_indices[3] | (3 << 30);
                right_active_count += 2u;
            }
            $else
            {
                active_indices[right_active_count + 0] = orig_indices[1] | (1 << 30);
                active_indices[right_active_count + 1] = orig_indices[2] | (2 << 30);
                active_indices[right_active_count + 2] = orig_indices[3] | (3 << 30);
                right_active_count += 3u;
            };
        }
        $else
        {
            active_indices[right_active_count + 0] = (abd_prefix + 4 * body_13_indices[1] + 0) | (0 << 28);
            active_indices[right_active_count + 1] = (abd_prefix + 4 * body_13_indices[1] + 1) | (1 << 28);
            active_indices[right_active_count + 2] = (abd_prefix + 4 * body_13_indices[1] + 2) | (2 << 28);
            active_indices[right_active_count + 3] = (abd_prefix + 4 * body_13_indices[1] + 3) | (3 << 28);
            right_active_count += 4;
        };

        right_active_count -= left_active_count;
    }
    luisa::compute::UInt4 get_active_indices(const luisa::compute::UInt2& body_13_indices, luisa::compute::UInt& dofs) const 
    { 
        const auto type = get_collision_type();
        luisa::compute::UInt4 result = get_indices();
        
        $if(body_13_indices[0] == -1u & body_13_indices[1] == -1u) // Soft-Soft
        {
            dofs = 4;
        }
        $elif(body_13_indices[0] == -1u & body_13_indices[1] != -1u) // Soft-Rigid
        {
            $if(type == lcs::CollisionPair::type_vf())
            {
                result[1] = body_13_indices[1];
                dofs = 2;
            }
            $elif(type == lcs::CollisionPair::type_ee())
            {
                result[2] = body_13_indices[1];
                dofs = 3;
            };
        }
        $elif(body_13_indices[0] != -1u & body_13_indices[1] == -1u)
        {
            $if(type == lcs::CollisionPair::type_vf())
            {
                result[0] = body_13_indices[0];
                dofs = 4;
            }
            $elif(type == lcs::CollisionPair::type_ee())
            {
                result[0] = body_13_indices[0];
                result[1] = body_13_indices[0];
                dofs = 3;
            };
        }
        $else // Rigid-Rigid
        {
            result[0] = body_13_indices[0];
            result[1] = body_13_indices[1];
            dofs = 2;
        };
        return result; 
    }

    luisa::compute::Float3 get_normal() const { return vec1.xyz(); }
    luisa::compute::Float get_stiffness() const { return vec1[3]; }

    luisa::compute::Float get_k1() const { return vec2[0]; }
    luisa::compute::Float get_k2() const { return vec2[1]; }
    luisa::compute::Float2 get_stiff() const { return vec2.xy(); }

    luisa::compute::Float4 get_vv_weight() const { return luisa::compute::make_float4(1.0f, 0.0f, -1.0f, 0.0f); }
    luisa::compute::Float4 get_ve_weight() const { return luisa::compute::make_float4(1.0f, 0.0f, -vec2[2], -vec2[3]); }
    luisa::compute::Float4 get_vf_weight() const { return luisa::compute::make_float4(1.0f, -vec2[2], -vec2[3], vec2[2] + vec2[3] - 1.0f); }
    luisa::compute::Float4 get_ee_weight() const { return luisa::compute::make_float4(vec2[2], 1.0f - vec2[2], -vec2[3], vec2[3] - 1.0f); }

    luisa::compute::Float4 get_weight() const
    {
        luisa::compute::UInt   type   = get_collision_type();
        luisa::compute::Float4 result = luisa::compute::make_float4(0.0f);
        $if(  type == lcs::CollisionPair::type_vf()) { result = get_vf_weight(); }
        $elif(type == lcs::CollisionPair::type_ee()) { result = get_ee_weight(); }
        $elif(type == lcs::CollisionPair::type_vv()) { result = get_vv_weight(); }
        $elif(type == lcs::CollisionPair::type_ve()) { result = get_ve_weight(); };
        return result;
    }
};
// clang-format on

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
        inline constexpr uint total_adj_pairs_offset() { return 4; }
        inline constexpr uint total_adj_verts_offset() { return 5; }
        // clang-format on
    }  // namespace CollisionCount
}  // namespace CollisionPair


// template <typename T>
// using BufferType = std::vector<T>;
template <template <typename...> typename BufferType>
struct CollisionData : SimulationType
{
    BufferType<uint> broad_phase_collision_count;   // 0: VV, 1: VE, 2: VF, 3: EE
    BufferType<uint> narrow_phase_collision_count;  // 0: VV, 1: VE, 2: VF, 3: EE
                                                    // 4: PerVertVV, 5: PerVertVE,
                                                    // 6: PerVertVF, 6: PervertEE

    BufferType<uint>  broad_phase_list_vf;
    BufferType<uint>  broad_phase_list_ee;
    BufferType<float> toi_per_vert;  // Actually is global min TOI
    BufferType<float> contact_energy;

    BufferType<CollisionPair::CollisionPairTemplate> narrow_phase_list;  // 0
    BufferType<uint>                                 sa_triplet_info;
    BufferType<MatrixTriplet3x3>                     sa_cgA_contact_offdiag_triplet;
    BufferType<uint2>                                sa_cgA_contact_offdiag_triplet_indices;
    BufferType<uint2>                                sa_cgA_contact_offdiag_triplet_indices2;
    BufferType<uint>                                 sa_cgA_contact_offdiag_triplet_property;
    BufferType<uint>                                 sa_cgA_contact_offdiag_triplet_property2;

    // BufferType<ReducedCollisionPairInfo> reduced_narrow_phase_list_info_vf;
    // BufferType<ReducedCollisionPairInfo> reduced_narrow_phase_list_info_ee;
    // BufferType<uint> narrow_phase_indices_ef;

    BufferType<uint> per_vert_num_broad_phase_vf;
    BufferType<uint> per_vert_num_broad_phase_ee;

    BufferType<uint> per_vert_num_adj_pairs;
    BufferType<uint> per_vert_num_adj_verts;
    BufferType<uint> per_vert_prefix_adj_pairs;
    BufferType<uint> per_vert_prefix_adj_verts;

    // luisa::compute::IndirectDispatchBuffer collision_indirect_cmd_buffer_broad_phase;
    // luisa::compute::IndirectDispatchBuffer collision_indirect_cmd_buffer_narrow_phase;


    const uint get_vv_count_offset() { return 0; }
    const uint get_ve_count_offset() { return 1; }
    const uint get_vf_count_offset() { return 2; }
    const uint get_ee_count_offset() { return 3; }

    // template<template<typename...> typename BufferType>
    inline void allocate_basic_buffers(luisa::compute::Device& device,
                                       const uint              num_verts,
                                       const uint              num_faces,
                                       const uint              num_edges,
                                       const uint              num_dofs)
    {
        constexpr bool use_vv_ve = false;
        lcs::Initializer::resize_buffer(device, this->broad_phase_collision_count, 4);
        lcs::Initializer::resize_buffer(device, this->narrow_phase_collision_count, 8);
        lcs::Initializer::resize_buffer(device, this->contact_energy, 4);
        lcs::Initializer::resize_buffer(device, this->toi_per_vert, num_verts);

        lcs::Initializer::resize_buffer(device, this->per_vert_num_broad_phase_vf, num_dofs);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_broad_phase_ee, num_dofs);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_adj_pairs, num_dofs);
        lcs::Initializer::resize_buffer(device, this->per_vert_num_adj_verts, num_dofs);
        lcs::Initializer::resize_buffer(device, this->per_vert_prefix_adj_pairs, num_dofs + 1);
        lcs::Initializer::resize_buffer(device, this->per_vert_prefix_adj_verts, num_dofs + 1);
    }

    inline void resize_collision_data_list(luisa::compute::Device& device,
                                           const uint              num_verts,
                                           const uint              num_faces,
                                           const uint              num_edges,
                                           const uint              num_dofs,
                                           const bool              allocate_contact_list,
                                           const bool              allocate_triplet)
    {
        const uint per_element_count_BP        = 48;
        const uint per_element_count_NP        = 24;
        const uint per_element_count_NP_culled = 12;

        if (allocate_contact_list)
        {
            lcs::Initializer::resize_buffer(device, this->broad_phase_list_vf, per_element_count_BP * num_verts);
            lcs::Initializer::resize_buffer(device, this->broad_phase_list_ee, per_element_count_BP * num_edges);
            // lcs::Initializer::resize_buffer(device, this->narrow_phase_list_vv, per_element_count_NP * num_verts);
            // lcs::Initializer::resize_buffer(device, this->narrow_phase_list_ve, per_element_count_NP * num_verts);

            const uint max_pairs   = per_element_count_NP * (num_verts + num_edges);
            const uint max_triplet = max_pairs * 12;  // 12 off-diagonal

            lcs::Initializer::resize_buffer(device, this->narrow_phase_list, max_pairs);
            lcs::Initializer::resize_buffer(device, this->sa_triplet_info, max_pairs * 4);

            lcs::Initializer::resize_buffer(device, this->sa_cgA_contact_offdiag_triplet_indices, max_triplet);
            lcs::Initializer::resize_buffer(device, this->sa_cgA_contact_offdiag_triplet_indices2, max_triplet);
            lcs::Initializer::resize_buffer(device, this->sa_cgA_contact_offdiag_triplet_property, max_triplet);
            lcs::Initializer::resize_buffer(device, this->sa_cgA_contact_offdiag_triplet_property2, max_triplet);
        }

        if (allocate_triplet)
        {
            const uint max_culled_triplet = per_element_count_NP_culled * (num_verts + num_edges);
            lcs::Initializer::resize_buffer(
                device, this->sa_cgA_contact_offdiag_triplet, max_scalar(max_culled_triplet, 256));
        }

        const size_t collision_pair_bytes =
            sizeof(uint) * this->broad_phase_list_vf.size() + sizeof(uint) * this->broad_phase_list_ee.size()
            + sizeof(CollisionPair::CollisionPairTemplate) * this->narrow_phase_list.size()
            + sizeof(uint) * this->sa_triplet_info.size()
            + sizeof(MatrixTriplet3x3) * this->sa_cgA_contact_offdiag_triplet.size()
            + sizeof(uint2) * this->sa_cgA_contact_offdiag_triplet_indices.size()
            + sizeof(uint2) * this->sa_cgA_contact_offdiag_triplet_indices2.size()
            + sizeof(uint) * this->sa_cgA_contact_offdiag_triplet_property.size()
            + sizeof(uint) * this->sa_cgA_contact_offdiag_triplet_property2.size();

        LUISA_INFO("Allocated collision buffer size {} MB", collision_pair_bytes / (1024 * 1024));
        if (float(collision_pair_bytes) / (1024 * 1024 * 1024) > 1.0f)
            LUISA_INFO("Allocated buffer size for collision pair = {} GB",
                       float(collision_pair_bytes) / (1024 * 1024 * 1024));
    }
};


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