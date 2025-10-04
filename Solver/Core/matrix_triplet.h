#pragma once

#include <array>
#include <luisa/core/basic_types.h>
#include <luisa/dsl/struct.h>

namespace lcs
{
struct MatrixTriplet3x3
{
    std::array<uint, 3>  triplet_info;
    std::array<float, 9> values;  // column major

    const uint get_row_idx() const { return triplet_info[0]; }
    const uint get_col_idx() const { return triplet_info[1]; }
    const uint get_matrix_property() const { return triplet_info[2]; }

    const luisa::float3x3 get_matrix() const
    {
        return luisa::make_float3x3(
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]);
        ;
    }
};
};  // namespace lcs

// clang-format off
LUISA_STRUCT(lcs::MatrixTriplet3x3, triplet_info, values) 
{ 
    const luisa::compute::Var<uint> get_row_idx() const { return triplet_info[0]; }
    const luisa::compute::Var<uint> get_col_idx() const { return triplet_info[1]; }
    const luisa::compute::Var<uint> get_matrix_property() const { return triplet_info[2]; }
    
    const luisa::compute::Var<luisa::float3x3> get_matrix() const
    {
        return luisa::compute::make_float3x3(
            values[0], values[1], values[2], 
            values[3], values[4], values[5], 
            values[6], values[7], values[8]);
        ;
    }
};
// clang-format on

namespace lcs
{

namespace MatrixTriplet
{
    constexpr uint is_first_col_in_row()
    {
        return 1 << 0;
    }
    constexpr uint is_last_col_in_row()
    {
        return 1 << 1;
    }
    constexpr uint write_use_atomic()
    {
        return 1 << 2;
    }
    constexpr uint is_invalid()
    {
        return 1 << 3;
    }
    template <typename T>
    auto is_first_col_in_row(const T& mask)
    {
        return (mask & is_first_col_in_row()) != 0;
    }
    template <typename T>
    auto is_last_col_in_row(const T& mask)
    {
        return (mask & is_last_col_in_row()) != 0;
    }
    template <typename T>
    auto is_invalid(const T& mask)
    {
        return (mask & is_invalid()) != 0;
    }
    template <typename T>
    auto write_use_atomic(const T& mask)
    {
        return (mask & write_use_atomic()) != 0;
    }

    template <typename T>
    inline T write_lane_id_of_first_colIdx_in_warp_to_mask(const T lane_id)
    {
        return lane_id << 8;
    }
    template <typename T>
    inline T read_lane_id_of_first_colIdx_in_warp(const T matrix_info)
    {
        return (matrix_info >> 8) & 0xFF;
    }

};  // namespace MatrixTriplet


inline luisa::uint3 make_matrix_triplet_info(const uint row, const uint col, const uint matrix_property)
{
    return luisa::compute::make_uint3(row, col, matrix_property);
}
inline luisa::compute::UInt3 make_matrix_triplet_info(const luisa::compute::Var<uint> row,
                                                      const luisa::compute::Var<uint> col,
                                                      const luisa::compute::Var<uint> matrix_property)
{
    return luisa::compute::make_uint3(row, col, matrix_property);
}
inline MatrixTriplet3x3 make_matrix_triplet(const uint row, const uint col, const uint matrix_property, const luisa::float3x3& values)
{
    MatrixTriplet3x3 triplet;
    triplet.triplet_info[0] = row;
    triplet.triplet_info[1] = col;
    triplet.triplet_info[2] = matrix_property;
    triplet.values[0]       = values[0][0];
    triplet.values[1]       = values[0][1];
    triplet.values[2]       = values[0][2];
    triplet.values[3]       = values[1][0];
    triplet.values[4]       = values[1][1];
    triplet.values[5]       = values[1][2];
    triplet.values[6]       = values[2][0];
    triplet.values[7]       = values[2][1];
    triplet.values[8]       = values[2][2];
    return triplet;
}
inline luisa::compute::Var<MatrixTriplet3x3> make_matrix_triplet(const luisa::compute::Var<uint> row,
                                                                 const luisa::compute::Var<uint> col,
                                                                 const luisa::compute::Var<uint> matrix_property,
                                                                 const luisa::compute::Var<luisa::float3x3>& values)
{
    luisa::compute::Var<MatrixTriplet3x3> triplet;
    triplet.triplet_info[0] = row;
    triplet.triplet_info[1] = col;
    triplet.triplet_info[2] = matrix_property;
    triplet.values[0]       = values[0][0];
    triplet.values[1]       = values[0][1];
    triplet.values[2]       = values[0][2];
    triplet.values[3]       = values[1][0];
    triplet.values[4]       = values[1][1];
    triplet.values[5]       = values[1][2];
    triplet.values[6]       = values[2][0];
    triplet.values[7]       = values[2][1];
    triplet.values[8]       = values[2][2];
    return triplet;
}


inline luisa::float3x3 read_triplet_matrix(const MatrixTriplet3x3& triplet)
{
    return luisa::make_float3x3(triplet.values[0],
                                triplet.values[1],
                                triplet.values[2],
                                triplet.values[3],
                                triplet.values[4],
                                triplet.values[5],
                                triplet.values[6],
                                triplet.values[7],
                                triplet.values[8]);
}
inline luisa::compute::Var<luisa::float3x3> read_triplet_matrix(const luisa::compute::Var<MatrixTriplet3x3>& triplet)
{
    return luisa::compute::make_float3x3(triplet.values[0],
                                         triplet.values[1],
                                         triplet.values[2],
                                         triplet.values[3],
                                         triplet.values[4],
                                         triplet.values[5],
                                         triplet.values[6],
                                         triplet.values[7],
                                         triplet.values[8]);
}


inline void write_triplet_matrix(MatrixTriplet3x3& triplet, const luisa::float3x3& values)
{
    triplet.values[0] = values[0][0];
    triplet.values[1] = values[0][1];
    triplet.values[2] = values[0][2];
    triplet.values[3] = values[1][0];
    triplet.values[4] = values[1][1];
    triplet.values[5] = values[1][2];
    triplet.values[6] = values[2][0];
    triplet.values[7] = values[2][1];
    triplet.values[8] = values[2][2];
}
inline void write_triplet_matrix(luisa::compute::Var<MatrixTriplet3x3>&      triplet,
                                 const luisa::compute::Var<luisa::float3x3>& values)
{
    triplet.values[0] = values[0][0];
    triplet.values[1] = values[0][1];
    triplet.values[2] = values[0][2];
    triplet.values[3] = values[1][0];
    triplet.values[4] = values[1][1];
    triplet.values[5] = values[1][2];
    triplet.values[6] = values[2][0];
    triplet.values[7] = values[2][1];
    triplet.values[8] = values[2][2];
}


inline void add_triplet_matrix(luisa::compute::Var<MatrixTriplet3x3>&      triplet,
                               const luisa::compute::Var<luisa::float3x3>& values)
{
    triplet.values[0] += values[0][0];
    triplet.values[1] += values[0][1];
    triplet.values[2] += values[0][2];
    triplet.values[3] += values[1][0];
    triplet.values[4] += values[1][1];
    triplet.values[5] += values[1][2];
    triplet.values[6] += values[2][0];
    triplet.values[7] += values[2][1];
    triplet.values[8] += values[2][2];
}
inline void add_triplet_matrix(MatrixTriplet3x3& triplet, const luisa::float3x3& values)
{
    triplet.values[0] += values[0][0];
    triplet.values[1] += values[0][1];
    triplet.values[2] += values[0][2];
    triplet.values[3] += values[1][0];
    triplet.values[4] += values[1][1];
    triplet.values[5] += values[1][2];
    triplet.values[6] += values[2][0];
    triplet.values[7] += values[2][1];
    triplet.values[8] += values[2][2];
}
inline void atomic_add_triplet_matrix(luisa::compute::Var<luisa::compute::Buffer<MatrixTriplet3x3>>& triplet,
                                      const luisa::compute::UInt                  index,
                                      const luisa::compute::Var<luisa::float3x3>& values)
{
    triplet.atomic(index).values[0].fetch_add(values[0][0]);
    triplet.atomic(index).values[1].fetch_add(values[0][1]);
    triplet.atomic(index).values[2].fetch_add(values[0][2]);
    triplet.atomic(index).values[3].fetch_add(values[1][0]);
    triplet.atomic(index).values[4].fetch_add(values[1][1]);
    triplet.atomic(index).values[5].fetch_add(values[1][2]);
    triplet.atomic(index).values[6].fetch_add(values[2][0]);
    triplet.atomic(index).values[7].fetch_add(values[2][1]);
    triplet.atomic(index).values[8].fetch_add(values[2][2]);
}


};  // namespace lcs