#pragma once

#include "Core/scalar.h"
#include <array>
#include <luisa/core/basic_types.h>
#include <luisa/dsl/struct.h>
#include <luisa/dsl/sugar.h>

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
    constexpr uint is_valid()
    {
        return 1 << 3;
    }
    constexpr uint is_first_and_last_col_in_same_warp()
    {
        return 1 << 4;
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
    auto is_valid(const T& mask)
    {
        return (mask & is_valid()) != 0;
    }
    template <typename T>
    auto is_first_and_last_col_in_same_warp(const T& mask)
    {
        return (mask & is_first_and_last_col_in_same_warp()) != 0;
    }
    template <typename T>
    auto write_use_atomic(const T& mask)
    {
        return (mask & write_use_atomic()) != 0;
    }

    template <typename T>
    inline T write_first_col_info(const T lane_id)
    {
        return lane_id << 8;
    }
    template <typename T>
    inline T read_first_col_info(const T matrix_info)
    {
        return (matrix_info >> 8) & 0xFF;
    }

    template <typename T>
    inline T write_first_col_threadIdx(const T lane_id)
    {
        return lane_id << 16;
    }
    template <typename T>
    inline T read_first_col_threadIdx(const T matrix_info)
    {
        return (matrix_info >> 16) & 0xFF;
    }

};  // namespace MatrixTriplet


namespace MatrixTriplet
{
    inline uint make_triplet_property_in_block(const uint idx, const uint first_triplet_idx, const uint last_triplet_idx)
    {
        uint triplet_property = MatrixTriplet::is_valid();

        constexpr uint blockDim = 256;
        const uint     blockIdx = idx / blockDim;

        const uint blockStartPrefix           = blockIdx * 256;
        const uint blockEndPrefix             = blockStartPrefix + 256;
        const uint first_triplet_idx_in_block = max_scalar(blockStartPrefix, first_triplet_idx);
        const uint last_triplet_idx_in_block  = min_scalar(blockEndPrefix - 1, last_triplet_idx);

        if (idx == first_triplet_idx_in_block)
        {
            triplet_property |= (MatrixTriplet::is_first_col_in_row());

            if (idx / 32 == last_triplet_idx_in_block / 32)  // In the same warp -> Provide using warp intrinsic
            {
                triplet_property |= MatrixTriplet::is_first_and_last_col_in_same_warp();
            }
            else  // If not in the same warp -> Write it to the cache
            {
            }
            // if (idx / segment_size == last_triplet_idx / segment_size)  // In the same block -> Try to provide value to the next prefix
            // else  // Not in the same block -> Try to provide value to the last threadIdx
        }
        if (idx == last_triplet_idx_in_block)
        {
            triplet_property |= (MatrixTriplet::is_last_col_in_row());  // Last in row

            if (idx / 32 == first_triplet_idx_in_block / 32)  // In the same warp -> Read using warp intrinsic
            {
                const uint first_lane_id = first_triplet_idx_in_block % 32;
                triplet_property |= MatrixTriplet::is_first_and_last_col_in_same_warp();
                triplet_property |= MatrixTriplet::write_first_col_info(first_lane_id);
            }
            else  // If not in the same warp -> Read from the cache
            {
                const uint first_warp_id = (first_triplet_idx_in_block % blockDim) / 32;
                triplet_property |= MatrixTriplet::write_first_col_info(first_warp_id);
            }
            triplet_property |= MatrixTriplet::write_first_col_threadIdx(first_triplet_idx_in_block % 256);
            // if (idx / segment_size == last_triplet_idx / segment_size)  // In the same block -> Try to get value from the block
            // else  // Not in the same block -> Try to get value from the first threadIdx

            if (first_triplet_idx / blockDim != last_triplet_idx / blockDim)
            {
                triplet_property |= MatrixTriplet::write_use_atomic();
            }
            else
            {
            }
        }
        return triplet_property;
    }
    inline luisa::compute::UInt make_triplet_property_in_block(const luisa::compute::UInt& idx,
                                                               const luisa::compute::UInt& first_triplet_idx,
                                                               const luisa::compute::UInt& last_triplet_idx)
    {
        luisa::compute::UInt triplet_property = MatrixTriplet::is_valid();

        const uint                 blockDim = 256;
        const luisa::compute::UInt blockIdx = idx / blockDim;

        const luisa::compute::UInt blockStartPrefix = blockIdx * 256;
        const luisa::compute::UInt blockEndPrefix   = blockStartPrefix + 256;
        const luisa::compute::UInt first_triplet_idx_in_block = max_scalar(blockStartPrefix, first_triplet_idx);
        const luisa::compute::UInt last_triplet_idx_in_block = min_scalar(blockEndPrefix - 1, last_triplet_idx);

        $if(idx == first_triplet_idx_in_block)
        {
            triplet_property |= (MatrixTriplet::is_first_col_in_row());

            $if(idx / 32 == last_triplet_idx_in_block / 32)
            {
                triplet_property |= MatrixTriplet::is_first_and_last_col_in_same_warp();
            };
        };
        $if(idx == last_triplet_idx_in_block)
        {
            triplet_property |= (MatrixTriplet::is_last_col_in_row());

            $if(idx / 32 == first_triplet_idx_in_block / 32)
            {
                const luisa::compute::UInt first_lane_id = first_triplet_idx_in_block % 32;
                triplet_property |= MatrixTriplet::is_first_and_last_col_in_same_warp();
                triplet_property |= MatrixTriplet::write_first_col_info(first_lane_id);
            }
            $else
            {
                const luisa::compute::UInt first_warp_id = (first_triplet_idx_in_block % blockDim) / 32;
                triplet_property |= MatrixTriplet::write_first_col_info(first_warp_id);
            };
            triplet_property |= MatrixTriplet::write_first_col_threadIdx(first_triplet_idx_in_block % 256);
            $if(first_triplet_idx / blockDim != last_triplet_idx / blockDim)
            {
                triplet_property |= MatrixTriplet::write_use_atomic();
            };
        };
        return triplet_property;
    }
    inline uint make_triplet_property_in_warp(const uint idx, const uint curr_prefix, const uint next_prefix)
    {
        constexpr uint blockDim = 256;
        const uint     blockIdx = idx / blockDim;

        const uint first_triplet_idx = curr_prefix;
        const uint last_triplet_idx  = next_prefix - 1;

        const uint first_triplet_idx_in_block = std::max(blockIdx * blockDim, first_triplet_idx);
        const uint last_triplet_idx_in_block = std::min(blockIdx * blockDim + blockDim - 1, last_triplet_idx);

        uint triplet_property = MatrixTriplet::is_valid();
        if (idx == first_triplet_idx)
        {
            triplet_property |= (MatrixTriplet::is_first_col_in_row());
        }
        if (idx == last_triplet_idx)
        {
            triplet_property |= (MatrixTriplet::is_last_col_in_row());  // Last in row
            if (idx / 32 == curr_prefix / 32)  // In the same warp -> Read the first column
            {
                const uint first_lane_id = curr_prefix % 32;
                triplet_property |= MatrixTriplet::write_first_col_info(first_lane_id);
            }
            else  // Not in the same warp -> Read the first lane
            {
                triplet_property |= MatrixTriplet::write_first_col_info(0);
            }
            if (first_triplet_idx / 32 != last_triplet_idx / 32)
            {
                triplet_property |= MatrixTriplet::write_use_atomic();
            }
        }
        return triplet_property;
    }
}  // namespace MatrixTriplet


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
inline void atomic_add_triplet_matrix(luisa::compute::BufferView<MatrixTriplet3x3> triplet,
                                      const luisa::compute::UInt                   index,
                                      const luisa::compute::Var<luisa::float3x3>&  values)
{
    triplet->atomic(index).values[0].fetch_add(values[0][0]);
    triplet->atomic(index).values[1].fetch_add(values[0][1]);
    triplet->atomic(index).values[2].fetch_add(values[0][2]);
    triplet->atomic(index).values[3].fetch_add(values[1][0]);
    triplet->atomic(index).values[4].fetch_add(values[1][1]);
    triplet->atomic(index).values[5].fetch_add(values[1][2]);
    triplet->atomic(index).values[6].fetch_add(values[2][0]);
    triplet->atomic(index).values[7].fetch_add(values[2][1]);
    triplet->atomic(index).values[8].fetch_add(values[2][2]);
}
inline void atomic_add_triplet_matrix(luisa::compute::BufferVar<MatrixTriplet3x3>& triplet,
                                      const luisa::compute::UInt                   index,
                                      const luisa::compute::Var<luisa::float3x3>&  values)
{
    triplet->atomic(index).values[0].fetch_add(values[0][0]);
    triplet->atomic(index).values[1].fetch_add(values[0][1]);
    triplet->atomic(index).values[2].fetch_add(values[0][2]);
    triplet->atomic(index).values[3].fetch_add(values[1][0]);
    triplet->atomic(index).values[4].fetch_add(values[1][1]);
    triplet->atomic(index).values[5].fetch_add(values[1][2]);
    triplet->atomic(index).values[6].fetch_add(values[2][0]);
    triplet->atomic(index).values[7].fetch_add(values[2][1]);
    triplet->atomic(index).values[8].fetch_add(values[2][2]);
}


};  // namespace lcs