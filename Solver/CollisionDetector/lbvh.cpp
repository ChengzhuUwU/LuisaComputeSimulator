#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/aabb.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"

namespace lcsv
{

// void LBVH::init(luisa::compute::Device& device, luisa::compute::Stream& stream, 
//         const uint input_num, const LBVHTreeType tree_type, const LBVHUpdateType update_type)
// {
//     lbvh_data->allocate(device, input_num, tree_type, update_type);
// }

template<typename UintType>
static inline UintType expand_bits(UintType bits)
{
    bits = (bits | (bits << 16)) & static_cast<UintType>(0x030000FF);
    bits = (bits | (bits << 8))  & static_cast<UintType>(0x0300F00F);
    bits = (bits | (bits << 4))  & static_cast<UintType>(0x030C30C3);
    return (bits | (bits << 2))  & static_cast<UintType>(0x09249249);
}

static inline Var<uint> make_morton32(const luisa::compute::Float3& pos) 
{
    using namespace luisa::compute;
    const Uint precision = 10;
    const Float min_value = 0.0f;
    const Float max_value = (1 << precision) - 1;
    const Float range = 1 << precision;

    Float x = clamp_scalar(pos[0] * range, min_value, max_value);
    Float y = clamp_scalar(pos[1] * range, min_value, max_value);
    Float z = clamp_scalar(pos[2] * range, min_value, max_value);

    Uint xx = expand_bits(static_cast<Uint>(x));
    Uint yy = expand_bits(static_cast<Uint>(y));
    Uint zz = expand_bits(static_cast<Uint>(z));

    return (xx << 2) | (yy << 1) | zz;
}
static inline uint make_morton32(const luisa::float3& pos) 
{
    const uint precision = 10;

    float x = clamp_scalar(pos[0] * (1 << precision), static_cast<float>(0.0f), (1 << precision) - 1.0f);
    float y = clamp_scalar(pos[1] * (1 << precision), static_cast<float>(0.0f), (1 << precision) - 1.0f);
    float z = clamp_scalar(pos[2] * (1 << precision), static_cast<float>(0.0f), (1 << precision) - 1.0f);

    uint xx = expand_bits(static_cast<uint>(x));
    uint yy = expand_bits(static_cast<uint>(y));
    uint zz = expand_bits(static_cast<uint>(z));

    return (xx << 2) | (yy << 1) | zz;
}

static inline Morton64 make_morton64(const luisa::compute::Float3& pos, const luisa::compute::Uint index) 
{
    return (static_cast<Morton64>(make_morton32(pos)) << 32) | (static_cast<Morton64>(index) & static_cast<Morton64>(0xFFFFFFFF));
}
static inline morton64 make_morton64(const luisa::float3& pos, const uint index) 
{
    return (static_cast<morton64>(make_morton32(pos)) << 32) | (static_cast<morton64>(index) & static_cast<morton64>(0xFFFFFFFF));
}

// Existing functions
template<typename UintType> UintType make_leaf(const UintType mask)     { return (mask) | static_cast<UintType>(1 << 31); }
template<typename UintType> UintType is_leaf(const UintType mask)       { return (mask & static_cast<UintType>(1 << 31)) != 0; }
template<typename UintType> UintType extract_leaf(const UintType mask)  { return (mask) & (~(static_cast<UintType>(1 << 31))); }


int clz_ulong(morton64 x) 
{
    return __builtin_clzll(x);
}
Var<int> clz_ulong(Var<morton64> x) 
{
    Var<int> count = 0;
    $while (x != Var<morton64>(0)) 
    {
        count += 1;
        x >>= 1;
    };
    return Var<int>(64 - count);
}

template<typename MortonType>
auto find_common_prefix(const MortonType& left, const MortonType& right) 
{
    return clz_ulong(left ^ right);
}

inline Var<morton64> get_morton(const luisa::compute::BufferView<morton64>& buffer, const Var<uint> index){ return buffer->read(index); }
inline Var<morton64> get_morton(const luisa::compute::BufferView<morton64>& buffer, const Var<int> index) { return buffer->read(index); }
inline morton64 get_morton(const std::vector<morton64>& buffer, const int index){ return buffer[index]; }


template<typename MortonType, typename UintType, typename IntType, typename BufferType>
IntType cp_i_j(const MortonType& mi, IntType j, const BufferType& sa_morton_sorted, const UintType num_leaves) 
{
    auto isValid = (j >= 0 & static_cast<UintType>(j) < num_leaves);
    return select(isValid, find_common_prefix(mi, get_morton(sa_morton_sorted, j)), static_cast<IntType>(-1));
}

// template<template<typename> typename T, typename BufferType>
// T<int> cp_i_j(const T<morton64>& mi, T<int> j, const BufferType& sa_morton_sorted, const T<uint> num_leaves) 
// {
//     auto isValid = (j >= 0 & j < num_leaves);
//     return select(isValid, find_common_prefix(mi, get_morton(sa_morton_sorted, j)), T<int>(-1));
// }

Var<int2> determineRange(const Var<uint> index, const luisa::compute::BufferView<morton64>& sa_morton_sorted, const Var<uint> num_leaves) 
{
    using IndexType = Var<int>;
    IndexType i = index;
    auto mi = get_morton(sa_morton_sorted, i);
    auto cp_left  = find_common_prefix(mi, get_morton(sa_morton_sorted, i - 1));
    auto cp_right = find_common_prefix(mi, get_morton(sa_morton_sorted, i + 1));

    IndexType d = select(cp_left < cp_right, IndexType(1), IndexType(-1));
    IndexType cp_min = min_scalar(cp_left, cp_right);

    IndexType lmax = 2;
    $while (cp_i_j(mi, i + lmax * d, sa_morton_sorted, num_leaves) > cp_min) 
    {
        lmax <<= 1;
    };

    // for index 1 : d = -1 , cp_left = 11, cp_right =  8, cp_min =  8 , lmax = 2
    // for index 2 : d = -1 , cp_left =  8, cp_right =  5, cp_min =  5 , lmax = 4
    // for index 3 : d =  1 , cp_left =  5, cp_right =  8, cp_min =  5 , lmax = 2
    // for index 4 : d = -1 , cp_left =  8, cp_right =  2, cp_min =  2 , lmax = 8
    // for index 5 : d =  1 , cp_left =  2, cp_right = 11, cp_min =  2 , lmax = 4
    // for index 6 : d = -1 , cp_left = 11, cp_right =  8, cp_min =  8 , lmax = 2
    // luisa::compute::device_log("for index {} : d = {} , cp_left = {}, cp_right = {}, cp_min = {} , lmax = {}", index, d, cp_left, cp_right, cp_min, lmax);

    IndexType l = 0;
    IndexType t = lmax >> 1;
    $while (t >= 1) 
    {
        $if (cp_i_j(mi, i + (l + t) * d, sa_morton_sorted, num_leaves) > cp_min) 
        {
            l += t;
        };
        t >>= 1;
    };

    IndexType j = i + l * d;
    return makeInt2(i, j);
}

Var<int> findSplit(const Var<int2>& ranges, const luisa::compute::BufferView<morton64>& sa_morton_sorted) 
{
    using IndexType = Var<int>;

    IndexType d = select(ranges[0] < ranges[1], static_cast<IndexType>(1), static_cast<IndexType>(-1));
    IndexType i = ranges[0];
    IndexType j = ranges[1];
    $if (d < 0) { swap_scalar(i, j); };

    Morton64 mi = get_morton(sa_morton_sorted, i);
    Morton64 mj = get_morton(sa_morton_sorted, j);
    IndexType cp_node = find_common_prefix(mi, mj);

    IndexType split = 0;
    $if (mi == mj) {
        split = (i + j) >> 1;
    }
    $else {
        IndexType t = j - i;
        split = i;
        $while (true) {
            t = (t + 1) >> 1;
            IndexType newSplit = split + t;
            Morton64 ms = get_morton(sa_morton_sorted, newSplit);
            IndexType cp_split = find_common_prefix(mi, ms);
            $if (cp_split > cp_node) {
                split = newSplit;
            };
            $if (!(t > 1)) { $break; };
        };
    };
    return split;
}

void LBVH::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    const uint num_leaves = lbvh_data->num_leaves;
    const uint num_inner_nodes = lbvh_data->num_inner_nodes;
    const uint num_nodes = lbvh_data->num_nodes;

    // Construct
    auto reduce_aabb_1_pass_template = [
        sa_block_aabb = lbvh_data->sa_block_aabb.view(),
        sa_leaf_center = lbvh_data->sa_leaf_center.view()
    ](
        const Float3& center,
        const Float2x3& aabb
    )
    {
        luisa::compute::set_block_size(256);
        const Uint vid = luisa::compute::dispatch_id().x;

        Var<float3>  min_vec = AABB::get_aabb_min(aabb);
        Var<float3>  max_vec = AABB::get_aabb_max(aabb);
        min_vec = ParallelIntrinsic::block_intrinsic_reduce(vid, min_vec, ParallelIntrinsic::warp_reduce_op_min<float3>);
        max_vec = ParallelIntrinsic::block_intrinsic_reduce(vid, max_vec, ParallelIntrinsic::warp_reduce_op_max<float3>);
        auto reduced_aabb = AABB::make_aabb(min_vec, max_vec);

        $if (vid % 256 == 0)
        {
            const Uint blockIdx = vid / ParallelIntrinsic::reduce_block_dim;
            sa_block_aabb->write(blockIdx, reduced_aabb);
        };
    };

    fn_reduce_aabb_2_pass = device.compile<1>([
        sa_block_aabb = lbvh_data->sa_block_aabb.view()
    ]()
    {
        luisa::compute::set_block_size(256);
        const Uint vid = luisa::compute::dispatch_id().x;

        auto aabb = sa_block_aabb->read(vid);

        // Float2x3 reduced_aabb = ParallelIntrinsic::block_reduce(vid, aabb, AABB::reduce_aabb);

        Float3 min_vec = AABB::get_aabb_min(aabb);
        Float3 max_vec = AABB::get_aabb_max(aabb);
        min_vec = ParallelIntrinsic::block_intrinsic_reduce(vid, min_vec, ParallelIntrinsic::warp_reduce_op_min<float3>);
        max_vec = ParallelIntrinsic::block_intrinsic_reduce(vid, max_vec, ParallelIntrinsic::warp_reduce_op_max<float3>);
        Float2x3 reduced_aabb = AABB::make_aabb(min_vec, max_vec);

        $if (vid % 256 == 0)
        {
            const Uint blockIdx = vid / 256;
            sa_block_aabb->write(blockIdx, reduced_aabb);
        };
    });
    
    fn_reduce_vert_tree_global_aabb = device.compile<1>([
        &reduce_aabb_1_pass_template
    ]
    (
        const Var<luisa::compute::BufferView<float3>> input_position
    )
    {
        const Uint vid = luisa::compute::dispatch_id().x;
        Float3 vert_pos = input_position->read(vid);
        Float2x3 aabb = AABB::make_aabb(vert_pos);

        reduce_aabb_1_pass_template(vert_pos, aabb);
    });

    fn_reduce_edge_tree_global_aabb = device.compile<1>([
        &reduce_aabb_1_pass_template
    ]
    (
        const Var<luisa::compute::BufferView<float3>> input_position,
        const Var<luisa::compute::BufferView<uint2>> input_edge
    )
    {
        luisa::compute::set_block_size(256);
        
        const Uint fid = luisa::compute::dispatch_id().x;
        const UInt2 edge = input_edge.read(fid);
        Float3 positions[2] = {
            input_position->read(edge[0]),
            input_position->read(edge[1])
        };

        Float3 center = 0.5f * (positions[0] + positions[1]);
        Float2x3 aabb = AABB::make_aabb(
            positions[0],
            positions[1]
        );
        reduce_aabb_1_pass_template(center, aabb);
    });

    fn_reduce_face_tree_global_aabb = device.compile<1>([
        &reduce_aabb_1_pass_template
    ]
    (
        const Var<luisa::compute::BufferView<float3>> input_position,
        const Var<luisa::compute::BufferView<uint3>> input_face
    )
    {
        luisa::compute::set_block_size(256);
        
        const Uint fid = luisa::compute::dispatch_id().x;
        const UInt3 face = input_face.read(fid);
        Float3 positions[3] = {
            input_position->read(face[0]),
            input_position->read(face[1]),
            input_position->read(face[2])
        };

        Float3 center = 0.333333f * (positions[0] + positions[1] + positions[2]);
        Float2x3 aabb = AABB::make_aabb(
            positions[0],
            positions[1],
            positions[2]
        );
        reduce_aabb_1_pass_template(center, aabb);
    });


    fn_reset_tree = device.compile<1>([
        sa_is_healthy = lbvh_data->sa_is_healthy.view(),
        sa_parrent = lbvh_data->sa_parrent.view(),
        sa_escape_index = lbvh_data->sa_escape_index.view()
    ]()
    {
        const Uint vid = luisa::compute::dispatch_id().x;
        $if (vid == 0)
        {
            sa_is_healthy->write(0, 1u);
            sa_parrent->write(0, -1u);
        };
        sa_escape_index->write(vid, -1u);
    });
    
    fn_compute_mortons = device.compile<1>([
        sa_block_aabb = lbvh_data->sa_block_aabb.view(),
        sa_leaf_center = lbvh_data->sa_leaf_center.view(),
        sa_morton = lbvh_data->sa_morton.view(),
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view()
    ]()
    {
        const Uint lid = luisa::compute::dispatch_id().x;

        Float3 min_pos = AABB::get_aabb_min(sa_block_aabb->read(0));
        Float3 max_pos = AABB::get_aabb_max(sa_block_aabb->read(0));
        Float3 inv_dim = 1.0f / max_vec(max_pos - min_pos, makeFloat3Var(1e-6));
        Float3 norm_position = (sa_leaf_center->read(lid) - min_pos) * inv_dim;
        sa_leaf_center->write(lid, norm_position);
        auto mc64 = make_morton64(norm_position, lid);
        sa_morton->write(lid, mc64);
        sa_sorted_get_original->write(lid, lid);
    });

    fn_apply_sorted = device.compile<1>([
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        sa_morton = lbvh_data->sa_morton.view(),
        sa_morton_sorted = lbvh_data->sa_morton_sorted.view(),
        sa_children = lbvh_data->sa_children.view(),
        num_inner_nodes
    ]() {
        const Uint lid = dispatch_id().x;
        const Uint orig_vid = sa_sorted_get_original->read(lid);
        sa_morton_sorted->write(lid, sa_morton->read(orig_vid));
        sa_children->write(num_inner_nodes + lid, make_uint2(orig_vid, orig_vid));
    });

    fn_build_inner_nodes = device.compile<1>([
        sa_morton_sorted = lbvh_data->sa_morton_sorted.view(),
        sa_children = lbvh_data->sa_children.view(),
        sa_parrent = lbvh_data->sa_parrent.view(),
        sa_is_healthy = lbvh_data->sa_is_healthy.view(),
        num_inner_nodes
    ]() {
        const Uint nid = dispatch_id().x;

        Int num_inners = num_inner_nodes;
        Int num_leaves = num_inner_nodes + 1;

        Uint2 ranges = makeUint2(0, num_inner_nodes);
        $if (nid != 0) 
        {
            ranges = determineRange(nid, sa_morton_sorted, num_leaves);
        };

        // $if (unit_test)
        {
            // Should be [0 -> 7, 1 -> 0, 2 -> 0, 3 -> 4, 4 -> 0, 5 -> 7, 6 -> 5]
            // device_log("range {} = {} -> {}", nid, ranges[0], ranges[1]);
        };

        Int i = ranges[0];
        Int j = ranges[1];
        Int split = findSplit(ranges, sa_morton_sorted); // 

        // $if (unit_test)
        {
            // Should be [4, 0, 1, 3, 2, 6, 5,]
            // device_log("split {} = {}", nid, split);
        };

        Int child_left = select(min_scalar(i, j) == split, (num_inners + split), split);
        Int child_right = select(max_scalar(i, j) == split + 1, (num_inners + split + 1), (split + 1));

        $if (child_right >= num_inners) 
        {
            Int tmp = child_left;
            child_left = child_right;
            child_right = tmp;
        };

        // $if (unit_test)
        {
            // Should be [[4, 5] [8, 7] [9, 1] [11, 10] [2, 3] [14, 6] [13, 12] [0, 0] [0, 0] [0, 0] [0, 0] [0, 0] [0, 0] [0, 0] [0, 0]]
            // device_log("children {} = {}", nid, makeUint2(child_left, child_right));
        };

        // Should be [0, 2, 4, 4, 0, 0, 5, 1, 1, 2, 3, 3, 6, 6, 5, ]
        sa_parrent->write(child_left, nid); // 
        sa_parrent->write(child_right, nid);
        sa_children->write(nid, makeUint2(child_left, child_right));
    });

    fn_check_construction = device.compile<1>([
        sa_children = lbvh_data->sa_children.view(),
        sa_parrent = lbvh_data->sa_parrent.view(),
        sa_is_healthy = lbvh_data->sa_is_healthy.view()
    ]() {
        const Uint nid = dispatch_id().x;
        Uint2 child = sa_children->read(nid);
        Uint parrent_of_left = sa_parrent->read(child[0]);
        Uint parrent_of_right = sa_parrent->read(child[1]);
        $if (parrent_of_left != Uint(nid) | parrent_of_right != Uint(nid)) 
        {
            sa_is_healthy->write(0, 0u);
        };
    });

    fn_set_escape_index = device.compile<1>([
        sa_children = lbvh_data->sa_children.view(),
        sa_escape_index = lbvh_data->sa_escape_index.view()
    ]() {
        const Uint nid = dispatch_id().x;
        Uint2 child = sa_children->read(nid);
        Uint escape_index = child[1];
        sa_escape_index->write(child[0], escape_index);
        Uint2 child2 = sa_children->read(child[0]);
        $while (child2[0] != child2[1]) 
        {
            sa_escape_index->write(child2[1], escape_index);
            child2 = sa_children->read(child2[1]);
        };
    });

    fn_set_left_and_escape = device.compile<1>([
        sa_children = lbvh_data->sa_children.view(),
        sa_left_and_escape = lbvh_data->sa_left_and_escape.view(),
        sa_escape_index = lbvh_data->sa_escape_index.view()
    ]() {
        const Uint nid = dispatch_id().x;
        Uint2 child = sa_children->read(nid);
        Uint leftIdx = select(child[0] == child[1], make_leaf(child[0]), Uint(child[0]));
        Uint escapeIdx = sa_escape_index->read(nid);
        sa_left_and_escape->write(nid, make_uint2(leftIdx, escapeIdx));
    });


    // Refit
    fn_update_vert_tree_leave_aabb = device.compile<1>([
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        num_inner_nodes
    ](
        const Var<luisa::compute::BufferView<float3>> sa_x_start,
        const Var<luisa::compute::BufferView<float3>> sa_x_end,
        const Float thickness
    ) {
        const Uint lid = dispatch_id().x;
        Uint vid = sa_sorted_get_original->read(lid);
        Float2x3 aabb = AABB::make_aabb(sa_x_start->read(vid), sa_x_end->read(vid));
        aabb = AABB::add_thickness(aabb, thickness);
        sa_node_aabb->write(num_inner_nodes + lid, aabb);
    });

    fn_update_edge_tree_leave_aabb = device.compile<1>([
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        num_inner_nodes
    ](
        const Var<luisa::compute::BufferView<float3>> sa_x_start,
        const Var<luisa::compute::BufferView<float3>> sa_x_end,
        const Var<luisa::compute::BufferView<uint2>> input_edge,
        const Float thickness
    ) {
        const Uint lid = dispatch_id().x;
        Uint fid = sa_sorted_get_original->read(lid);
        UInt2 edge = input_edge->read(fid);
        Float3 start_positions[2] = {
            sa_x_start->read(edge[0]),
            sa_x_start->read(edge[1])
        };
        Float3 end_positions[2] = {
            sa_x_end->read(edge[0]),
            sa_x_end->read(edge[1])
        };
        Float2x3 aabb = AABB::make_aabb(start_positions[0], start_positions[1], end_positions[0], end_positions[1]);
        aabb = AABB::add_thickness(aabb, thickness);
        sa_node_aabb->write(num_inner_nodes + lid, aabb);
    });
    
    fn_update_face_tree_leave_aabb = device.compile<1>([
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        num_inner_nodes
    ](
        const Var<luisa::compute::BufferView<float3>> sa_x_start,
        const Var<luisa::compute::BufferView<float3>> sa_x_end,
        const Var<luisa::compute::BufferView<uint3>> input_face,
        const Float thickness
    ) {
        const Uint lid = dispatch_id().x;
        Uint fid = sa_sorted_get_original->read(lid);
        UInt3 face = input_face->read(fid);
        Float3 start_positions[3] = {
            sa_x_start->read(face[0]),
            sa_x_start->read(face[1]),
            sa_x_start->read(face[2])
        };
        Float3 end_positions[3] = {
            sa_x_end->read(face[0]),
            sa_x_end->read(face[1]),
            sa_x_end->read(face[2])
        };
        Float2x3 start_aabb = AABB::make_aabb(start_positions[0], start_positions[1], start_positions[2]);
        Float2x3 end_aabb = AABB::make_aabb(end_positions[0], end_positions[1], end_positions[2]);
        Float2x3 aabb = AABB::add_aabb(start_aabb, end_aabb);
        aabb = AABB::add_thickness(aabb, thickness);
        sa_node_aabb->write(num_inner_nodes + lid, aabb);
    });



    fn_clear_apply_flag = device.compile<1>([
        sa_apply_flag = lbvh_data->sa_apply_flag.view()
    ]() {
        const Uint nid = dispatch_id().x;
        sa_apply_flag->write(nid, 0u);
    });

    fn_refit_kernel = device.compile<1>([
        sa_apply_flag = lbvh_data->sa_apply_flag.view(),
        sa_parrent = lbvh_data->sa_parrent.view(),
        sa_children = lbvh_data->sa_children.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        sa_is_healthy = lbvh_data->sa_is_healthy.view(),
        num_inner_nodes
    ]() {
        const Uint lid = dispatch_id().x;
        Uint current = lid + num_inner_nodes;
        Uint parrent = sa_parrent->read(current);
        Uint loop = 0;
        $while (parrent != -1) {
            loop += 1;
            $if (loop > 10000) {
                sa_is_healthy->write(0, 0u);
                $break;
            };
            Uint orig_flag = sa_apply_flag->atomic(parrent).fetch_add(1u);
            $if (orig_flag == 0) {
                $break;
            } 
            $elif (orig_flag == 1) {
                sa_apply_flag->atomic(parrent).fetch_add(1u);
                Uint2 child_of_parrent = sa_children->read(parrent);
                Float2x3 aabb_left = sa_node_aabb->read(child_of_parrent[0]);
                Float2x3 aabb_right = sa_node_aabb->read(child_of_parrent[1]);
                sa_node_aabb->write(parrent, AABB::add_aabb(aabb_left, aabb_right));
                current = parrent;
                parrent = sa_parrent->read(current);
            } 
            $else {
                sa_is_healthy->write(0, 0u);
                $break;
            };
        };
    });

    // Query
    auto query_template = [
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        sa_left_and_escape = lbvh_data->sa_left_and_escape.view(),
        num_leaves
    ](
        const Float2x3& input_aabb,
        Var<BufferView<uint>>& broadphase_count,
        Var<BufferView<uint>>& broad_phase_list
    )
    {
        const Uint vid = dispatch_id().x;
        Uint current = num_leaves;
        Uint loop = 0;
        $while (true) {
            loop += 1;
            $if (loop > 10000) { $break; };
            Float2x3 aabb = sa_node_aabb->read(current);
            Uint2 leftAndEscape = sa_left_and_escape->read(current);
            Uint left = leftAndEscape[0];
            Uint escape = leftAndEscape[1];
            $if (AABB::is_overlap_aabb(aabb, input_aabb)) {
                $if (is_leaf(left) == 1) {
                    Uint adj_vid = extract_leaf(left);
                    Uint idx = broadphase_count->atomic(0).fetch_add(1u);
                    broad_phase_list->write(idx * 2 + 0, vid);
                    broad_phase_list->write(idx * 2 + 1, adj_vid);
                } $else {
                    current = left;
                    $continue;
                };
            };
            current = (escape);
            $if (current == -1) { $break; };
        };
    };

    fn_reset_collision_count = device.compile<1>([](
        Var<BufferView<uint>> broadphase_count
    ){
        const Uint vid = dispatch_id().x;
        broadphase_count.write(vid, 0u);
    });

    fn_query_vert = device.compile<1>([
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        sa_left_and_escape = lbvh_data->sa_left_and_escape.view(),
        num_leaves, &query_template
    ](
        Var<BufferView<float3>> sa_x_begin,
        Var<BufferView<float3>> sa_x_end,
        Var<BufferView<uint>> broad_phase_list,
        Var<BufferView<uint>> broadphase_count,
        const Float thickness
    ) {
        const Uint vid = dispatch_id().x;
        // Float3 pos = sa_x_begin->read(vid);
        // Float2x3 vert_aabb = AABB::make_aabb(pos - make_float3(thickness), pos + make_float3(thickness));
        Float2x3 vert_aabb = AABB::make_aabb(sa_x_end.read(vid), sa_x_end.read(vid));
        query_template(vert_aabb, broadphase_count, broad_phase_list);
    });

    // auto buffer = device.create_buffer<bool>(1);
    // auto read_bool = device.compile<1>([
    //     buffer = buffer.view()
    // ](){
    //     buffer->write(0, false);
    // });
}

template <typename T>
static inline bool is_the_same(luisa::compute::Stream& stream, luisa::compute::Buffer<T>& buffer, std::vector<T>& vector)
{
    std::vector<T> buffer_result(buffer.size());
    stream << buffer.copy_to(buffer_result) << luisa::compute::synchronize();
    for (uint i = 0; i < buffer.size(); i++)
    {
        if (buffer_result[i] != vector[i])
        {
            luisa::log_info("Not equal at {} : get {} desire {}", i, buffer_result[i], vector[i]);
            return false;
        }
    }
    return true;
} 

void LBVH::unit_test(luisa::compute::Device& device, luisa::compute::Stream& stream)
{
    using namespace luisa::compute;

    LbvhData<luisa::compute::Buffer> tmp_lbvh_data;
    tmp_lbvh_data.allocate(device, 8, LBVHTreeTypeVert, LBVHUpdateTypeCloth);
    set_lbvh_data(&tmp_lbvh_data);
    compile(device);

    const uint num_leaves = lbvh_data->num_leaves;
    const uint num_inner_nodes = lbvh_data->num_inner_nodes;
    const uint num_nodes = lbvh_data->num_nodes;
    
    const std::vector<morton64> answer_morton32 = {
        0, 2064888,
        16519104, 117698623,
        132152839, 939524096,
        941588984, 956043200,
    };
    const std::vector<morton64> answer_morton64 = {
        0, 8868626429902849,
        70949011439222786, 505511736569233411,
        567592121578553348, 4035225266123964421,
        4044093892553867270, 4106174277563187207
    };

    auto init_test = device.compile<1>([
        sa_morton_sorted = lbvh_data->sa_morton_sorted.view(),
        sa_children = lbvh_data->sa_children.view()
    ]()
    {
        const Uint lid = dispatch_x();
        Float pos = Float(lid) / 10.0f;
        auto mc32 = make_morton32(makeFloat3(pos));
        auto mc64 = make_morton64(makeFloat3(pos), lid);
        // auto mc64 = (static_cast<Morton64>(mc32) << 32) | (static_cast<Morton64>(lid) & static_cast<Morton64>(0xFFFFFFFF));
        sa_morton_sorted->write(lid, mc64);
        sa_children->write(7 + lid, makeUint2(lid));
        {
            device_log("lid {} morton32 = {}, morton64 = {}", lid, mc32, mc64);
        }
    });

    stream 
        << init_test().dispatch(8)
        << synchronize();

    // construct_tree(stream);
    std::vector<uint> host_parrent(15);
    std::vector<uint2> host_children(15);

    stream 
        << fn_build_inner_nodes().dispatch(num_inner_nodes)
        << lbvh_data->sa_parrent.copy_to(host_parrent.data())
        << lbvh_data->sa_children.copy_to(host_children.data())
        << synchronize();

    for (uint i = 0; i < host_parrent.size(); i++)
    {
        auto parrent = host_parrent[i];
        luisa::log_info("parrent of {} = {}", i, parrent);
    }
    for (uint i = 0; i < host_children.size(); i++)
    {
        auto parrent = host_children[i];
        luisa::log_info("children of {} = {}", i, parrent);
    }
}

// Construct

void LBVH::reduce_vert_tree_aabb(Stream& stream, const Buffer<float3>& input_position)
{
    if (input_position.size() > 256 * 256) { luisa::log_error("Buffer size out of reduce range"); exit(0); }
    stream 
        << fn_reduce_vert_tree_global_aabb(input_position).dispatch(input_position.size())
        << fn_reduce_aabb_2_pass().dispatch(get_dispatch_block(input_position.size(), 256));
}
void LBVH::reduce_edge_tree_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint2>& input_edges)
{
    if (input_edges.size() > 256 * 256) { luisa::log_error("Buffer size out of reduce range"); exit(0); }
    stream 
        << fn_reduce_edge_tree_global_aabb(input_position, input_edges).dispatch(input_edges.size())
        << fn_reduce_aabb_2_pass().dispatch(get_dispatch_block(input_edges.size(), 256));
}
void LBVH::reduce_face_tree_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint3>& input_faces)
{
    if (input_faces.size() > 256 * 256) { luisa::log_error("Buffer size out of reduce range"); exit(0); }
    stream 
        << fn_reduce_face_tree_global_aabb(input_position, input_faces).dispatch(input_faces.size())
        << fn_reduce_aabb_2_pass().dispatch(get_dispatch_block(input_faces.size(), 256));
}
void LBVH::construct_tree(Stream& stream)
{
    const uint num_leaves = lbvh_data->num_leaves;
    const uint num_inner_nodes = lbvh_data->num_inner_nodes;
    const uint num_nodes = lbvh_data->num_nodes;


    static std::vector<morton64> host_morton64;
    static std::vector<uint> host_sorted_get_original;
    if (host_morton64.empty())
    {
        host_morton64.resize(num_leaves);
        host_sorted_get_original.resize(num_leaves);
    }

    stream 
        << fn_reset_tree().dispatch(num_nodes) 
        << fn_compute_mortons().dispatch(num_leaves) 
        << lbvh_data->sa_morton.copy_to(host_morton64.data())
        << lbvh_data->sa_sorted_get_original.copy_to(host_sorted_get_original.data())
        << luisa::compute::synchronize();

    CpuParallel::parallel_sort(host_sorted_get_original.data(), host_sorted_get_original.data() + num_leaves, [&](const uint idx1, const uint idx2) -> bool 
    {
        return host_morton64[idx1] < host_morton64[idx2];
    });

    stream 
        << lbvh_data->sa_morton.copy_from(host_morton64.data())
        << lbvh_data->sa_sorted_get_original.copy_from(host_sorted_get_original.data())
        << fn_apply_sorted().dispatch(num_leaves)
        << fn_build_inner_nodes().dispatch(num_inner_nodes)

        << fn_check_construction().dispatch(num_inner_nodes)
        << fn_set_escape_index().dispatch(num_inner_nodes)
        << fn_set_left_and_escape().dispatch(num_nodes)
        ;
}

// Refit 

void LBVH::update_vert_tree_leave_aabb(Stream& stream, 
    const Buffer<float3>& start_position, 
    const Buffer<float3>& end_position)
{
    stream << fn_update_vert_tree_leave_aabb(start_position, end_position, 0.01f).dispatch(start_position.size());
}
void LBVH::update_edge_tree_leave_aabb(Stream& stream, 
    const Buffer<float3>& start_position, 
    const Buffer<float3>& end_position, 
    const Buffer<uint2>& input_edges)
{
    stream << fn_update_edge_tree_leave_aabb(start_position, end_position, input_edges, 0.01f).dispatch(start_position.size());
}
void LBVH::update_face_tree_leave_aabb(Stream& stream, 
    const Buffer<float3>& start_position, 
    const Buffer<float3>& end_position, 
    const Buffer<uint3>& input_faces)
{
    stream << fn_update_face_tree_leave_aabb(start_position, end_position, input_faces, 0.01f).dispatch(start_position.size());
}
void LBVH::refit(Stream& stream)
{
    stream 
        << fn_clear_apply_flag().dispatch(lbvh_data->sa_apply_flag.size())
        << fn_refit_kernel().dispatch(lbvh_data->num_leaves);
}


void LBVH::broad_phase_query_vert(
    Stream& stream, 
    const Buffer<float3>& sa_x_begin, 
    const Buffer<float3>& sa_x_end, 
    Buffer<uint>& broad_phase_list, 
    Buffer<uint>& broadphase_count, 
    const float thickness)
{
    stream
        << fn_reset_collision_count(broadphase_count).dispatch(8)
        << fn_query_vert(sa_x_begin, sa_x_end, broad_phase_list, broadphase_count, thickness).dispatch(sa_x_begin.size());
}


};