#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/aabb.h"

namespace lcsv
{

template<typename T>
void resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& buffer, const uint size)
{
    buffer = device.create_buffer<T>(size);
}

void LBVH::init(luisa::compute::Device& device, luisa::compute::Stream& stream, 
        const uint input_num, const LBVHTreeType tree_type, const LBVHUpdateType update_type)
{
    const uint num_leaves = input_num;
    const uint num_inner_nodes = num_leaves - 1;
    const uint num_nodes = num_leaves + num_inner_nodes;

    lbvh_data->num_leaves = num_leaves;
    lbvh_data->num_inner_nodes = num_inner_nodes;
    lbvh_data->num_nodes = num_nodes;
    lbvh_data->tree_type = tree_type;
    lbvh_data->update_type = update_type;

    resize_buffer(device, lbvh_data->sa_leaf_center, num_leaves);
    resize_buffer(device, lbvh_data->sa_block_aabb, get_dispatch_block(num_leaves, 256));
    resize_buffer(device, lbvh_data->sa_morton, num_leaves);
    resize_buffer(device, lbvh_data->sa_morton_sorted, num_leaves);
    resize_buffer(device, lbvh_data->sa_sorted_get_original, num_leaves);
    resize_buffer(device, lbvh_data->sa_parrent, num_nodes);
    resize_buffer(device, lbvh_data->sa_children, num_nodes);
    resize_buffer(device, lbvh_data->sa_escape_index, num_nodes);
    resize_buffer(device, lbvh_data->sa_left_and_escape, num_nodes);
    resize_buffer(device, lbvh_data->sa_node_aabb, num_nodes);
    resize_buffer(device, lbvh_data->sa_apply_flag, num_nodes);
    resize_buffer(device, lbvh_data->sa_is_healthy, 1);
}


// Existing functions
template<typename UintType> UintType make_leaf(const UintType mask)     { return (mask) | static_cast<UintType>(1 << 31); }
template<typename UintType> UintType is_leaf(const UintType mask)       { return (mask & static_cast<UintType>(1 << 31)) != 0; }
template<typename UintType> UintType extract_leaf(const UintType mask)  { return (mask) & (~(static_cast<UintType>(1 << 31))); }


int clz_ulong(morton64 x) 
{
    int count = 0;
    while (x != 0) 
    {
        count += 1;
        x >>= 1;
    }
    return int(64 - count);
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
inline Var<morton64> get_morton(const luisa::compute::BufferView<morton64>& buffer, const Var<int> index){ return buffer->read(index); }
inline morton64 get_morton(const std::vector<morton64>& buffer, const int index){ return buffer[index]; }


template<typename MortonType, typename UintType, typename IntType, typename BufferType>
IntType cp_i_j(const MortonType& mi, IntType j, const BufferType& sa_morton_sorted, const UintType num_leaves) 
{
    auto isValid = (j >= 0 & static_cast<UintType>(j) < num_leaves);
    return select(isValid, find_common_prefix(mi, get_morton(sa_morton_sorted, j)), static_cast<IntType>(-1));
}

auto determineRange(const Var<uint> index, const luisa::compute::BufferView<morton64>& sa_morton_sorted, const Var<uint> num_leaves) 
{
    using IndexType = Var<int>;
    IndexType i = index;
    auto mi = get_morton(sa_morton_sorted, i);
    auto cp_left  = find_common_prefix(mi, get_morton(sa_morton_sorted, i - 1));
    auto cp_right = find_common_prefix(mi, get_morton(sa_morton_sorted, i + 1));

    IndexType d = select(cp_left < cp_right, Var<int>(1), Var<int>(-1));
    IndexType cp_min = min_scalar(cp_left, cp_right);

    IndexType lmax = 2;
    $while (cp_i_j(mi, i + lmax * d, sa_morton_sorted, num_leaves) > cp_min) 
    {
        lmax <<= 1;
    };

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

#define reduce_with_cache(thread_value, tid, wid, dim, type, set_op, reduce_op, get_op)  \
        luisa::compute::Shared<type> cache_aabb(dim);                                   \
        set_op(cache_aabb[tid], thread_value)                                           \
            luisa::compute::sync_block();                   \
            luisa::compute::Uint s = dim >> 1;              \
			$while(true) {	                \
                $if (s == 0) { $break; };   \
				$if(tid < s) {                                              \
                    reduce_op(cache_aabb[tid], cache_aabb[tid + s]);        \
                };                                                          \
				luisa::compute::sync_block();                               \
                s >>= 1;                                                    \
			};                                                              \
            $if (wid == 0) {                                                \
                get_op(thread_value, cache_aabb[0]);                        \
            };

#define set_op_aabb(a, b)    a.cols[0] = b.cols[0]; a.cols[1] = b.cols[1];
#define get_op_aabb(a, b)    a.cols[0] = b.cols[0]; a.cols[1] = b.cols[1];
#define reduce_op_aabb(a, b) a.cols[0] = min_vec(a.cols[0], b.cols[0]); a.cols[1] = max_vec(a.cols[1], b.cols[1]);
#define reduce_aabb(aabb, tid, wid, dim) reduce_with_cache(aabb, tid, wid, dim, float2x3, set_op_aabb, reduce_op_aabb, get_op_aabb)   

void LBVH::compile(luisa::compute::Device& device)
{
    using namespace luisa::compute;

    auto reduce_vert_tree_global_aabb = device.compile<1>([
        sa_block_aabb = lbvh_data->sa_block_aabb.view(),
        sa_leaf_center = lbvh_data->sa_leaf_center.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        sa_is_healthy = lbvh_data->sa_is_healthy.view()
    ]
    (
        const Var<luisa::compute::BufferView<float3>> input_position
    )
    {
        luisa::compute::set_block_size(256);
        
        const Uint vid = luisa::compute::dispatch_id().x;
        Float2x3 default_aabb = makeFloat2x3(
            makeFloat3(Var<float>( 1000.0f)), 
            makeFloat3(Var<float>(-1000.0f))
        );
        
        Float3 vert_pos = input_position->read(vid);
        sa_leaf_center->write(vid, vert_pos);
        
        auto aabb = AABB::make_aabb(vert_pos);
        

        const Uint threadIdx = vid % 256;
        const Uint warpIdx = vid / 256;
        
        reduce_aabb(aabb, threadIdx, warpIdx, 256);

        sa_block_aabb->write(warpIdx, aabb);;
    });

    auto sa_block_aabb = lbvh_data->sa_block_aabb.view();
    auto sa_leaf_center = lbvh_data->sa_leaf_center.view();
    auto sa_node_aabb = lbvh_data->sa_node_aabb.view();
    auto sa_is_healthy = lbvh_data->sa_is_healthy.view();

}
void LBVH::construct_tree(Stream& stream)
{

}
void LBVH::refit(Stream& stream)
{

}
void LBVH::update_vert_tree_leave_aabb(Stream& stream, const Buffer<float3>& input_position)
{

}
void LBVH::update_face_tree_leave_aabb(Stream& stream, const Buffer<float3>& input_position, const Buffer<uint3>& input_faces)
{

}
void LBVH::broad_phase_query(Stream& stream, const Buffer<float3>& input_position, Buffer<uint>& broad_phase_list, Buffer<uint>& broadphase_count, const float thickness)
{

}


};