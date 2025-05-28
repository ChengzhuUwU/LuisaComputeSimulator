#include "CollisionDetector/lbvh.h"
#include "CollisionDetector/aabb.h"
#include "Utils/reduce_helper.h"

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

    const uint num_leaves = lbvh_data->num_leaves;
    const uint num_inner_nodes = lbvh_data->num_inner_nodes;
    const uint num_nodes = lbvh_data->num_nodes;

    // Reduce AABB and get leaf center
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

        auto reduced_aabb = ParallelIntrinsic::block_reduce(vid, aabb, AABB::reduce_aabb);

        $if (vid % 256 == 0)
        {
            const Uint blockIdx = vid / ParallelIntrinsic::reduce_block_dim;
            sa_block_aabb->write(blockIdx, aabb);
        };
    };

    auto reduce_aabb_2_pass_template = [
        sa_block_aabb = lbvh_data->sa_block_aabb.view()
    ]()
    {
        luisa::compute::set_block_size(256);
        const Uint vid = luisa::compute::dispatch_id().x;

        auto aabb = sa_block_aabb->read(vid);
        auto reduced_aabb = ParallelIntrinsic::block_reduce(vid, aabb, AABB::reduce_aabb);

        $if (vid % 256 == 0)
        {
            const Uint blockIdx = vid / ParallelIntrinsic::reduce_block_dim;
            sa_block_aabb->write(blockIdx, aabb);
        };
    };

    auto reduce_vert_tree_global_aabb = device.compile<1>([
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

    auto reduce_face_tree_global_aabb = device.compile<1>([
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
        Float3 face_pos[3] = {
            input_position->read(face[0]),
            input_position->read(face[1]),
            input_position->read(face[2])
        };

        Float3 face_center = 0.333333f * (face_pos[0] + face_pos[1] + face_pos[2]);
        Float2x3 aabb = AABB::make_aabb(
            face_pos[0],
            face_pos[1],
            face_pos[2]
        );
        reduce_aabb_1_pass_template(face_center, aabb);
    });

    auto reset_tree = device.compile<1>([
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

    auto compute_mortons = device.compile<1>([
        sa_is_healthy = lbvh_data->sa_is_healthy.view(),
        sa_parrent = lbvh_data->sa_parrent.view(),
        sa_escape_index = lbvh_data->sa_escape_index.view(),
        sa_block_aabb = lbvh_data->sa_block_aabb.view(),
        sa_leaf_center = lbvh_data->sa_leaf_center.view(),
        sa_morton = lbvh_data->sa_morton.view(),
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        &reduce_aabb_1_pass_template
    ]
    ()
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

    // Apply Sorted
    auto apply_sorted = device.compile<1>([
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
    
    auto build_inner_nodes = device.compile<1>([
        sa_morton_sorted = lbvh_data->sa_morton_sorted.view(),
        sa_children = lbvh_data->sa_children.view(),
        sa_parrent = lbvh_data->sa_parrent.view(),
        sa_is_healthy = lbvh_data->sa_is_healthy.view(),
        num_inner_nodes = Var<int>(num_inner_nodes),
        num_leaves = Var<int>(num_leaves)
    ]() {
        const Uint nid = dispatch_id().x;
        Uint2 ranges = makeUint2(0, num_inner_nodes);
        $if (nid != 0) {
            ranges = determineRange(nid, sa_morton_sorted, num_leaves);
        };
        Int i = ranges[0];
        Int j = ranges[1];
        Int split = findSplit(ranges, sa_morton_sorted);

        Int child_left = select(min_scalar(i, j) == split, (num_inner_nodes + split), split);
        Int child_right = select(max_scalar(i, j) == split + 1, (num_inner_nodes + split + 1), (split + 1));

        $if (child_right >= Int(num_inner_nodes)) 
        {
            Int tmp = child_left;
            child_left = child_right;
            child_right = tmp;
        };
        sa_parrent->write(child_left, nid);
        sa_parrent->write(child_right, nid);
        sa_children->write(nid, makeUint2(child_left, child_right));
    });

    // 3. Check Construction
    auto check_construction = device.compile<1>([
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


    auto set_escape_index = device.compile<1>([
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

    auto set_left_and_escape = device.compile<1>([
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

    auto update_vert_tree_leave_aabb = device.compile<1>([
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        num_inner_nodes
    ](
        const Var<luisa::compute::BufferView<float3>> input_position,
        const Float thickness
    ) {
        const Uint lid = dispatch_id().x;
        Uint vid = sa_sorted_get_original->read(lid);
        Float2x3 aabb = AABB::make_aabb(input_position->read(vid));
        aabb = AABB::add_thickness(aabb, thickness);
        sa_node_aabb->write(num_inner_nodes + lid, aabb);
    });

    auto update_face_tree_leave_aabb = device.compile<1>([
        sa_sorted_get_original = lbvh_data->sa_sorted_get_original.view(),
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        num_inner_nodes
    ](
        const Var<luisa::compute::BufferView<float3>> input_position,
        const Var<luisa::compute::BufferView<uint3>> input_face,
        const Float thickness
    ) {
        const Uint lid = dispatch_id().x;
        Uint fid = sa_sorted_get_original->read(lid);
        UInt3 face = input_face->read(fid);
        Float3 face_pos[3] = {
            input_position->read(face[0]),
            input_position->read(face[1]),
            input_position->read(face[2])
        };
        Float2x3 aabb = AABB::make_aabb(face_pos[0], face_pos[1], face_pos[2]);
        aabb = AABB::add_thickness(aabb, thickness);
        sa_node_aabb->write(num_inner_nodes + lid, aabb);
    });

    auto clear_apply_flag = device.compile<1>([
        sa_apply_flag = lbvh_data->sa_apply_flag.view()
    ]() {
        const Uint nid = dispatch_id().x;
        sa_apply_flag->write(nid, 0u);
    });

    auto buffer = device.create_buffer<bool>(1);
    auto read_bool = device.compile<1>([
        buffer = buffer.view()
    ](){
        buffer->write(0, false);
    });

    auto refit_kernel = device.compile<1>([
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

    auto query_kernel = device.compile<1>([
        sa_node_aabb = lbvh_data->sa_node_aabb.view(),
        sa_left_and_escape = lbvh_data->sa_left_and_escape.view(),
        num_leaves = Uint(num_leaves)
    ](
        Var<BufferView<float3>> input_position,
        Var<BufferView<uint>> broad_phase_list,
        Var<BufferView<uint>> broadphase_count,
        const Float thickness
    ) {
        const Uint vid = dispatch_id().x;
        Float3 pos = input_position->read(vid);
        Float2x3 vert_aabb = AABB::make_aabb(pos - make_float3(thickness), pos + make_float3(thickness));
        Uint current = num_leaves;
        Uint loop = 0;
        $while (true) {
            loop += 1;
            $if (loop > 10000) { $break; };
            Float2x3 aabb = sa_node_aabb->read(current);
            Uint2 leftAndEscape = sa_left_and_escape->read(current);
            Uint left = leftAndEscape[0];
            Uint escape = leftAndEscape[1];
            $if (AABB::is_overlap_pos(aabb, pos)) {
                $if (is_leaf(left) == 1) {
                    Uint adj_vid = extract_leaf(left);
                    Uint idx = broadphase_count->atomic(0).fetch_add(1u);
                    broad_phase_list->write(idx * 2 + 0, vid);
                    broad_phase_list->write(idx * 2 + 1, adj_vid);
                } $else {
                    current = Uint(left);
                    $continue;
                };
            };
            current = (escape);
            $if (current == -1) { $break; };
        };
    });
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