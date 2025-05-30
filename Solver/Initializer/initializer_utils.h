#pragma once

#include "luisa/core/logging.h"
#include "luisa/runtime/device.h"

namespace lcsv 
{

namespace Initializater
{


inline void fn_graph_coloring_per_vertex(const std::vector< std::vector<uint> >& vert_adj_, std::vector<std::vector<uint>>& clusterd_vertices, std::vector<uint>& prefix_vert)
{
    const uint num_verts_total = vert_adj_.size();
    std::vector<bool> marked_verts(num_verts_total, false);
    uint total_marked_count = 0;

    while (true) 
    {
        std::vector<uint> current_cluster;
        std::vector<bool> current_marked(marked_verts);

        for (uint vid = 0; vid < num_verts_total; vid++) 
        {
            if (current_marked[vid]) 
            {
                continue;
            }
            else 
            {
                // Add To Sets
                marked_verts[vid] = true;
                current_cluster.push_back(vid);

                // Mark
                current_marked[vid] = true;
                const auto& list = vert_adj_[vid];
                for (const uint& adj_vid : list) 
                {
                    current_marked[adj_vid] = true;
                }
            }
        }
        clusterd_vertices.push_back(current_cluster);
        total_marked_count += current_cluster.size();

        if (total_marked_count == num_verts_total) break;
    }
    

    prefix_vert.resize(clusterd_vertices.size() + 1); uint curr_prefix = 0;
    for (uint cluster = 0; cluster < clusterd_vertices.size(); cluster++)
    {
        prefix_vert[cluster] = curr_prefix;
        curr_prefix += clusterd_vertices[cluster].size();
    } prefix_vert[clusterd_vertices.size()] = curr_prefix;
};

inline void fn_graph_coloring_per_constraint(const uint num_elements, const std::string& constraint_name, 
    std::vector< std::vector<uint> > & clusterd_constraint, 
    const std::vector< std::vector<uint> > & vert_adj_elements, const auto& element_indices, const uint nv)
{ 
    std::vector< bool > marked_constrains(num_elements, false);
    uint total_marked_count = 0;


    const uint color_threashold = 2000;
    std::vector<uint> rest_cluster;

    //
    // while there exist unmarked constraints
    //     create new set S
    //     clear all particle marks
    //     for all unmarked constraints C
    //      if no adjacent particle is marked
    //          add C to S
    //          mark C
    //          mark all adjacent particles
    //
    
    const bool merge_small_cluster = false;

    while (true) 
    {
        std::vector<uint> current_cluster;
        std::vector<bool> current_marked(marked_constrains);
        for (uint id = 0; id < num_elements; id++) 
        {
            if (current_marked[id]) 
            {
                continue;
            }
            else 
            {
                // Add To Sets
                marked_constrains[id] = true;
                current_cluster.push_back(id);

                // Mark
                current_marked[id] = true;
                auto element = element_indices[id];
                for (uint j = 0; j < nv; j++) 
                {
                    for (const uint& adj_eid : vert_adj_elements[element[j]]) 
                    { 
                        current_marked[adj_eid] = true; 
                    }
                }
            }
        }
        
        const uint cluster_size = static_cast<uint>(current_cluster.size());
        total_marked_count += cluster_size;

        
        if (merge_small_cluster && cluster_size < color_threashold) 
        {
            rest_cluster.insert(rest_cluster.end(), current_cluster.begin(), current_cluster.end());
        }
        else 
        {
            clusterd_constraint.push_back(current_cluster);
        }
        
        if (total_marked_count == num_elements) break;
    }

    if (merge_small_cluster && !rest_cluster.empty()) 
    {
        clusterd_constraint.push_back(rest_cluster);
    }

    luisa::log_info("Cluster Count of {} = {}", constraint_name, clusterd_constraint.size());
};

inline void fn_get_prefix(auto& prefix_buffer, const std::vector< std::vector<uint> >& clusterd_constraint)
{
    const uint num_cluster = clusterd_constraint.size();
    prefix_buffer.resize(num_cluster + 1);
    uint prefix = 0;
    for (uint cluster = 0; cluster < num_cluster; cluster++)
    {
        prefix_buffer[cluster] = prefix;
        prefix += clusterd_constraint[cluster].size();
    }
    prefix_buffer[num_cluster] = prefix;
};


template<typename T>
static inline auto upload_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& dest, const std::vector<T>& src)
{
    dest = device.create_buffer<T>(src.size());
    return dest.copy_from(src.data());
};

template<typename T>
inline void upload_from(std::vector<T>& dest, const std::vector<T>& input_data) 
{ 
    dest.resize(input_data.size());
    std::memcpy(dest.data(), input_data.data(), dest.size() * sizeof(T));  
}
inline uint upload_2d_csr_from(std::vector<uint>& dest, const std::vector<std::vector<uint>>& input_map) 
{
    uint num_outer = input_map.size();
    uint current_prefix = num_outer + 1;
    
    std::vector<uint> prefix_list(num_outer + 1);

    uint max_count = 0;
    for (uint i = 0; i < num_outer; i++) 
    {
        const auto& inner_list = input_map[i];
        uint num_inner = inner_list.size(); max_count = std::max(max_count, num_inner);
        prefix_list[i] = current_prefix;
        current_prefix += num_inner;
    }
    uint num_data = current_prefix;
    prefix_list[num_outer] = current_prefix;
    
    dest.resize(num_data);
    std::memcpy(dest.data(), prefix_list.data(), (num_outer + 1) * sizeof(uint));

    for (uint i = 0; i < num_outer; i++) 
    {
        const auto& inner_list = input_map[i];
        uint current_prefix = prefix_list[i];
        uint current_end = prefix_list[i + 1];
        for (uint j = current_prefix; j < current_end; j++) 
        {
            dest[j] = inner_list[j - current_prefix];
        }
    }
    return max_count;
}
template<typename T>
static inline auto resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& dest, const std::vector<T>& src)
{
    dest = device.create_buffer<T>(src.size());
};
template<typename T>
static inline auto resize_buffer(luisa::compute::Device& device, luisa::compute::Buffer<T>& dest, const uint size)
{
    dest = device.create_buffer<T>(size);
};
template<typename T>
static inline auto resize_buffer(std::vector<T>& dest, const uint size)
{
    dest.resize(size);
};


} // namespace Initializater


} // namespace lcsv