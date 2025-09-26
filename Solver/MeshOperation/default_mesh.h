#pragma once

#include <vector>
#include "MeshOperation/mesh_reader.h"

namespace SimMesh
{

namespace BoundingBox
{

    inline constexpr uint get_num_vertices()
    {
        return 8;
    }
    inline constexpr uint get_num_faces()
    {
        return 12;
    }
    inline constexpr std::vector<Int3> get_box_faces()
    {
        return std::vector<Int3>{Int3({0, 1, 3}),
                                 Int3({0, 2, 3}),
                                 Int3({4, 5, 7}),
                                 Int3({4, 6, 7}),
                                 Int3({2, 3, 6}),
                                 Int3({3, 6, 7}),
                                 Int3({0, 1, 4}),
                                 Int3({1, 4, 5}),
                                 Int3({1, 3, 5}),
                                 Int3({3, 5, 7}),
                                 Int3({0, 2, 4}),
                                 Int3({2, 4, 6})};
    }
    template <typename T>
    inline void update_vertices(std::vector<T>& vertices, const Float3& aabb_min, const Float3& aabb_max)
    {
        vertices[0] = T{aabb_min[0], aabb_min[1], aabb_min[2]};
        vertices[1] = T{aabb_max[0], aabb_min[1], aabb_min[2]};
        vertices[2] = T{aabb_min[0], aabb_max[1], aabb_min[2]};
        vertices[3] = T{aabb_max[0], aabb_max[1], aabb_min[2]};
        vertices[4] = T{aabb_min[0], aabb_min[1], aabb_max[2]};
        vertices[5] = T{aabb_max[0], aabb_min[1], aabb_max[2]};
        vertices[6] = T{aabb_min[0], aabb_max[1], aabb_max[2]};
        vertices[7] = T{aabb_max[0], aabb_max[1], aabb_max[2]};
    }

}  // namespace BoundingBox


}  // namespace SimMesh