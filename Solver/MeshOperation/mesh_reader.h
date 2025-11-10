#pragma once

#include <string>
#include <iostream>
#include <string>
#include <fstream>
#include <array>
#include <algorithm>
#include <math.h>
#include <luisa/core/stl/vector.h>

#include "MeshOperation/tiny_obj_loader.h"
#include "Utils/cpu_parallel.h"


namespace SimMesh
{

// typedef OpenMesh::TriMesh_ArrayKernelT<>  MeshInfo;
using MeshShape  = std::vector<tinyobj::shape_t>;
using MeshAttrib = tinyobj::attrib_t;
using MeshMat    = std::vector<tinyobj::material_t>;

using uint   = unsigned int;
using Float2 = std::array<float, 2>;
using Float3 = std::array<float, 3>;
using Int2   = std::array<uint, 2>;
using Int3   = std::array<uint, 3>;
using Int4   = std::array<uint, 4>;
using uchar  = unsigned char;

struct TriangleMeshData
{
    std::vector<Float3> model_positions;
    std::vector<Float2> uv_positions;
    std::vector<Int3>   faces;
    std::vector<Int3>   normal_faces;
    std::vector<Int3>   texcoord_faces;
    std::vector<uint>   uv_to_vert_map;
    std::vector<Int2>   edges;
    std::vector<Int4>   dihedral_edges;
    std::vector<Int2>   boundary_edges;
    std::vector<Int2>   edge_adj_faces;
    std::vector<Int2>   bending_edge_adj_faces;

    std::vector<Int4> tetrahedrons;
    std::vector<uint> surface_verts;
    std::vector<Int2> surface_edges;
    std::vector<Int3> surface_faces;

    std::vector<std::string> material_names;
    std::vector<int>         material_ids;

    // For rendering only
    std::vector<Int3> invalid_faces;
    std::vector<Int3> invalid_normal_faces;
    std::vector<Int3> invalid_texcoord_faces;
    std::vector<int>  invalid_material_ids;

    bool has_uv = false;

    TriangleMeshData() {}
};

// static inline Int4 makeInt4()

void extract_surface_face_and_vert_from_tets(const std::vector<Float3>& input_position,
                                             const std::vector<Int4>&   input_tets,
                                             std::vector<uint>&         inner_tets,
                                             std::vector<uint>&         outer_tets,
                                             std::vector<Int3>&         surface_faces,
                                             std::vector<uint>&         surface_verts);

void extract_edges_from_surface(const std::vector<Int3>& input_faces,
                                std::vector<Int2>&       output_edges,
                                std::vector<Int4>&       output_bending_edges,
                                bool                     extract_bending_edge);


bool read_tet_file_t(std::string_view mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets);
bool read_tet_file_vtk(std::string_view mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets);
bool read_mesh_file(std::string_view mesh_name, TriangleMeshData& meshes);
// bool read_tet_file_t(std::string mesh_name, TetrahedralMeshData& meshes);

// template<typename Vert, typename Face>
bool saveToOBJ_combined(std::vector<std::vector<Float3>> const& sa_rendering_vertices,
                        std::vector<std::vector<Int3>> const&   sa_rendering_faces,
                        std::string_view                        addition_path,
                        std::string_view                        addition_str,
                        const uint                              frame);
};  // namespace SimMesh
