#include "MeshOperation/mesh_reader.h"
#include <filesystem>

namespace SimMesh
{

static inline Float3 makeFloat3(const float v1, const float v2, const float v3)
{
    return Float3{v1, v2, v3};
}
static inline Int3 makeInt3(const uint v1, const uint v2, const uint v3)
{
    return Int3{v1, v2, v3};
}
static inline Int4 makeInt4(const uint v1, const uint v2, const uint v3, const uint v4)
{
    return Int4{v1, v2, v3, v4};
}


bool read_mesh_file(std::string mesh_name, TriangleMeshData& mesh_data)
{
    std::string err, warn;


    std::string full_path = mesh_name;

    std::string mtl_path = std::filesystem::path(full_path).replace_extension(".mtl").string();
    

    tinyobj::ObjReader reader; tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = std::filesystem::path(full_path).parent_path().string();
    if (!reader.ParseFromFile(full_path, reader_config)) 
    {
        if (!reader.Warning().empty()) 
        {
            std::cerr << std::format("Warning : {}", reader.Warning());
        }
        if (!reader.Error().empty()) 
        {
            std::cerr << std::format("Warning : {}", reader.Error());
        }
        exit(1);
    }


    MeshAttrib mesh_attrib = reader.GetAttrib();
    MeshShape mesh_shape = reader.GetShapes();
    MeshMat material = reader.GetMaterials();
    
    
    const uint num_verts = static_cast<uint>(mesh_attrib.vertices.size() / 3);
    uint num_faces = 0;
    for (auto& sub_obj : mesh_shape)
    {
        num_faces += sub_obj.mesh.indices.size() / 3;
    }

    mesh_data.model_positions.resize(num_verts);
    mesh_data.faces.reserve(num_faces);
    mesh_data.normal_faces.reserve(num_faces);
    mesh_data.texcoord_faces.reserve(num_faces);
    
    mesh_data.material_ids.reserve(num_faces);
    mesh_data.material_names.reserve(material.size());

    CpuParallel::parallel_for(0, num_verts, [&](const uint vid)
    {
        Float3 local_pos = Float3{
            mesh_attrib.vertices[vid * 3 + 0], 
            mesh_attrib.vertices[vid * 3 + 1], 
            mesh_attrib.vertices[vid * 3 + 2]};
        mesh_data.model_positions[vid] = local_pos;
    });

    const bool has_uv = !mesh_attrib.texcoords.empty();
    if (has_uv)
    {
        mesh_data.has_uv = true; // fast_format(" NumUV = {}, NumVerts = {}", mesh_attrib.texcoords.size() / 2, num_verts);

        const uint num_uvs = mesh_attrib.texcoords.size() / 2;
        mesh_data.uv_positions.resize(num_uvs);
        mesh_data.uv_to_vert_map.resize(num_uvs);

        CpuParallel::parallel_for(0, num_uvs, [&](const uint vid)
        {
            Float2 uv_pos = Float2{
                mesh_attrib.texcoords[vid * 2 + 0], 
                mesh_attrib.texcoords[vid * 2 + 1]};
            mesh_data.uv_positions[vid] = uv_pos;
            mesh_data.uv_to_vert_map[vid] = vid;
        });
    }
    else 
    {
        mesh_data.has_uv = false;

        // const uint num_uvs = num_verts;
        // mesh_data.uv_positions.resize(num_uvs);
        // mesh_data.uv_to_vert_map.resize(num_uvs);

        // // Generate UV By Projection Into Diagonal Plane of AABB 
        // const AABB local_aabb = parallel_for_and_reduce_sum<AABB>(0, num_verts, [&](const uint vid)
        // {
        //     return mesh_data.model_positions[vid];
        // });
        // const Float3 pos_min = local_aabb.min_pos;
        // const Float3 pos_max = local_aabb.max_pos;
        // const Float3 pos_dim = local_aabb.range();

        // struct dim_range{
        //     uint axis_idx;
        //     float axis_width;
        //     dim_range(uint idx, float width) : axis_idx(idx), axis_width(width) {}
        // };
        // dim_range tmp[3]{ dim_range(0u, pos_dim[0]), dim_range(1u, pos_dim[1]), dim_range(2u, pos_dim[2]) };
        // std::sort(tmp, tmp + 3, [](const dim_range& a, const dim_range& b){
        //     return a.axis_width < b.axis_width;
        // });
        
        // Float3 tmp_e2 = Zero3;
        // tmp_e2[tmp[0].axis_idx] = tmp[0].axis_width;
        // tmp_e2[tmp[1].axis_idx] = tmp[1].axis_width;
        // Float3 tmp_e1 = Zero3;
        // tmp_e1[tmp[2].axis_idx] = tmp[2].axis_width; // 将最大的跨度作为主轴，这样不会出现投影均为0的问题

        // Float3 tmp_normal = normalize_vec(cross_vec(tmp_e1, tmp_e2));
        // CpuParallel::parallel_for(0, num_verts, [&](uint vid)
        // {
        //     Float3 pos = mesh_data.model_positions[vid];
        //     float distance = dot_vec(tmp_normal, pos - pos_min); // 向量由面指向点
        //     Float3 proj_p = pos - distance * tmp_normal;
        //     Float3 tmp_vec = pos - pos_min;
        //     float u = length_vec(project_vec(tmp_vec, tmp_e1));
        //     float v = length_vec(project_vec(tmp_vec, tmp_e2));

        //     mesh_data.uv_positions[vid] = makeFloat2(u, v);
        //     mesh_data.uv_to_vert_map[vid] = vid;
        // });
    }

    uint face_prefix = 0;
    for (size_t submesh_idx = 0; submesh_idx < mesh_shape.size(); submesh_idx++) 
    {
        const auto& sub_mesh = mesh_shape[submesh_idx];

        auto& face_list = sub_mesh.mesh.indices;
        const uint curr_num_faces = face_list.size() / 3;
        
        for (uint fid = 0; fid < curr_num_faces; fid++)
        {
            tinyobj::index_t v0 = face_list[fid * 3 + 0];
            tinyobj::index_t v1 = face_list[fid * 3 + 1];
            tinyobj::index_t v2 = face_list[fid * 3 + 2];
            
            if (mesh_data.has_uv)
            {
                mesh_data.uv_to_vert_map[v0.texcoord_index] = v0.vertex_index;
                mesh_data.uv_to_vert_map[v1.texcoord_index] = v1.vertex_index;
                mesh_data.uv_to_vert_map[v2.texcoord_index] = v2.vertex_index;
            }
            
            int material_id = sub_mesh.mesh.material_ids[fid];
            {
                Int3 orig_face = makeInt3(
                    v0.vertex_index, 
                    v1.vertex_index, 
                    v2.vertex_index);

                if (orig_face[0] == orig_face[1] || orig_face[0] == orig_face[2] || orig_face[1] == orig_face[2])
                {   
                    std::cerr << std::format("Illigal Face Input {} : {}/{}/{}", fid, (orig_face[0]), orig_face[1], orig_face[2]);  
                    mesh_data.invalid_material_ids.push_back(material_id);
                    mesh_data.invalid_faces.push_back(makeInt3(v0.vertex_index, v1.vertex_index, v2.vertex_index));
                    mesh_data.invalid_normal_faces.push_back(makeInt3(v0.normal_index, v1.normal_index, v2.normal_index));
                    mesh_data.invalid_texcoord_faces.push_back(makeInt3(v0.texcoord_index, v1.texcoord_index, v2.texcoord_index));
                    continue;
                }
                else 
                {
                    mesh_data.material_ids.push_back(material_id);
                    mesh_data.faces.push_back(makeInt3(v0.vertex_index, v1.vertex_index, v2.vertex_index));
                    mesh_data.normal_faces.push_back(makeInt3(v0.normal_index, v1.normal_index, v2.normal_index));
                    mesh_data.texcoord_faces.push_back(makeInt3(v0.texcoord_index, v1.texcoord_index, v2.texcoord_index));
                }  
            }
        }
        face_prefix += curr_num_faces;

        // std::cout << "Shape of submesh " << submesh_idx << " : " << mesh_shape[submesh_idx].name << std::endl;
    }
    {
        for (auto& mat : material)
        {
            mesh_data.material_names.push_back(mat.name);
            // fast_format("Materials : {} ", mat.name); // Can Not Read Materials That Have Several Entities, tinyobjloader Can Not Capture The Name
        }
    }

    extract_edges_from_surface<true>(mesh_data.faces, mesh_data.edges, mesh_data.bending_edges);
    
    // fast_format("   Readed Mesh Data {} : numSubMesh = {}, numVerts = {}, numFaces = {}, numEdges = {}, numBendingEdges = {}", 
    //     mesh_name, mesh_shape.size(), num_verts, num_faces, mesh_data.edges.size(), mesh_data.bending_edges.size());

    return true;
}
bool read_tet_file_t(std::string mesh_name, std::vector<Float3>& position, std::vector<Int4>& tets) 
{
    std::string err, warn;
    std::string full_path = mesh_name;

    bool load = true;
    {
        std::ifstream infile(full_path);
        if (!infile.is_open()) 
        {
            std::cerr << "Error opening file " << full_path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(infile, line)) 
        {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "Vertex") 
            {
                int index;
                float x, y, z;
                if (!(iss >> index >> x >> y >> z)) 
                {
                    std::cerr << "Error reading vertex data from file " << full_path << std::endl;
                    return false;
                }
                position.emplace_back(makeFloat3(x, y, z));
            }
            else if (prefix == "Tet") 
            {
                int index, i1, i2, i3, i4;
                if (!(iss >> index >> i1 >> i2 >> i3 >> i4)) 
                {
                    std::cerr << "Error reading tetrahedron data from file " << full_path << std::endl;
                    return false;
                }
                tets.emplace_back(makeInt4(i1, i2, i3, i4));
            }
        }

        infile.close();
    }
    return true;
}
bool read_tet_file_vtk(std::string file_name, std::vector<Float3>& positions, std::vector<Int4>& tets) 
{
    std::string full_path = file_name;

    std::ifstream infile(full_path);
    if (!infile.is_open()) 
    {
        std::cerr << "Error opening file: " << full_path << std::endl;
        return false;
    }

    std::string line;
    bool reading_points = false;
    bool reading_cells = false;
    size_t expected_points = 0, expected_cells = 0;

    while (std::getline(infile, line)) 
    {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "POINTS") 
        {
            // Read the number of points and data type (e.g., double/float)
            iss >> expected_points;
            std::string data_type;
            iss >> data_type;

            positions.reserve(expected_points);
            
            for (int i = 0; i < expected_points; ++i) 
            {
                float x, y, z;
                if (!(infile >> x >> y >> z)) 
                {
                    std::cerr << "Error reading vertex coordinates from file " << full_path << std::endl;
                    return false;
                }
                positions.emplace_back(makeFloat3(x, y, z));  // 假设 makeFloat3 用于将坐标存储为 Float3 类型
            }

        } 
        else if (keyword == "CELLS") 
        {
            // Read the number of cells and total numbers of indices
            iss >> expected_cells;
            size_t total_indices;
            iss >> total_indices;

            for (int i = 0; i < expected_cells; ++i) 
            {
                int num_points_in_cell;  // 四面体有 4 个顶点
                int p1, p2, p3, p4;
                if (!(infile >> num_points_in_cell >> p1 >> p2 >> p3 >> p4)) 
                {
                    std::cerr << "Error reading cell data from file " << full_path << std::endl;
                    return false;
                }
                // 检查四面体顶点数是否为 4
                if (num_points_in_cell != 4) 
                {
                    std::cerr << "Invalid number of points in cell " << i << std::endl;
                    return false;
                }
                tets.emplace_back(makeInt4(p1, p2, p3, p4));  // 假设 makeInt4 用于存储四面体索引
            }
        } 
        else if (keyword == "CELL_TYPES") 
        {
            // Stop reading as we no longer need the cell types for tetrahedra
            break;
        } 
    }

    infile.close();

    if (positions.empty() || tets.empty()) { std::cerr << std::format("Reading Result is Empty!!! Actual Get {} Verts And {} Tetrahedrals", positions.size(), tets.size()); exit(0); }

    // fast_format("   Readed Tetrahedral Data {} : numVerts = {}, numFaces = {}, numEdges = {}, numBendingEdges = {}", 
    //     file_name, positions.size(), , num_faces, mesh_data.edges.size(), mesh_data.bending_edges.size());

    return true;
}


// template<typename Vert, typename Face>
bool saveToOBJ_combined(
    std::vector<std::vector<Float3>> sa_rendering_vertices, 
    std::vector<std::vector<Int3>> sa_rendering_faces, 
    const std::string& addition_str, const uint frame) 
{
    const std::string filename = std::format("frame_{}{}.obj", frame, addition_str);
    std::string full_directory = std::string(LCSV_RESOURCE_PATH) + std::string("/OutputMesh/");
    std::string full_path = full_directory + filename;

    // Ensure the directory exists
    {
        std::filesystem::path dir_path(full_directory);
        if (!std::filesystem::exists(dir_path)) 
        {
            try 
            {
                std::filesystem::create_directories(dir_path);
                std::cout << "Created directory: " << dir_path << std::endl;
            } 
            catch (const std::filesystem::filesystem_error& e) 
            {
                std::cerr << "Error creating directory: " << e.what() << std::endl;
                return false;
            }
        }
    }
    
    std::ofstream file(full_path, std::ios::out);
    if (file.is_open()) 
    {
        file << "# Simulated by LuisaComputeSolver" << std::endl;
        uint prefix_vid = 1;
        for (uint meshIdx = 0; meshIdx < sa_rendering_vertices.size(); meshIdx++) 
        {
            const auto& curr_vertices = sa_rendering_vertices[meshIdx];
            const auto& curr_faces = sa_rendering_faces[meshIdx];

            file << "o mesh_" << meshIdx << std::endl;
            for (uint vid = 0; vid < curr_vertices.size(); vid++) 
            {
                const auto vertex = curr_vertices[vid];
                file << "v " 
                    << vertex[0] << " " 
                    << vertex[1] << " " 
                    << vertex[2] << std::endl;
            }

            for (uint fid = 0; fid < curr_faces.size(); fid++) 
            {
                const auto f = curr_faces[fid];
                file << "f " 
                    << (prefix_vid + f[0]) << " " 
                    << (prefix_vid + f[1]) << " " 
                    << (prefix_vid + f[2]) << std::endl;
            }
            prefix_vid += curr_vertices.size();
        }

        
        file.close();
        std::cout << "OBJ file saved: " << full_path << std::endl;
 
        return true;
    } 
    else 
    {
        std::cerr << "Unable to open file: " << full_path << std::endl;
        return false;
    }
}

};