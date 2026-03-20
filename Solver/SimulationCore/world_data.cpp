#include "world_data.h"
#include "Core/affine_position.h"
#include "Core/float_nxn.h"
#include "MeshOperation/mesh_reader.h"
#include "Initializer/initializer_utils.h"
#include "Utils/cpu_parallel.h"
#include <algorithm>
#include <numeric>

namespace lcs::Initializer // WorldData
{
	static std::vector<SimMesh::Int2> extract_all_edges_from_tets(const std::vector<SimMesh::Int4>& tets)
	{
		std::vector<SimMesh::Int2> edges;
		edges.reserve(tets.size() * 6);

		auto push_edge = [&](uint v0, uint v1)
		{
			if (v0 > v1)
				std::swap(v0, v1);
			edges.push_back({ v0, v1 });
		};

		for (const auto& tet : tets)
		{
			push_edge(tet[0], tet[1]);
			push_edge(tet[0], tet[2]);
			push_edge(tet[0], tet[3]);
			push_edge(tet[1], tet[2]);
			push_edge(tet[1], tet[3]);
			push_edge(tet[2], tet[3]);
		}

		std::sort(edges.begin(),
			edges.end(),
			[](const SimMesh::Int2& lhs, const SimMesh::Int2& rhs)
			{
				if (lhs[0] != rhs[0])
					return lhs[0] < rhs[0];
				return lhs[1] < rhs[1];
			});
		edges.erase(std::unique(edges.begin(),
						edges.end(),
						[](const SimMesh::Int2& lhs, const SimMesh::Int2& rhs)
						{ return lhs[0] == rhs[0] && lhs[1] == rhs[1]; }),
			edges.end());

		return edges;
	}

	struct AABB
	{
		float3 packed_min;
		float3 packed_max;
		AABB   operator+(const AABB& input_aabb) const
		{
			AABB tmp;
			tmp.packed_min = lcs::min_vec(packed_min, input_aabb.packed_min);
			tmp.packed_max = lcs::max_vec(packed_max, input_aabb.packed_max);
			return tmp;
		}
		AABB()
			: packed_min(float3(Float_max))
			, packed_max(float3(-Float_max))
		{
		}
		AABB(const float3& pos)
			: packed_min(pos)
			, packed_max(pos)
		{
		}
	};

	// parse FixedPointsType from string (same names used in JSON/config)
	FixedPointsType parse_fixed_method_py(const std::string_view& s)
	{
		constexpr std::array<std::pair<std::string_view, FixedPointsType>, 14> table = {
			std::pair{ "None", FixedPointsType::None },
			std::pair{ "FromIndices", FixedPointsType::FromIndices },
			std::pair{ "FromFunction", FixedPointsType::FromFunction },
			std::pair{ "Left", FixedPointsType::Left },
			std::pair{ "Right", FixedPointsType::Right },
			std::pair{ "Front", FixedPointsType::Front },
			std::pair{ "Back", FixedPointsType::Back },
			std::pair{ "Up", FixedPointsType::Up },
			std::pair{ "Down", FixedPointsType::Down },
			std::pair{ "LeftBack", FixedPointsType::LeftBack },
			std::pair{ "LeftFront", FixedPointsType::LeftFront },
			std::pair{ "RightBack", FixedPointsType::RightBack },
			std::pair{ "RightFront", FixedPointsType::RightFront },
			std::pair{ "All", FixedPointsType::All },
		};

		for (const auto& [name, method] : table)
		{
			if (name == std::string_view{ s })
				return method;
		}
		return FixedPointsType::All;
	}

	void WorldData::set_pinned_verts_from_functions(const std::function<bool(uint)>& func,
		const FixedPointDefaultAnimation&											 fixed_info)
	{
		for (uint vid = 0; vid < input_mesh.model_positions.size(); vid++)
		{
			if (func(vid))
			{
				fixed_point_indices.emplace_back(vid);
				fixed_point_default_animations.emplace_back(fixed_info).local_vid = vid;
			}
		}
	}
	void WorldData::set_pinned_verts_from_norm_position(const std::function<bool(const float3&)>& func,
		const FixedPointDefaultAnimation&														  fixed_info)
	{
		AABB local_aabb = CpuParallel::parallel_for_and_reduce_sum<AABB>(
			0,
			input_mesh.model_positions.size(),
			[&](const uint vid)
			{
				auto   read_pos = input_mesh.model_positions[vid];
				float3 pos = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
				return AABB(pos);
			});

		auto pos_min = local_aabb.packed_min;
		auto pos_max = local_aabb.packed_max;
		auto pos_dim_inv = 1.0f / luisa::max(pos_max - pos_min, 0.0001f);

		for (uint vid = 0; vid < input_mesh.model_positions.size(); vid++)
		{
			auto   read_pos = input_mesh.model_positions[vid];
			float3 pos = luisa::make_float3(read_pos[0], read_pos[1], read_pos[2]);
			float3 norm_pos = (pos - pos_min) * pos_dim_inv;

			if (func(norm_pos))
			{
				fixed_point_indices.emplace_back(vid);
				fixed_point_default_animations.emplace_back(fixed_info).local_vid = vid;
			}
		}
	}
	void WorldData::set_pinned_verts_from_indices(const std::vector<uint>& indices,
		const FixedPointDefaultAnimation&								   fixed_info)
	{
		for (const uint vid : indices)
		{
			fixed_point_indices.emplace_back(vid);
			fixed_point_default_animations.emplace_back(fixed_info).local_vid = vid;
		}
	}
	WorldData& WorldData::add_fixed_point_from_indices(const std::vector<uint>& indices)
	{
		set_pinned_verts_from_indices(indices);
		return *this;
	}
	WorldData& WorldData::add_fixed_point_from_method(const MakeFixedPointsInterface& fixed_point_func)
	{
		auto from_norm_position = [&](const std::function<bool(const float3&)>& func,
									  const FixedPointDefaultAnimation&			info = FixedPointDefaultAnimation())
		{ set_pinned_verts_from_norm_position(func, info); };

		{
			const auto& range = fixed_point_func.range;
			if (fixed_point_func.method == FixedPointsType::All)
			{
				from_norm_position([](const float3& norm_pos)
					{ return true; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::Left)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::Right)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::Front)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.z < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::Back)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.z > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::Up)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.y > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::Down)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.y < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::LeftUp)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x < range && norm_pos.y > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::LeftDown)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x < range && norm_pos.y < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::LeftFront)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x < range && norm_pos.z > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::LeftBack)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x < range && norm_pos.z < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::RightUp)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x > 1.0f - range && norm_pos.y > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::RightDown)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x > 1.0f - range && norm_pos.y < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::RightFront)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x > 1.0f - range && norm_pos.z > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::RightBack)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.x > 1.0f - range && norm_pos.z < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::FrontUp)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.z < range && norm_pos.y > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::FrontDown)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.z < range && norm_pos.y < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::BackUp)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.z > 1.0f - range && norm_pos.y > 1.0f - range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::BackDown)
			{
				from_norm_position([range = fixed_point_func.range](const float3& norm_pos)
					{ return norm_pos.z > 1.0f - range && norm_pos.y < range; },
					fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::FromIndices)
			{
				auto indices = *((std::vector<uint>*)fixed_point_func.data_ptr);
				set_pinned_verts_from_indices(indices, fixed_point_func.fixed_info);
			}
			else if (fixed_point_func.method == FixedPointsType::FromFunction)
			{
				auto func = *((std::function<bool(uint)>*)fixed_point_func.data_ptr);
				set_pinned_verts_from_functions(func, fixed_point_func.fixed_info);
			}
			else
			{
				LUISA_ERROR("Unsupported FixedPointsType {} in provided fix point info.",
					int(fixed_point_func.method));
			}
		}

		return *this;
	}
	void WorldData::update_default_vertex_animations(const float time, std::vector<Animation::PerVertexAnimation>& vertex_animation)
	{
		vertex_animation.resize(fixed_point_default_animations.size());
		for (uint index = 0; index < fixed_point_default_animations.size(); index++)
		{
			const auto& fixed_info = fixed_point_default_animations[index];
			const uint	local_vid = fixed_info.local_vid;
			const auto	model_pos = input_mesh.model_positions[local_vid];
			auto		transform_matrix = lcs::make_model_matrix(translation, rotation, scale);
			const auto	rest_pos =
				(transform_matrix * luisa::make_float4(model_pos[0], model_pos[1], model_pos[2], 1.0f)).xyz();

			auto target = FixedPointDefaultAnimation::fn_affine_position(fixed_info, time, rest_pos);
			vertex_animation[index] = { .vertex_id = local_vid, .translation = { target.x, target.y, target.z } };
		}
	}
	void WorldData::update_default_body_animations(const float time, Animation::PerBodyAnimation& body_animation)
	{
		const auto& fixed_info = fixed_point_default_animations.front();
		auto		transform_matrix = lcs::make_model_matrix(translation, rotation, scale);
		auto		rest_pos = (transform_matrix * luisa::make_float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz();
		auto		target = FixedPointDefaultAnimation::fn_affine_position(fixed_info, time, rest_pos);
		body_animation = {
			.dof_start = get_registration_index(),
			.translation = { target.x, target.y, target.z },
			.rotation = { rotation.x, rotation.y, rotation.z }
		};
	}
	void WorldData::get_rest_positions(std::vector<std::array<float, 3>>& rest_positions) const
	{
		rest_positions.resize(input_mesh.model_positions.size());
		auto transform_matrix = lcs::make_model_matrix(translation, rotation, scale);
		for (uint vid = 0; vid < input_mesh.model_positions.size(); vid++)
		{
			const auto model_pos = input_mesh.model_positions[vid];
			auto	   rest_pos =
				(transform_matrix * luisa::make_float4(model_pos[0], model_pos[1], model_pos[2], 1.0f)).xyz();
			std::array<float, 3> output;
			output[0] = rest_pos.x;
			output[1] = rest_pos.y;
			output[2] = rest_pos.z;
			rest_positions[vid] = output;
		}
	}
	std::vector<std::array<float, 3>> WorldData::get_rest_positions() const
	{
		std::vector<std::array<float, 3>> rest_positions;
		get_rest_positions(rest_positions);
		return rest_positions;
	}

	// template <typename Int, typename Real>
	WorldData& WorldData::load_mesh_from_array(const std::vector<std::array<float, 3>>& vertices, const std::vector<std::array<uint, 3>>& faces)
	{
		input_mesh.model_positions.resize(vertices.size());
		input_mesh.faces.resize(faces.size());
		for (size_t i = 0; i < vertices.size(); i++)
		{
			input_mesh.model_positions[i] = { float(vertices[i][0]), float(vertices[i][1]), (vertices[i][2]) };
		}
		for (size_t i = 0; i < faces.size(); i++)
		{
			input_mesh.faces[i] = { uint(faces[i][0]), uint(faces[i][1]), uint(faces[i][2]) };
		}
		SimMesh::extract_edges_from_surface(input_mesh.faces, input_mesh.edges, input_mesh.dihedral_edges, true);
		input_mesh.surface_faces = input_mesh.faces;
		input_mesh.surface_verts.resize(vertices.size());
		std::iota(input_mesh.surface_verts.begin(), input_mesh.surface_verts.end(), 0);
		input_mesh.surface_edges = input_mesh.edges;
		return *this;
	}
	WorldData& WorldData::load_mesh_from_path(const std::string_view& path)
	{
		bool succ = SimMesh::read_mesh_file(path, input_mesh);
		input_mesh.surface_edges = input_mesh.edges;
		input_mesh.surface_faces = input_mesh.faces;
		input_mesh.surface_verts.resize(input_mesh.model_positions.size());
		std::iota(input_mesh.surface_verts.begin(), input_mesh.surface_verts.end(), 0);
		return *this;
	}

	// ---------------------------------------------------------------------------
	// Tetrahedral mesh loaders
	// ---------------------------------------------------------------------------

	WorldData& WorldData::load_tet_mesh_from_array(
		const std::vector<std::array<float, 3>>& vertices,
		const std::vector<std::array<uint, 4>>&	 tets)
	{
		// 1. Copy vertex positions into input_mesh (same field used for all types)
		input_mesh.model_positions.resize(vertices.size());
		for (size_t i = 0; i < vertices.size(); i++)
			input_mesh.model_positions[i] = { vertices[i][0], vertices[i][1], vertices[i][2] };

		// 2. Copy tetrahedra
		input_mesh.tetrahedrons.resize(tets.size());
		for (size_t i = 0; i < tets.size(); i++)
			input_mesh.tetrahedrons[i] = { tets[i][0], tets[i][1], tets[i][2], tets[i][3] };

		// 3. Extract surface faces, surface verts from tet topology
		std::vector<uint> inner_tets, outer_tets;
		SimMesh::extract_surface_face_and_vert_from_tets(
			input_mesh.model_positions,
			input_mesh.tetrahedrons,
			inner_tets,
			outer_tets,
			input_mesh.surface_faces,
			input_mesh.surface_verts);

		// 4. Use surface_faces as the triangle mesh faces (for rendering, collision, etc.)
		input_mesh.faces = input_mesh.surface_faces;

		// 5. Extract surface edges from surface faces (for collision/rendering adjacency)
		SimMesh::extract_edges_from_surface(input_mesh.surface_faces, input_mesh.surface_edges, input_mesh.dihedral_edges, true);

		// 6. Extract full tet edges (including interior edges) for spring constraints
		input_mesh.edges = extract_all_edges_from_tets(input_mesh.tetrahedrons);

		// 7. Mark material type
		material_type = Material::MaterialType::Tetrahedral;

		// LUISA_INFO("Loaded tet mesh from array: {} verts, {} edges, {} tets => {} surface verts, {} surface faces, {} surface edges",
		// 	vertices.size(), input_mesh.edges.size(), tets.size(),
		// 	input_mesh.surface_verts.size(), input_mesh.surface_faces.size(), input_mesh.surface_edges.size());

		return *this;
	}

	WorldData& WorldData::load_tet_mesh_from_path(const std::string_view& path)
	{
		std::vector<SimMesh::Float3> positions;
		std::vector<SimMesh::Int4>	 raw_tets;

		// Try .t format first, then .vtk
		bool succ = SimMesh::read_tet_file_t(path, positions, raw_tets);
		if (!succ)
			succ = SimMesh::read_tet_file_vtk(path, positions, raw_tets);

		if (!succ)
		{
			LUISA_ERROR("Failed to read tetrahedral mesh from path: {}", path);
			return *this;
		}

		// Convert to the array format expected by load_tet_mesh_from_array
		std::vector<std::array<float, 3>> verts(positions.size());
		for (size_t i = 0; i < positions.size(); i++)
			verts[i] = { positions[i][0], positions[i][1], positions[i][2] };

		std::vector<std::array<uint, 4>> tets(raw_tets.size());
		for (size_t i = 0; i < raw_tets.size(); i++)
			tets[i] = { raw_tets[i][0], raw_tets[i][1], raw_tets[i][2], raw_tets[i][3] };

		return load_tet_mesh_from_array(verts, tets);
	}

} // namespace lcs::Initializer