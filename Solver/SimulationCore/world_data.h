#pragma once

#include "Core/affine_position.h"
#include "MeshOperation/mesh_reader.h"
#include "SimulationCore/base_mesh.h"
#include "SimulationCore/physical_material.h"
#include <limits>

namespace lcs::Initializer
{

	enum struct FixedPointsType
	{
		None,
		FromIndices,
		FromFunction,
		Left,
		Right,
		Front,
		Back,
		Up,
		Down,
		LeftUp,
		LeftDown,
		LeftFront,
		LeftBack,
		RightUp,
		RightDown,
		RightFront,
		RightBack,
		FrontUp,
		FrontDown,
		BackUp,
		BackDown,
		All,
	};

	struct FixedPointDefaultAnimation
	{
		uint local_vid;

		bool   use_translate = false;
		float3 translate = luisa::make_float3(0.0f);

		bool   use_scale = false;
		float3 scale = luisa::make_float3(1.0f);

		bool   use_rotate = false;
		float3 rotCenter;
		float3 rotAxis;
		float  rotAngVelDeg = 0.0f;

		bool   use_setting_position = false;
		float3 setting_position;

		static float3 fn_affine_position(const lcs::Initializer::FixedPointDefaultAnimation& fixed_point,
			const float																		 time,
			const lcs::float3&																 pos)
		{
			auto fn_scale = [](const lcs::Initializer::FixedPointDefaultAnimation& fixed_point,
								const float										   time,
								const lcs::float3&								   pos)
			{ return (luisa::scaling(fixed_point.scale * time) * luisa::make_float4(pos, 1.0f)).xyz(); };
			auto fn_rotate = [](const lcs::Initializer::FixedPointDefaultAnimation& fixed_point,
								 const float										time,
								 const lcs::float3&									pos)
			{
				const float rotAngRad = time * fixed_point.rotAngVelDeg / 180.0f * float(lcs::Pi);
				const auto	relative_vec = pos - fixed_point.rotCenter;
				auto		matrix = luisa::rotation(fixed_point.rotAxis, rotAngRad);
				const auto	rotated_pos = matrix * luisa::make_float4(relative_vec, 1.0f);
				return fixed_point.rotCenter + rotated_pos.xyz();
			};
			auto fn_translate = [](const lcs::Initializer::FixedPointDefaultAnimation& fixed_point,
									const float										   time,
									const lcs::float3&								   pos)
			{
				return (luisa::translation(fixed_point.translate * time) * luisa::make_float4(pos, 1.0f)).xyz();
			};
			auto new_pos = pos;
			if (fixed_point.use_scale)
				new_pos = fn_scale(fixed_point, time, new_pos);
			if (fixed_point.use_rotate)
				new_pos = fn_rotate(fixed_point, time, new_pos);
			if (fixed_point.use_translate)
				new_pos = fn_translate(fixed_point, time, new_pos);
			return new_pos;
		};
	};

	FixedPointsType parse_fixed_method_py(const std::string_view& s);

	struct MakeFixedPointsInterface
	{
		FixedPointsType			   method = FixedPointsType::All;
		FixedPointDefaultAnimation fixed_info;
		float					   range = 0.001f;
		void*					   data_ptr = nullptr;
	};

	struct WorldData
	{
		// std::string file_path;
		std::string model_name;
		float3		translation = luisa::make_float3(0.0f, 0.0f, 0.0f);
		float3		rotation = luisa::make_float3(0.0f * lcs::Pi); // Rotation in x-channel means rotate along with x-axis
		float3		scale = luisa::make_float3(1.0f);

		Material::MaterialVariant physics_material; // Maybe we can use Polymorphism

		std::vector<uint>						fixed_point_indices;
		std::vector<FixedPointDefaultAnimation> fixed_point_default_animations;

		Material::MaterialType	  material_type = Material::MaterialType::Cloth;
		SimMesh::TriangleMeshData input_mesh;

		uint registration_index = std::numeric_limits<uint>::max();
		uint sorted_index = std::numeric_limits<uint>::max();

	public:
		const SimMesh::TriangleMeshData& get_mesh() const { return input_mesh; }

		template <typename T>
		bool holds() const
		{
			return std::holds_alternative<T>(physics_material);
		}
		template <typename T>
		T& get_material()
		{
			return std::get<T>(physics_material);
		}
		template <typename T>
		const T& get_material() const
		{
			return std::get<T>(physics_material);
		}

		WorldData()
			: model_name("unnamed")
			, material_type(Material::MaterialType::Particle)
		{
		}
		WorldData& set_name(const std::string_view& model_name)
		{
			this->model_name = model_name;
			return *this;
		}
		// WorldData& set_file_path(const std::string_view& file_path)
		// {
		// 	this->file_path = file_path;
		// 	return *this;
		// }
		WorldData& set_material_type(const Material::MaterialType& sim_type)
		{
			this->material_type = sim_type;
			return *this;
		}
		WorldData& add_fixed_point_from_method(const MakeFixedPointsInterface& info);
		WorldData& add_fixed_point_from_indices(const std::vector<uint>& indices);

		// template <typename Int, typename Real>
		WorldData& load_mesh_from_array(const std::vector<std::array<float, 3>>& vertices, const std::vector<std::array<uint, 3>>& faces);
		WorldData& load_mesh_from_path(const std::string_view& path);

		WorldData& load_tet_mesh_from_array(
			const std::vector<std::array<float, 3>>& vertices,
			const std::vector<std::array<uint, 4>>&	 tets);
		WorldData& load_tet_mesh_from_path(const std::string_view& path);
		WorldData& set_physics_material_tet(
			float						   youngs_modulus = Material::TetMaterial::default_youngs_modulus(),
			float						   poisson_ratio = Material::TetMaterial::default_poisson_ratio(),
			Material::ConstitutiveModelTet model = Material::TetMaterial::default_model())
		{
			material_type = Material::MaterialType::Tetrahedral;
			Material::TetMaterial mat;
			mat.model = model;
			mat.youngs_modulus = youngs_modulus;
			mat.poisson_ratio = poisson_ratio;
			physics_material = mat;
			return *this;
		}

		WorldData& set_translation(const float3& t)
		{
			this->translation = t;
			return *this;
		}
		WorldData& set_translation(const float x, const float y, const float z)
		{
			this->translation = luisa::make_float3(x, y, z);
			return *this;
		}
		WorldData& set_rotation(const float3& r)
		{
			this->rotation = r;
			return *this;
		}
		WorldData& set_rotation(const float axis_x, const float axis_y, const float axis_z)
		{
			this->rotation = luisa::make_float3(axis_x, axis_y, axis_z);
			return *this;
		}
		WorldData& set_scale(const float3& s)
		{
			this->scale = s;
			return *this;
		}
		WorldData& set_scale(const float scale_x, const float scale_y, const float scale_z)
		{
			this->scale = luisa::make_float3(scale_x, scale_y, scale_z);
			return *this;
		}
		WorldData& set_scale(const float s)
		{
			this->scale = luisa::make_float3(s);
			return *this;
		}
		float3 get_rest_position(const float3& model_pos) const
		{
			float4x4 model_matrix = lcs::make_model_matrix(translation, rotation, scale);
			return lcs::affine_position(model_matrix, model_pos);
		}
		float3 get_rest_position(const uint local_vid) const
		{
			if (local_vid >= input_mesh.model_positions.size())
			{
				throw std::runtime_error("Vertex ID out of range in get_rest_position");
			}
			const auto model_pos = input_mesh.model_positions[local_vid];
			return get_rest_position(luisa::make_float3(model_pos[0], model_pos[1], model_pos[2]));
		}
		WorldData& set_physics_material(const Material::MaterialVariant& mat)
		{
			this->physics_material = mat;
			return *this;
		}

		// template <typename T>
		// T* get_if()
		// {
		//     return std::get_if<T>(&physics_material);
		// }

		bool get_is_shell() const
		{
			return std::visit([](auto const& m) noexcept
				{ return m.is_shell; },
				physics_material);
		}
		float get_mass() const
		{
			return std::visit([](auto const& m) noexcept
				{ return m.mass; },
				physics_material);
		}
		float get_density() const
		{
			return std::visit([](auto const& m) noexcept
				{ return m.density; },
				physics_material);
		}
		float get_d_hat() const
		{
			return std::visit([](auto const& m) noexcept
				{ return m.d_hat; },
				physics_material);
		}
		float get_contact_offset() const
		{
			return std::visit([](auto const& m) noexcept
				{ return m.contact_offset; },
				physics_material);
		}
		float get_friction_mu() const
		{
			return std::visit([](auto const& m) noexcept
				{ return m.friction_mu; },
				physics_material);
		}
		float get_thickness() const
		{
			if (get_is_shell())
			{
				if (holds<Material::ClothMaterial>())
				{
					return get_material<Material::ClothMaterial>().thickness;
				}
				else if (holds<Material::RigidMaterial>())
				{
					return get_material<Material::RigidMaterial>().thickness;
				}
				else if (holds<Material::RodMaterial>())
				{
					return get_material<Material::RodMaterial>().radius * 2.0f;
				}
				else
				{
					LUISA_ERROR("Mesh {} should not have thickness", get_model_name());
					return 0.0f;
				}
			}
			else
			{
				return 0.0f;
			}
		}
		// std::string get_model_path() const
		// {
		// 	return file_path;
		// }
		std::string get_model_name() const
		{
			return model_name;
		}
		uint get_registration_index() const
		{
			return registration_index;
		}
		uint get_sorted_index() const
		{
			return sorted_index;
		}

		void update_default_vertex_animations(const float time, std::vector<Animation::PerVertexAnimation>& vertex_animations);
		void update_default_body_animations(const float time, Animation::PerBodyAnimation& body_animation);

		void							  get_rest_positions(std::vector<std::array<float, 3>>& rest_positions) const;
		std::vector<std::array<float, 3>> get_rest_positions() const;

	private:
		void set_pinned_verts_from_norm_position(const std::function<bool(const float3&)>& func, const FixedPointDefaultAnimation& info = FixedPointDefaultAnimation());
		void set_pinned_verts_from_functions(const std::function<bool(uint)>& func, const FixedPointDefaultAnimation& info = FixedPointDefaultAnimation());
		void set_pinned_verts_from_indices(const std::vector<uint>& indices, const FixedPointDefaultAnimation& info = FixedPointDefaultAnimation());
	};

} // namespace lcs::Initializer