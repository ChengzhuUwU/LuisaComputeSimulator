#include "SimulationSolver/solver_interface.h"
#include "CollisionDetector/cipc_kernel.hpp"
#include "CollisionDetector/friction_kernel.hpp"
#include <stdexcept>
#include "Core/affine_position.h"
#include "Core/scalar.h"
#include "Energies/abd_inertia_energy.h"
#include "Energies/abd_ortho_energy.h"
#include "Energies/bending_energy_kernel.h"
#include "Energies/ground_collision_energy.h"
#include "Energies/soft_inertia_energy.h"
#include "Energies/spring_energy.h"
#include "Energies/stretch_face_energy.h"
#include "Energy/bending_energy.h"
#include "Energy/stretch_energy.h"
#include "Initializer/init_collision_data.h"
#include "Initializer/init_sim_data.h"
#include "Utils/cpu_parallel.h"
#include "Utils/reduce_helper.h"
#include "SimulationCore/scene_params.h"
#include "MeshOperation/mesh_reader.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/builtin.h"
#include "luisa/runtime/buffer.h"
#include "luisa/runtime/stream.h"
#include <numeric>

namespace lcs
{

	void SolverInterface::set_data_pointer(SolverData& solver_data, SolverHelper& solver_helper)
	{
		// Data pointer
		this->host_mesh_data = &solver_data.host_mesh_data;
		this->host_sim_data = &solver_data.host_sim_data;

		this->mesh_data = &solver_data.mesh_data;
		this->sim_data = &solver_data.sim_data;

		this->lbvh_data_face = &solver_data.lbvh_data_face;
		this->lbvh_data_edge = &solver_data.lbvh_data_edge;

		this->host_collision_data = &solver_data.host_collision_data;
		this->collision_data = &solver_data.collision_data;

		// Tool class pointer
		this->lbvh_face = &solver_helper.lbvh_face;
		this->lbvh_edge = &solver_helper.lbvh_edge;
		this->device_parallel = &solver_helper.device_parallel;
		this->buffer_filler = &solver_helper.buffer_filler;
		this->narrow_phase_detector = &solver_helper.narrow_phase_detector;
		pcg_solver = &solver_helper.pcg_solver;
	}
	void SolverInterface::init_data(luisa::compute::Device& device,
		luisa::compute::Stream&								stream)
	{
		set_data_pointer(solver_data, solver_helper);

		init_world_data();

		// Init data
		{
			lcs::Initializer::init_mesh_data(world_data, host_mesh_data);
			lcs::Initializer::upload_mesh_buffers(device, stream, host_mesh_data, mesh_data);
		}

		{
			lcs::Initializer::init_sim_data(world_data, host_mesh_data, host_sim_data);
			lcs::Initializer::upload_sim_buffers(device, stream, host_sim_data, sim_data);
			lcs::Initializer::resize_pcg_data(device, stream, host_mesh_data, host_sim_data, sim_data);
		}

		{
			lcs::Initializer::init_collision_data(world_data, host_mesh_data, host_sim_data, host_collision_data);
			lcs::Initializer::upload_collision_buffers(device, stream, host_sim_data, sim_data, host_collision_data, collision_data);
		}

		{
			lbvh_data_face->allocate(device, host_mesh_data->num_faces, lcs::LBVHTreeTypeFace);
			lbvh_data_edge->allocate(device, host_mesh_data->num_edges, lcs::LBVHTreeTypeEdge);
			lcs::Initializer::init_lbvh_data(device, stream, lbvh_data_face);
			lcs::Initializer::init_lbvh_data(device, stream, lbvh_data_edge);
			// lbvh_cloth_vert.unit_test(device, stream);
		}

		{
			host_collision_data->allocate_basic_buffers(device,
				host_mesh_data->num_verts,
				host_mesh_data->num_faces,
				host_mesh_data->num_edges,
				host_sim_data->num_dof);

			collision_data->allocate_basic_buffers(device,
				host_mesh_data->num_verts,
				host_mesh_data->num_faces,
				host_mesh_data->num_edges,
				host_sim_data->num_dof);

			host_collision_data->resize_collision_data_list(device,
				host_mesh_data->num_verts,
				host_mesh_data->num_faces,
				host_mesh_data->num_edges,
				host_sim_data->num_dof,
				false,
				true);

			collision_data->resize_collision_data_list(device,
				host_mesh_data->num_verts,
				host_mesh_data->num_faces,
				host_mesh_data->num_edges,
				host_sim_data->num_dof,
				true,
				true);
		}

		init_animation();
	}
	void SolverInterface::init_world_data()
	{
		const uint num_meshes = world_data.size();

		using namespace lcs::Initializer;
		using namespace lcs::Material;

		// Sort world data by material type, for better memory coherence in simulation and easier management
		{
			std::sort(world_data.begin(),
				world_data.end(),
				[](const Initializer::WorldData& left, const Initializer::WorldData& right)
				{
					if (left.material_type != right.material_type)
						return int(left.material_type) < int(right.material_type);
					return left.get_registration_index() < right.get_registration_index();
				});

			for (uint i = 0; i < num_meshes; i++)
			{
				world_data[i].sorted_index = i;
			}
		}

		// Pre-process materials
		for (uint meshIdx = 0; meshIdx < num_meshes; meshIdx++)
		{
			auto&		shell_info = world_data[meshIdx];
			const auto& input_mesh = shell_info.get_mesh();
			if (input_mesh.model_positions.empty())
			{
				LUISA_ERROR("Mesh {} has no vertex positions.", shell_info.get_model_name());
			}

			if (shell_info.material_type == MaterialType::Cloth)
			{
				if (!shell_info.holds<ClothMaterial>())
				{
					shell_info.set_physics_material(ClothMaterial());
				}
				auto& mat = shell_info.get_material<ClothMaterial>();
				mat.is_shell = true; // Cloth material must be shell
			}
			else if (shell_info.material_type == MaterialType::Tetrahedral)
			{
				if (!shell_info.holds<TetMaterial>())
				{
					shell_info.set_physics_material(TetMaterial());
				}
				auto& mat = shell_info.get_material<TetMaterial>();
				mat.is_shell = false; // Tetrahedral mesh must be solid
			}
			else if (shell_info.material_type == MaterialType::Rigid)
			{
				if (!shell_info.holds<RigidMaterial>())
				{
					shell_info.set_physics_material(RigidMaterial());
				}
				const bool has_boundary =
					input_mesh.dihedral_edges.size() != input_mesh.edges.size();

				auto& mat = shell_info.get_material<RigidMaterial>();
				mat.is_shell = !mat.is_solid;
				if (mat.is_shell)
				{
					if (has_boundary)
					{
						// TODO: Later we may construct a virtual volume mesh for shell
						LUISA_ERROR("Non-closed mesh simulation is currently not supported for rigid body ");
					}
				}
				else
				{
					if (has_boundary)
					{
						LUISA_ERROR("The solid mesh is not closed");
					}
					mat.thickness = 0.0f;
				}
			}
			else if (shell_info.material_type == MaterialType::Rod)
			{
				if (!shell_info.holds<RodMaterial>())
				{
					shell_info.set_physics_material(RodMaterial());
				}
				auto& mat = shell_info.get_material<RodMaterial>();
				mat.is_shell = true;
			}
		}
	}
	void SolverInterface::init_animation()
	{
		vid_to_animation_idx_map.clear();
		body_to_animation_idx_map.clear();
		per_vertex_animations.clear();
		per_body_animations.clear();

		//
		for (uint mesh_idx = 0; mesh_idx < host_mesh_data->num_meshes; mesh_idx++)
		{
			const auto& wd = world_data[mesh_idx];
			const uint	prefix_vid = host_mesh_data->prefix_num_verts[mesh_idx];
			const uint	suffix_vid = host_mesh_data->prefix_num_verts[mesh_idx + 1];

			if (!wd.fixed_point_indices.empty())
			{
				if (wd.holds<lcs::Material::RigidMaterial>())
				{
					const uint local_vid = wd.fixed_point_indices.front();
					const uint global_vid = prefix_vid + local_vid;
					const uint dof_idx = host_sim_data->sa_x_to_dof_map[global_vid].get_dof_idx();

					Animation::PerBodyAnimation body_anim;
					{
						const auto rest_t = wd.translation;
						const auto rest_r = wd.rotation;
						const auto rest_s = wd.scale;
						body_anim.dof_start = dof_idx;
						body_anim.set_translation(rest_t.x, rest_t.y, rest_t.z);
						body_anim.set_rotation(rest_r.x, rest_r.y, rest_r.z);
						body_anim.set_scale(1.0f, 1.0f, 1.0f); // Since we use |AAT-I|, we don't explicitly animate scale
					}
					body_to_animation_idx_map[mesh_idx] = per_body_animations.size();
					per_body_animations.push_back(body_anim);
				}
				else
				{
					for (uint index = 0; index < wd.fixed_point_indices.size(); index++)
					{
						const uint local_vid = wd.fixed_point_indices[index];
						const uint global_vid = prefix_vid + local_vid;
						const auto rest_pos = wd.get_rest_position(local_vid);

						Animation::PerVertexAnimation vertex_anim;
						{
							vertex_anim.vertex_id = global_vid;
							vertex_anim.set_translation(rest_pos.x, rest_pos.y, rest_pos.z);
						}
						vid_to_animation_idx_map[global_vid] = per_vertex_animations.size();
						per_vertex_animations.push_back(vertex_anim);
						// LUISA_INFO(" -> Init animation for mesh {}, local_vid {}, global_vid {}, rest_pos ({:.3f}, {:.3f}, {:.3f})",
						// 	mesh_idx, local_vid, global_vid, rest_pos[0], rest_pos[1], rest_pos[2]);
					}
				}
			}
		}
	}
	void SolverInterface::compile(AsyncCompiler& compiler)
	{
		compile_compute_energy(compiler);

		{
			lbvh_face->set_lbvh_data(lbvh_data_face);
			lbvh_edge->set_lbvh_data(lbvh_data_edge);
			lbvh_face->compile(compiler);
			lbvh_edge->compile(compiler);
		}

		LUISA_INFO("JIT Compiling Narrow Phase Detector...");
		{
			narrow_phase_detector->set_collision_data(host_collision_data, collision_data);
			narrow_phase_detector->compile(compiler);
			auto tmp_stream = compiler.device().create_stream();
			narrow_phase_detector->unit_test(compiler.device(), tmp_stream);
		}

		LUISA_INFO("JIT Compiling Solver...");
		{
			pcg_solver->set_data(host_mesh_data, mesh_data, host_sim_data, sim_data);
			pcg_solver->compile(compiler);
		}
	}

	template <typename T>
	static void buffer_copy(const std::vector<T>& src, std::vector<T>& dst)
	{
		if (src.size() != dst.size())
		{
			LUISA_ERROR("Buffer size mismatch {} != {}", src.size(), dst.size());
		}
		std::memcpy(dst.data(), src.data(), sizeof(T) * src.size());
	}
	void SolverInterface::physics_step_prev_operation()
	{
		// Set target position velocity for fixed verts
		for (uint index = 0; index < per_vertex_animations.size(); index++)
		{
			const auto&	 animate = per_vertex_animations[index];
			const uint	 vid = animate.vertex_id;
			const float3 curr_pos = host_sim_data->sa_q_outer[vid];
			const float3 target_pos =
				luisa::make_float3(animate.translation[0], animate.translation[1], animate.translation[2]);
			const float3 desire_vel = (target_pos - curr_pos) / get_scene_params().implicit_dt;
			host_sim_data->sa_q_v_outer[vid] = desire_vel;
		}

		for (uint index = 0; index < per_body_animations.size(); index++)
		{
			const auto& animate = per_body_animations[index];

			const uint dof_idx = animate.dof_start;

			auto	 transform = animate.to_transform_matrix();
			float4x3 target_q = AffineBodyDynamics::extract_q_from_affine_matrix(transform);

			float4x3 curr_q;
			curr_q[0] = host_sim_data->sa_q_outer[dof_idx + 0u];
			curr_q[1] = host_sim_data->sa_q_outer[dof_idx + 1u];
			curr_q[2] = host_sim_data->sa_q_outer[dof_idx + 2u];
			curr_q[3] = host_sim_data->sa_q_outer[dof_idx + 3u];

			auto desire_vel = (target_q - curr_q) / get_scene_params().implicit_dt;
			host_sim_data->sa_q_v_outer[dof_idx + 0u] = desire_vel[0];
			host_sim_data->sa_q_v_outer[dof_idx + 1u] = desire_vel[1];
			host_sim_data->sa_q_v_outer[dof_idx + 2u] = desire_vel[2];
			host_sim_data->sa_q_v_outer[dof_idx + 3u] = desire_vel[3];
		}

		buffer_copy(host_sim_data->sa_q_outer, host_sim_data->sa_q_step_start);
		buffer_copy(host_sim_data->sa_q_v_outer, host_sim_data->sa_q_v);
	}
	void SolverInterface::physics_step_post_operation()
	{
		buffer_copy(host_sim_data->sa_x, host_sim_data->sa_x_outer);
		buffer_copy(host_sim_data->sa_v, host_sim_data->sa_v_outer);
		buffer_copy(host_sim_data->sa_q, host_sim_data->sa_q_outer);
		buffer_copy(host_sim_data->sa_q_v, host_sim_data->sa_q_v_outer);
		get_scene_params().current_frame += 1;
	}

	void SolverInterface::restart_system()
	{
		buffer_copy(host_sim_data->sa_rest_x, host_sim_data->sa_x_outer);
		buffer_copy(host_sim_data->sa_rest_v, host_sim_data->sa_v_outer);
		buffer_copy(host_sim_data->sa_rest_q, host_sim_data->sa_q_outer);
		buffer_copy(host_sim_data->sa_rest_q_v, host_sim_data->sa_q_v_outer);
	}

	void SolverInterface::get_curr_vertices_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions)
	{
		const auto& sim_result_positions = host_sim_data->sa_x_outer;

		if (output_positions.size() != host_mesh_data->num_meshes)
		{
			output_positions.resize(host_mesh_data->num_meshes);
		}

		for (uint sortedIdx = 0; sortedIdx < host_mesh_data->num_meshes; sortedIdx++)
		{
			const uint meshIdx = host_mesh_data->sorted_to_input_mesh_id[sortedIdx];
			const uint prefix = host_mesh_data->prefix_num_verts[sortedIdx];
			const uint suffix = host_mesh_data->prefix_num_verts[sortedIdx + 1];
			const uint num_verts = suffix - prefix;
			if (output_positions[meshIdx].size() != num_verts)
			{
				output_positions[meshIdx].resize(num_verts);
			}
			for (uint vid = 0; vid < num_verts; vid++)
			{
				auto pos = sim_result_positions[prefix + vid];
				output_positions[meshIdx][vid] = { pos.x, pos.y, pos.z };
			}
		}
	}
	void SolverInterface::get_rest_vertices_to_host(std::vector<std::vector<std::array<float, 3>>>& output_positions)
	{
		const auto& rest_positions = host_mesh_data->sa_rest_x;

		if (output_positions.size() != host_mesh_data->num_meshes)
		{
			output_positions.resize(host_mesh_data->num_meshes);
		}

		for (uint sortedIdx = 0; sortedIdx < host_mesh_data->num_meshes; sortedIdx++)
		{
			const uint meshIdx = host_mesh_data->sorted_to_input_mesh_id[sortedIdx];
			const uint prefix = host_mesh_data->prefix_num_verts[sortedIdx];
			const uint suffix = host_mesh_data->prefix_num_verts[sortedIdx + 1];
			const uint num_verts = suffix - prefix;
			if (output_positions[meshIdx].size() != num_verts)
			{
				output_positions[meshIdx].resize(num_verts);
			}
			for (uint vid = 0; vid < num_verts; vid++)
			{
				auto pos = rest_positions[prefix + vid];
				output_positions[meshIdx][vid] = { pos.x, pos.y, pos.z };
			}
		}
	}
	void SolverInterface::get_triangles_to_host(std::vector<std::vector<std::array<uint, 3>>>& output_triangles)
	{
		const auto& triangles = host_mesh_data->sa_faces;

		if (output_triangles.size() != host_mesh_data->num_meshes)
		{
			output_triangles.resize(host_mesh_data->num_meshes);
		}

		for (uint sortedIdx = 0; sortedIdx < host_mesh_data->num_meshes; sortedIdx++)
		{
			const uint meshIdx = host_mesh_data->sorted_to_input_mesh_id[sortedIdx];
			const uint prefix_face = host_mesh_data->prefix_num_faces[sortedIdx];
			const uint suffix_face = host_mesh_data->prefix_num_faces[sortedIdx + 1];
			const uint num_faces = suffix_face - prefix_face;
			const uint prefix_vert = host_mesh_data->prefix_num_verts[sortedIdx];

			if (output_triangles[meshIdx].size() != num_faces)
			{
				output_triangles[meshIdx].resize(num_faces);
			}

			for (uint fid = 0; fid < num_faces; fid++)
			{
				auto tri = triangles[prefix_face + fid];
				output_triangles[meshIdx][fid] = {
					tri.x - prefix_vert,
					tri.y - prefix_vert,
					tri.z - prefix_vert,
				};
			}
		}
	}
	void SolverInterface::update_per_vertex_animation(const uint registerIdx,
		const uint												 local_vid,
		const std::array<float, 3>&								 target_position)
	{
		const uint sortedIdx = host_mesh_data->input_to_sorted_mesh_id[registerIdx];
		const uint prefix = host_mesh_data->prefix_num_verts[sortedIdx];
		const uint global_vid = prefix + local_vid;
		if (vid_to_animation_idx_map.contains(global_vid))
		{
			const uint animation_idx = vid_to_animation_idx_map[global_vid];
			per_vertex_animations[animation_idx].set_translation(target_position[0], target_position[1], target_position[2]);
		}
		else
		{
			LUISA_ERROR("Vertex {} in mesh {} (Sorted = {}) is not a pinned vertex. Cannot update position.", local_vid, registerIdx, sortedIdx);
		}
	}
	void SolverInterface::update_per_body_animation(const uint registerIdx,
		const std::array<float, 3>&							   target_translation,
		const std::array<float, 3>&							   target_rotation)
	{
		const uint sortedIdx = host_mesh_data->input_to_sorted_mesh_id[registerIdx];

		if (body_to_animation_idx_map.contains(sortedIdx))
		{
			const uint animation_idx = body_to_animation_idx_map[sortedIdx];
			per_body_animations[animation_idx].set_translation(target_translation[0], target_translation[1], target_translation[2]);
			per_body_animations[animation_idx].set_rotation(target_rotation[0], target_rotation[1], target_rotation[2]);
		}
		else
		{
			LUISA_ERROR("Mesh {} (SortedIdx = {}) is not a pinned rigid body. Cannot update state.", registerIdx, sortedIdx);
		}
	}
	void SolverInterface::update_default_animations()
	{
		const uint curr_frame = get_scene_params().current_frame;

		// Animation for fixed points
		for (uint sorted_idx = 0; sorted_idx < world_data.size(); sorted_idx++)
		{
			// Just sample code for animation, you can replace it with your own animation logic
			const float curr_time = curr_frame * lcs::get_scene_params().implicit_dt;
			auto&		wd = world_data[sorted_idx];
			const uint	register_idx = wd.get_registration_index();
			if (!wd.fixed_point_default_animations.empty())
			{
				if (wd.holds<Material::RigidMaterial>())
				{
					lcs::Animation::PerBodyAnimation tmp_body_animations;
					wd.update_default_body_animations(curr_time, tmp_body_animations);
					update_per_body_animation(register_idx, tmp_body_animations.translation, tmp_body_animations.rotation);
				}
				else
				{
					std::vector<lcs::Animation::PerVertexAnimation> tmp_vertex_animations;
					wd.update_default_vertex_animations(curr_time, tmp_vertex_animations);
					for (const auto& animate : tmp_vertex_animations)
					{
						update_per_vertex_animation(register_idx, animate.vertex_id, animate.translation);
					}
				}
			}
		}
	}

	void SolverInterface::save_current_frame_state_to_host(const std::string_view& full_path)
	{
		// save_current_frame_state();
		std::vector<float3> sa_q_frame_saved(host_sim_data->sa_q_outer);
		std::vector<float3> sa_qv_frame_saved(host_sim_data->sa_q_v_outer);

		std::ofstream file(std::string(full_path), std::ios::out);

		if (file.is_open())
		{
			file << "o State" << std::endl;
			for (uint vid = 0; vid < host_sim_data->num_dof; vid++)
			{
				const auto vertex = sa_q_frame_saved[vid];
				file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
			}
			file << "o Velocity" << std::endl;
			for (uint vid = 0; vid < host_sim_data->num_dof; vid++)
			{
				const auto vel = sa_qv_frame_saved[vid];
				file << "v " << vel.x << " " << vel.y << " " << vel.z << std::endl;
			}

			file.close();
			LUISA_INFO("State file saved: {}", full_path);
		}
		else
		{
			LUISA_ERROR("Unable to open file: {}", full_path);
		}
	}
	static float3 fn_apply_template(const std::vector<VertexToDofMap>& sa_x_to_dof_map,
		const std::vector<float3>&									   sa_scaled_model_x,
		const std::vector<float3>&									   input_q,
		const uint													   vid)
	{
		float3	   new_dx;
		const auto map_info = sa_x_to_dof_map[vid];
		const uint dof_idx = map_info.get_dof_idx();
		if (map_info.is_soft_body()) // Soft body
		{
			new_dx = input_q[dof_idx];
		}
		else // Rigid body
		{
			const float3 rest_x = sa_scaled_model_x[vid];
			float3		 p;
			float3x3	 A;
			p = input_q[dof_idx + 0];
			A[0] = input_q[dof_idx + 1];
			A[1] = input_q[dof_idx + 2];
			A[2] = input_q[dof_idx + 3];
			new_dx = A * rest_x + p; // Affine position
		};
		return new_dx;
	};

	uint SolverInterface::query_sorted_index_by_registration_id(uint registration_id) const
	{
		if (registration_id >= host_mesh_data->input_to_sorted_mesh_id.size())
		{
			LUISA_ERROR("Invalid registration_id {}. Out of range.", registration_id);
		}
		return host_mesh_data->input_to_sorted_mesh_id[registration_id];
	}

	uint SolverInterface::query_registration_id_by_sorted_index(uint sorted_index) const
	{
		if (sorted_index >= host_mesh_data->num_meshes)
		{
			LUISA_ERROR("Invalid sorted index {}. Out of range.", sorted_index);
		}
		return host_mesh_data->sorted_to_input_mesh_id[sorted_index];
	}

	static void fn_get_object_sim_result_by_sorted_index(const lcs::MeshData<std::vector>* host_mesh_data,
		const lcs::SimulationData<std::vector>*											   host_sim_data,
		const uint																		   sorted_idx,
		std::vector<std::array<float, 3>>&												   output_positions,
		std::vector<std::array<uint, 3>>&												   output_triangles)
	{
		if (sorted_idx >= host_mesh_data->num_meshes)
		{
			LUISA_ERROR("Invalid sorted index {}. Out of range.", sorted_idx);
		}

		const uint prefix_vert = host_mesh_data->prefix_num_verts[sorted_idx];
		const uint suffix_vert = host_mesh_data->prefix_num_verts[sorted_idx + 1];
		const uint num_verts = suffix_vert - prefix_vert;
		output_positions.resize(num_verts);
		for (uint vid = 0; vid < num_verts; vid++)
		{
			const auto pos = host_sim_data->sa_x_outer[prefix_vert + vid];
			output_positions[vid] = { pos.x, pos.y, pos.z };
		}

		const uint prefix_face = host_mesh_data->prefix_num_faces[sorted_idx];
		const uint suffix_face = host_mesh_data->prefix_num_faces[sorted_idx + 1];
		const uint num_faces = suffix_face - prefix_face;
		output_triangles.resize(num_faces);
		for (uint fid = 0; fid < num_faces; fid++)
		{
			const auto tri = host_mesh_data->sa_faces[prefix_face + fid];
			output_triangles[fid] = { tri.x - prefix_vert, tri.y - prefix_vert, tri.z - prefix_vert };
		}
	}

	void SolverInterface::get_object_sim_result_by_registration_id(uint registration_id,
		std::vector<std::array<float, 3>>&								output_positions,
		std::vector<std::array<uint, 3>>&								output_triangles)
	{
		const uint sorted_idx = query_sorted_index_by_registration_id(registration_id);
		fn_get_object_sim_result_by_sorted_index(host_mesh_data,
			host_sim_data,
			sorted_idx,
			output_positions,
			output_triangles);
	}

	void SolverInterface::load_saved_state_from_host(const std::string_view& full_path)
	{
		std::ifstream file(std::string(full_path), std::ios::in);
		if (!file.is_open())
		{
			LUISA_ERROR("Unable to open state file: {}", full_path);
			return;
		}

		// std::vector<float3> sa_x_frame_saved(host_mesh_data->sa_x_frame_outer.size());
		// std::vector<float3> sa_v_frame_saved(host_mesh_data->sa_v_frame_outer.size());
		std::vector<float3> sa_q_frame_saved(host_sim_data->sa_q_outer.size());
		std::vector<float3> sa_qv_frame_saved(host_sim_data->sa_q_v_outer.size());

		std::string line;
		enum Section
		{
			None,
			Q,
			Qv,
		};
		Section current_section = None;
		uint	index = 0;

		while (std::getline(file, line))
		{
			// LUISA_INFO("Reading line: {}", line);
			if (line.empty())
				continue;
			if (line.rfind("o State", 0) == 0)
			{
				current_section = Q;
				index = 0;
				continue;
			}
			if (line.rfind("o Velocity", 0) == 0)
			{
				current_section = Qv;
				index = 0;
				continue;
			}
			if (line[0] == 'v' && (current_section == Q || current_section == Qv))
			{
				std::istringstream iss(line.substr(1));
				float			   x, y, z;
				iss >> x >> y >> z;
				if (current_section == Q)
				{
					if (index < host_sim_data->num_dof)
						sa_q_frame_saved[index] = { x, y, z };
					else
					{
						LUISA_INFO("Count of loaded q vertices exceeds the number of state q in the sim data, stopping load.");
						file.close();
						return;
					}
					index++;
				}
				else if (current_section == Qv)
				{
					if (index < host_sim_data->num_dof)
						sa_qv_frame_saved[index] = { x, y, z };
					else
					{
						LUISA_INFO("Count of loaded qv vertices exceeds the number of state q_v in the sim data, stopping load.");
						file.close();
						return;
					}
					index++;
				}
			}
		}
		file.close();

		buffer_copy(sa_q_frame_saved, host_sim_data->sa_q_outer);
		buffer_copy(sa_qv_frame_saved, host_sim_data->sa_q_v_outer);

		CpuParallel::parallel_for(0,
			host_sim_data->num_verts_total,
			[&](uint vid)
			{
				float3 saved_x = fn_apply_template(host_sim_data->sa_x_to_dof_map,
					host_sim_data->sa_scaled_model_x,
					sa_q_frame_saved,
					vid);
				float3 saved_v = fn_apply_template(host_sim_data->sa_x_to_dof_map,
					host_sim_data->sa_scaled_model_x,
					sa_qv_frame_saved,
					vid);
				host_sim_data->sa_x_outer[vid] = saved_x;
				host_sim_data->sa_v_outer[vid] = saved_v;
			});

		LUISA_INFO("State file loaded: {}", full_path);
	}
	void SolverInterface::save_mesh_to_obj(const std::string_view& full_path)
	{
		const auto& position_buffer = host_sim_data->sa_x_outer;

		// Ensure the directory exists
		{
			std::filesystem::path file_directory = std::filesystem::path(full_path).parent_path();
			if (!std::filesystem::exists(file_directory))
			{
				try
				{
					std::filesystem::create_directories(file_directory);
					std::cout << "Created directory: " << file_directory.string() << std::endl;
				}
				catch (const std::filesystem::filesystem_error& e)
				{
					std::cerr << "Error creating directory: " << e.what() << std::endl;
					return;
				}
			}
		}

		std::ofstream file(std::string(full_path), std::ios::out);

		if (file.is_open())
		{
			file << "# Simulated Reulst" << std::endl;

			uint glocal_vert_id_prefix = 0;
			uint glocal_mesh_id_prefix = 0;

			// Cloth Part
			// if (lcs::get_scene_params().draw_cloth)
			{
				const uint num_clothes = host_mesh_data->prefix_num_verts.size() - 1;
				for (uint clothIdx = 0; clothIdx < num_clothes; clothIdx++)
				{
					const uint curr_prefix_num_verts = host_mesh_data->prefix_num_verts[clothIdx];
					const uint next_prefix_num_verts = host_mesh_data->prefix_num_verts[clothIdx + 1];
					const uint curr_prefix_num_faces = host_mesh_data->prefix_num_faces[clothIdx];
					const uint next_prefix_num_faces = host_mesh_data->prefix_num_faces[clothIdx + 1];

					{
						file << "o mesh_" << (glocal_mesh_id_prefix + clothIdx) << std::endl;
						for (uint vid = 0; vid < next_prefix_num_verts - curr_prefix_num_verts; vid++)
						{
							const auto vertex = position_buffer[curr_prefix_num_verts + vid];
							file << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
						}

						for (uint fid = 0; fid < next_prefix_num_faces - curr_prefix_num_faces; fid++)
						{
							const auto vid_prefix = glocal_vert_id_prefix + 1;
							const auto f = host_mesh_data->sa_faces[curr_prefix_num_faces + fid];
							file << "f " << vid_prefix + f.x << " " << vid_prefix + f.y << " "
								 << vid_prefix + f.z << std::endl;
						}
					}
				}
				glocal_vert_id_prefix += host_mesh_data->num_verts;
				glocal_mesh_id_prefix += 1;
			}

			file.close();
			std::cout << "OBJ file saved: " << full_path << std::endl;
		}
		else
		{
			std::cerr << "Unable to open file: " << full_path << std::endl;
		}
	}

	constexpr bool print_detail = false;

	void SolverInterface::compile_compute_energy(AsyncCompiler& compiler)
	{
		using namespace luisa::compute;
		const bool					 use_debug_info = false;
		luisa::compute::ShaderOption default_option = { .enable_debug_info = false };

		compiler.compile<1>(fn_reset_float,
			[](Var<BufferView<float>> buffer)
			{ buffer->write(dispatch_x(), 0.0f); });

		// instantiate energy objects and let them register their shaders
		inertia_energy =
			std::make_unique<SoftInertiaEnergy>(sim_data->sa_q_tilde.view(), sim_data->sa_system_energy.view());
		inertia_energy->compile(compiler);

		ground_collision_energy =
			std::make_unique<GroundCollisionEnergy>(mesh_data->sa_rest_vert_area.view(),
				mesh_data->sa_is_fixed.view(),
				sim_data->sa_contact_active_verts_offset.view(),
				sim_data->sa_contact_active_verts_d_hat.view(),
				sim_data->sa_contact_active_verts_friction_coeff.view(),
				sim_data->sa_x_step_start.view(),
				sim_data->sa_x.view(),
				sim_data->sa_scaled_model_x.view(),
				sim_data->sa_x_to_dof_map.view(),
				sim_data->sa_system_energy.view());
		ground_collision_energy->compile(compiler);

		spring_energy = std::make_unique<SpringEnergy>(sim_data->sa_system_energy.view());
		spring_energy->compile(compiler);

		stretch_face_energy = std::make_unique<StretchFaceEnergy>(sim_data->sa_system_energy.view());
		stretch_face_energy->compile(compiler);

		bending_energy = std::make_unique<BendingEnergy>(sim_data->sa_system_energy.view());
		bending_energy->compile(compiler);

		abd_inertia_energy =
			std::make_unique<AbdInertiaEnergy>(sim_data->sa_q_tilde.view(), sim_data->sa_system_energy.view());
		abd_inertia_energy->compile(compiler);

		abd_ortho_energy = std::make_unique<AbdOrthoEnergy>(sim_data->sa_system_energy.view(), sim_data->sa_q.view());
		abd_ortho_energy->compile(compiler);
	}
	void SolverInterface::device_compute_elastic_energy(luisa::compute::Stream& stream,
		std::map<std::string, double>&											energy_list)
	{
		const luisa::compute::Buffer<float3>& curr_x = sim_data->sa_x;
		const luisa::compute::Buffer<float3>& curr_q = sim_data->sa_q;

		stream << fn_reset_float(sim_data->sa_system_energy).dispatch(8);

		const auto& soft_inertia_constitution = sim_data->get_soft_inertia_data();
		if (soft_inertia_constitution.is_valid())
		{
			inertia_energy->device_compute_energy(
				stream, soft_inertia_constitution, curr_x, get_scene_params().get_substep_dt(), host_sim_data->num_verts_soft);
		}

		const auto& abd_inertia_data = sim_data->get_abd_inertia_data();
		if (abd_inertia_data.is_valid())
		{
			abd_inertia_energy->device_compute_energy(
				stream, abd_inertia_data, curr_q, get_scene_params().get_substep_dt(), abd_inertia_data.get_num_indices());
		}

		const auto& abd_ortho_data = sim_data->get_abd_orthogonality_data();
		if (abd_ortho_data.is_valid())
		{
			abd_ortho_energy->device_compute_energy(stream, abd_ortho_data, curr_q, abd_ortho_data.get_num_indices());
		}

		if (get_scene_params().use_floor)
		{
			ground_collision_energy->device_compute_energy(stream,
				curr_x,
				get_scene_params().floor.y,
				get_scene_params().use_floor,
				get_scene_params().stiffness_collision,
				get_scene_params().contact_energy_type,
				mesh_data->num_verts);
		}

		const auto& stretch_spring_constitution = sim_data->get_stretch_spring_data();
		if (stretch_spring_constitution.is_valid())
		{
			spring_energy->device_compute_energy(
				stream, stretch_spring_constitution, curr_x, stretch_spring_constitution.get_num_indices());
		}

		const auto& stretch_face_constitution = sim_data->get_stretch_face_data();
		if (stretch_face_constitution.is_valid())
		{
			stretch_face_energy->device_compute_energy(
				stream, stretch_face_constitution, curr_x, stretch_face_constitution.get_num_indices());
		}

		const auto& bending_edge_constitution = sim_data->get_bending_edge_data();
		if (bending_edge_constitution.is_valid())
		{
			bending_energy->device_compute_energy(stream,
				bending_edge_constitution,
				curr_x,
				get_scene_params().get_bending_stiffness_scaling(),
				bending_edge_constitution.get_num_indices());
		}

		auto& host_energy = host_sim_data->sa_system_energy;
		stream << sim_data->sa_system_energy.view(0, 8).copy_to(host_energy.data()) << luisa::compute::synchronize();

		energy_list.insert(std::make_pair("Inertia Soft Body", host_energy[offset_inertia]));
		energy_list.insert(std::make_pair("Inertia Rigid Body", host_energy[offset_abd_inertia]));
		energy_list.insert(std::make_pair("Ground Collision", host_energy[offset_ground_collision]));
		energy_list.insert(std::make_pair("Ground Friction", host_energy[offset_ground_friction]));
		energy_list.insert(std::make_pair("Stretch Spring", host_energy[offset_stretch_spring]));
		energy_list.insert(std::make_pair("Stretch Face", host_energy[offset_stretch_face]));
		energy_list.insert(std::make_pair("Cloth Bending", host_energy[offset_bending]));
		energy_list.insert(std::make_pair("ABD Orthogonality", host_energy[offset_abd_ortho]));
	};

	// ---------------------------------------------------------------------------
	// Device management methods
	// ---------------------------------------------------------------------------

	void SolverInterface::create_device(const std::string& binary_path, const std::string& backend_name)
	{
		if (device_state.initialized)
			throw std::runtime_error("Device already initialized. Call cleanup_device() first"
									 "or use set_device_from_pointers() to set external device and stream.");

		luisa::log_level_info();

		LUISA_INFO("Creating luisa compute context/device/stream...");

		device_state.owned_context = std::make_unique<luisa::compute::Context>(binary_path);

		std::string backend = backend_name;
		if (backend.empty())
		{
#if defined(__APPLE__)
			backend = "metal";
#elif defined(_WIN32)
			backend = "dx";
#else
			backend = "cuda";
#endif
		}

		luisa::vector<luisa::string> device_names = device_state.owned_context->backend_device_names(backend);
		if (device_names.empty())
		{
			LUISA_WARNING("No hardware device found for backend '{}'.", backend);
			throw std::runtime_error("No hardware device found for backend: " + backend);
		}
		for (size_t i = 0; i < device_names.size(); ++i)
			LUISA_INFO("Device {}: {}", i, device_names[i]);

		try
		{
			auto dev = device_state.owned_context->create_device(backend, nullptr, false);
			device_state.owned_device = std::make_unique<luisa::compute::Device>(std::move(dev));
			auto st = device_state.owned_device->create_stream(luisa::compute::StreamTag::COMPUTE);
			device_state.owned_stream = std::make_unique<luisa::compute::Stream>(std::move(st));

			device_state.device = device_state.owned_device.get();
			device_state.stream = device_state.owned_stream.get();
			device_state.initialized = true;
			device_state.owns_resources = true;
		}
		catch (const std::exception& e)
		{
			throw std::runtime_error(std::string("Failed to create luisa device: ") + e.what());
		}
	}

	void SolverInterface::set_device_from_pointers(uintptr_t device_ptr, uintptr_t stream_ptr)
	{
		if (device_ptr == 0 || stream_ptr == 0)
			throw std::runtime_error("device_ptr and stream_ptr must be non-null.");
		if (device_state.initialized)
			throw std::runtime_error("Device already initialized. Call cleanup_device() first.");

		device_state.device = reinterpret_cast<luisa::compute::Device*>(device_ptr);
		device_state.stream = reinterpret_cast<luisa::compute::Stream*>(stream_ptr);
		device_state.initialized = true;
		device_state.owns_resources = false;
	}

	void SolverInterface::cleanup_device()
	{
		device_state.cleanup();
	}

	uintptr_t SolverInterface::get_device_ptr() const
	{
		return reinterpret_cast<uintptr_t>(device_state.device);
	}

	uintptr_t SolverInterface::get_stream_ptr() const
	{
		return reinterpret_cast<uintptr_t>(device_state.stream);
	}

} // namespace lcs